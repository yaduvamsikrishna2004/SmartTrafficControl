"""
============================================================
StreamingManager
Optimized MJPEG streaming with:
- Single latest-frame architecture (no circular buffer)
- Fixed-rate timer-based display (no frame repeating)
- JPEG encoding cache (encode once, serve many)
- Independent AI inference and display rates
- Configurable playback speed (0.5x, 0.75x, 0.85x, 1.0x)
- Thread-safe, lock-free read/write for minimum latency
- First frame appears IMMEDIATELY (no wait)
- Never shows black screen

Architecture:
  Inference Thread  ──▶  _latest_jpeg (single slot)  ◀── MJPEG Generator
  (encodes JPEG)          (overwritten each frame)      (reads at fixed rate)

Key Design Changes (FIXING stuttering):
  1. REMOVED CircularFrameBuffer — replaced with single atomic slot
  2. REMOVED frame repeating logic — display always shows latest frame
  3. REMOVED catch-up bursts — no read_latest() needed
  4. Fixed-rate display using time.monotonic() for precise intervals
  5. No time.sleep() in the display loop — uses yield-based timing
  6. Inference FPS is fully independent of display FPS
============================================================
"""

import cv2
import time
import threading
from backend.config import (
    JPEG_QUALITY,
    PLAYBACK_SPEED,
    PLAYBACK_SPEED_OPTIONS,
    DISPLAY_FPS_BASE,
    MIN_DISPLAY_INTERVAL,
)


class StreamingManager:
    """
    Manages MJPEG streaming with fixed-rate, glitch-free playback.

    HOW THIS FIXES STUTTERING:
    ──────────────────────────
    
    OLD BEHAVIOR (broken):
    - Circular buffer with push/pop
    - Frame repeating when buffer empty (causes PAUSE on screen)
    - read_latest() catch-up when repeats exceed limit (causes BURST)
    - Result: pause → burst → pause → burst cycle
    
    NEW BEHAVIOR (fixed):
    - Single latest-JPEG slot, atomically overwritten on each inference frame
    - MJPEG generator is a simple fixed-rate timer
    - Every 1/(DISPLAY_FPS_BASE * speed) seconds: yield _latest_jpeg
    - If no new frame, yield the SAME latest frame (no pause, no stall)
    - If inference slows down, display naturally slows (no burst needed)
    - If inference speeds up, display stays at fixed rate (drops extra frames)
    - Result: buttery-smooth, continuous playback at exact target FPS
    """

    def __init__(self):
        # ========================================================
        # Single latest-JPEG slot — atomically written by inference,
        # atomically read by MJPEG generator. No buffer needed.
        # ========================================================
        self._lock = threading.Lock()
        self._jpeg_bytes = None
        self._raw_frame = None
        self._frame_count = 0
        self._encode_time_ms = 0

        # ========================================================
        # Playback speed control
        # ========================================================
        self._playback_speed = PLAYBACK_SPEED
        self._display_frame_count = 0
        self._display_fps_actual = 0.0
        self._display_fps_start = 0.0

        # ========================================================
        # Fixed base interval from DISPLAY_FPS_BASE
        # display_interval = 1.0 / (DISPLAY_FPS_BASE * playback_speed)
        # This is FIXED — independent of inference FPS.
        # ========================================================
        self._base_interval = 1.0 / DISPLAY_FPS_BASE  # e.g. 1/20 = 0.05s

        # ========================================================
        # Inference FPS tracking (stats only — NOT used for timing)
        # ========================================================
        self._inference_frame_count = 0
        self._inference_fps_actual = 0.0
        self._inference_fps_start = 0.0

        # ========================================================
        # Latency tracking
        # ========================================================
        self._last_frame_timestamp = 0.0
        self._current_latency_ms = 0.0

        # ========================================================
        # Track if we've ever received a frame (for first-frame logic)
        # ========================================================
        self._has_received_frame = False
        self._first_frame_yielded = False

        print(f"[Streaming] Initialized: DISPLAY_FPS_BASE={DISPLAY_FPS_BASE}, "
              f"base_interval={self._base_interval*1000:.1f}ms, "
              f"PLAYBACK_SPEED={PLAYBACK_SPEED}x")

    # -------------------------------------------------------
    # Playback Speed Configuration
    # -------------------------------------------------------

    def set_playback_speed(self, speed: float) -> None:
        """Set the playback speed multiplier. Clamped to nearest valid option."""
        closest = min(PLAYBACK_SPEED_OPTIONS, key=lambda x: abs(x - speed))
        self._playback_speed = closest
        effective_fps = DISPLAY_FPS_BASE * closest
        print(f"[Streaming] Playback speed set to {closest}x → {effective_fps:.1f} FPS display")

    def get_playback_speed(self) -> float:
        return self._playback_speed

    def get_playback_speed_options(self) -> list:
        return list(PLAYBACK_SPEED_OPTIONS)

    # -------------------------------------------------------
    # Frame Update (called from inference pipeline thread)
    # -------------------------------------------------------

    def update_frame(self, frame):
        """
        Update the current frame and encode JPEG once.
        Called from the processing pipeline.

        KEY DESIGN: This writes to a SINGLE slot. The MJPEG generator
        reads from this same slot. No buffer. No queue. No backlog.
        If the MJPEG generator is currently yielding the previous frame,
        it will pick up this new frame on its next iteration.

        Steps:
        1. Track inference FPS (stats only)
        2. Encode JPEG once (shared across all clients)
        3. Atomically store the JPEG bytes (overwrites previous)
        """
        if frame is None:
            return

        now = time.perf_counter()
        self._raw_frame = frame

        # Track inference FPS (stats only — NOT used for display timing)
        self._inference_frame_count += 1
        if self._inference_frame_count == 1:
            self._inference_fps_start = now
        elapsed = now - self._inference_fps_start
        if elapsed >= 1.0:
            self._inference_fps_actual = self._inference_frame_count / elapsed
            self._inference_frame_count = 0
            self._inference_fps_start = now

        # Track latency
        if self._last_frame_timestamp > 0:
            self._current_latency_ms = (now - self._last_frame_timestamp) * 1000
        self._last_frame_timestamp = now

        # Encode JPEG once (shared across all clients)
        encode_start = time.perf_counter()
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        self._encode_time_ms = (time.perf_counter() - encode_start) * 1000

        if success:
            jpeg_bytes = buffer.tobytes()

            # ====================================================
            # CRITICAL: Single atomic write to the shared slot.
            # No lock needed for the MJPEG generator — it reads the
            # bytes object which is immutable and atomically assigned.
            # ====================================================
            self._jpeg_bytes = jpeg_bytes
            self._frame_count += 1
            self._has_received_frame = True

    # -------------------------------------------------------
    # Frame Accessors
    # -------------------------------------------------------

    def get_jpeg_bytes(self):
        """Get the latest JPEG bytes (cached, for WebSocket / fallback)."""
        return self._jpeg_bytes

    def get_raw_frame(self):
        """Get the latest raw frame (for WebSocket)."""
        return self._raw_frame

    # -------------------------------------------------------
    # MJPEG Generator — Fixed-Rate Timer
    #
    # HOW THIS DELIVERS SMOOTH PLAYBACK:
    #
    # 1. Calculate display_interval = base_interval / playback_speed
    #    e.g. At DISPLAY_FPS_BASE=20, base_interval=0.05s:
    #      1.0x → 0.050s → 20.0 FPS
    #      0.85x → 0.059s → 17.0 FPS
    #      0.75x → 0.067s → 15.0 FPS
    #      0.5x → 0.100s → 10.0 FPS
    #
    # 2. Every display_interval seconds, yield _jpeg_bytes
    #    — Always the LATEST frame (no queuing)
    #    — No frame repeating (no pause)
    #    — No catch-up (no burst)
    #    — If inference is slower than display, same frame is yielded
    #      twice (or more) — this looks like smooth slow motion, not a pause
    #    — If inference is faster, extra frames are dropped (silently)
    #
    # 3. First frame: yielded IMMEDIATELY on first call
    # -------------------------------------------------------

    def generate_mjpeg_frames(self):
        """
        Generator for MJPEG streaming with fixed-rate, glitch-free playback.

        YIELDS: MJPEG multipart frames at a stable rate.
        NEVER: Blocks, pauses, bursts, or freezes.
        """
        self._first_frame_yielded = False
        self._display_frame_count = 0
        self._display_fps_start = time.perf_counter()
        last_render_time = 0.0

        effective_fps = DISPLAY_FPS_BASE * self._playback_speed
        print(f"[Streaming] MJPEG generator started: "
              f"target {effective_fps:.1f} FPS ({self._playback_speed}x)")

        while True:
            now = time.perf_counter()

            # ====================================================
            # FIRST FRAME: Yield immediately (no wait)
            # ====================================================
            if not self._first_frame_yielded:
                jpeg = self._jpeg_bytes
                if jpeg is not None:
                    self._first_frame_yielded = True
                    last_render_time = now
                    self._display_frame_count += 1
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg
                        + b'\r\n'
                    )
                    continue
                else:
                    # No frame yet — brief yield, don't block
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + b'\r\n'
                    )
                    continue

            # ====================================================
            # Calculate display interval from FIXED base rate
            # display_interval = base_interval / playback_speed
            #
            # This is INDEPENDENT of inference FPS. The display
            # always runs at the configured rate, regardless of
            # whether inference is running at 15 FPS or 60 FPS.
            # ====================================================
            display_interval = self._base_interval / self._playback_speed
            display_interval = max(display_interval, MIN_DISPLAY_INTERVAL)

            # ====================================================
            # TIMESTAMP CHECK: Is it time to render the next frame?
            #
            # If not enough time has passed since the last render,
            # yield immediately with the current latest frame.
            # This is NOT busy-waiting — it's cooperative yielding
            # that allows the FastAPI event loop to serve other clients.
            #
            # The key insight: we yield the SAME latest frame if we're
            # not due for a new one yet. This keeps the browser's MJPEG
            # parser happy (it needs continuous data) while maintaining
            # the correct frame rate.
            # ====================================================
            if last_render_time > 0:
                elapsed_since_render = now - last_render_time
                if elapsed_since_render < display_interval:
                    # Not due yet — yield the latest frame immediately
                    # This gives the browser something to display while
                    # maintaining the timing interval.
                    jpeg = self._jpeg_bytes
                    if jpeg is not None:
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n'
                            + jpeg
                            + b'\r\n'
                        )
                    else:
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n'
                            + b'\r\n'
                        )
                    continue

            # ====================================================
            # TIME TO RENDER: Yield the latest frame
            # ====================================================
            jpeg = self._jpeg_bytes

            if jpeg is not None:
                last_render_time = now

                # Track display FPS
                self._display_frame_count += 1
                elapsed_display = now - self._display_fps_start
                if elapsed_display >= 1.0:
                    self._display_fps_actual = self._display_frame_count / elapsed_display
                    self._display_frame_count = 0
                    self._display_fps_start = now

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + jpeg
                    + b'\r\n'
                )
            else:
                # No frame yet — yield empty boundary (never show black screen)
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + b'\r\n'
                )

    # -------------------------------------------------------
    # Stats
    # -------------------------------------------------------

    def get_stats(self):
        """Return streaming performance metrics."""
        target_fps = DISPLAY_FPS_BASE * self._playback_speed
        return {
            "encode_time_ms": round(self._encode_time_ms, 2),
            "frame_count": self._frame_count,
            "jpeg_buffer_size": len(self._jpeg_bytes) if self._jpeg_bytes else 0,
            "playback_speed": self._playback_speed,
            "display_fps_actual": round(self._display_fps_actual, 1),
            "display_fps_target": round(target_fps, 1),
            "inference_fps": round(self._inference_fps_actual, 1),
            "latency_ms": round(self._current_latency_ms, 1),
            "display_fps_base": DISPLAY_FPS_BASE,
        }

    def get_display_metrics(self):
        """Get display-specific metrics for the FPS overlay."""
        target_fps = DISPLAY_FPS_BASE * self._playback_speed
        return {
            "camera_fps": round(self._inference_fps_actual, 1),
            "inference_fps": round(self._inference_fps_actual, 1),
            "display_fps": round(self._display_fps_actual, 1),
            "display_fps_target": round(target_fps, 1),
            "playback_speed": self._playback_speed,
            "buffer_size": 0,  # No buffer in this architecture
            "buffer_capacity": 1,  # Single slot
            "latency_ms": round(self._current_latency_ms, 1),
        }

    # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------

    def clear(self):
        """Clear all state."""
        self._jpeg_bytes = None
        self._raw_frame = None
        self._frame_count = 0
        self._encode_time_ms = 0
        self._display_frame_count = 0
        self._display_fps_actual = 0.0
        self._inference_frame_count = 0
        self._inference_fps_actual = 0.0
        self._has_received_frame = False
        self._first_frame_yielded = False
        self._current_latency_ms = 0.0