"""
============================================================
StreamingManager
Optimized MJPEG streaming with:
- Circular frame buffer for slow-motion playback
- Timestamp-based playback scheduler with precise timing
- JPEG encoding cache (encode once, serve many)
- Independent AI inference and display rates
- Frame repeating for smooth slow motion (no frame skipping)
- Configurable playback speed (0.25x, 0.5x, 0.75x, 1.0x)
- FIX: First frame appears IMMEDIATELY (no wait for buffer)
- FIX: Fallback to latest cached JPEG when buffer is empty
- FIX: Never shows black screen — always yields something

Architecture:
  Inference Pipeline  ──▶  CircularFrameBuffer  ──▶  MJPEG Generator
  (full speed)              (stores N frames)        (playback speed)

Key Design:
- AI inference runs at full speed (no blocking)
- Display reads from buffer at PLAYBACK_SPEED rate
- Frame repeating for smooth slow motion (no frame skipping)
- Circular buffer prevents unbounded memory growth
- No time.sleep() in the inference pipeline
- Emergency alerts/notifications remain real-time
============================================================
"""

import cv2
import time
import threading
from collections import namedtuple
from backend.config import (
    JPEG_QUALITY,
    PLAYBACK_SPEED,
    FRAME_BUFFER_SIZE,
    PLAYBACK_SPEED_OPTIONS,
    STREAMING_FRAME_TIMEOUT,
    MIN_DISPLAY_INTERVAL,
)


# A single entry in the circular frame buffer
FrameEntry = namedtuple("FrameEntry", ["jpeg_bytes", "frame_id", "timestamp"])


class CircularFrameBuffer:
    """
    Thread-safe circular buffer for processed video frames.

    - Fixed capacity (configurable via FRAME_BUFFER_SIZE)
    - Writer (inference pipeline) pushes frames continuously
    - Reader (MJPEG generator) pops frames at playback rate
    - Oldest frames are overwritten when buffer is full
    - Never blocks the writer
    """

    def __init__(self, capacity: int = FRAME_BUFFER_SIZE):
        self._capacity = capacity
        self._buffer = [None] * capacity
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0          # number of readable frames
        self._total_written = 0  # monotonic counter for frame identification
        self._lock = threading.Lock()

    # -------------------------------------------------------
    # Writer API (called from inference pipeline)
    # -------------------------------------------------------

    def push(self, jpeg_bytes: bytes, frame_id: int) -> None:
        """
        Push a new frame into the buffer.
        Never blocks. Overwrites oldest frame if buffer is full.
        """
        entry = FrameEntry(jpeg_bytes, frame_id, time.perf_counter())
        with self._lock:
            self._buffer[self._write_idx] = entry
            self._write_idx = (self._write_idx + 1) % self._capacity
            self._total_written += 1
            if self._count < self._capacity:
                self._count += 1
            else:
                # Buffer full — advance read pointer past the overwritten frame
                self._read_idx = (self._read_idx + 1) % self._capacity

    # -------------------------------------------------------
    # Reader API (called from MJPEG generator)
    # -------------------------------------------------------

    def read(self):
        """
        Read the next frame from the buffer and advance the read pointer.
        Returns FrameEntry or None if buffer is empty.
        """
        with self._lock:
            if self._count == 0:
                return None
            entry = self._buffer[self._read_idx]
            self._read_idx = (self._read_idx + 1) % self._capacity
            self._count -= 1
            return entry

    def peek(self):
        """
        Peek at the next frame without advancing the read pointer.
        Returns FrameEntry or None if buffer is empty.
        """
        with self._lock:
            if self._count == 0:
                return None
            return self._buffer[self._read_idx]

    def read_latest(self):
        """
        Read the LATEST frame from the buffer, skipping all older frames.
        This is used when the display is too far behind and needs to catch up.
        Returns FrameEntry or None if buffer is empty.
        """
        with self._lock:
            if self._count == 0:
                return None
            # Advance read pointer to the latest written frame
            latest_idx = (self._write_idx - 1) % self._capacity
            if self._buffer[latest_idx] is None:
                return None
            entry = self._buffer[latest_idx]
            # Reset buffer state - all frames consumed
            self._read_idx = self._write_idx
            self._count = 0
            return entry

    # -------------------------------------------------------
    # Status
    # -------------------------------------------------------

    def size(self) -> int:
        """Number of frames currently in the buffer."""
        with self._lock:
            return self._count

    def is_empty(self) -> bool:
        return self.size() == 0

    def is_full(self) -> bool:
        return self.size() >= self._capacity

    def capacity(self) -> int:
        return self._capacity

    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------

    def clear(self):
        with self._lock:
            self._buffer = [None] * self._capacity
            self._write_idx = 0
            self._read_idx = 0
            self._count = 0
            self._total_written = 0


class StreamingManager:
    """
    Manages MJPEG streaming with:
    - Circular frame buffer for slow-motion playback
    - Timestamp-based playback scheduler with precise timing
    - JPEG encoding cache (encode once, serve many)
    - Independent AI inference and display rate tracking
    - Configurable playback speed (0.25x, 0.5x, 0.75x, 1.0x)
    - Frame repeating for smooth slow motion (no frame skipping)
    - Automatic catch-up if display falls behind
    - FIX: First frame appears IMMEDIATELY
    - FIX: Fallback to latest cached JPEG when buffer empty
    - FIX: Never shows black screen
    """

    def __init__(self):
        # ========================================================
        # Core state
        # ========================================================
        self._lock = threading.Lock()
        self._jpeg_buffer = None
        self._jpeg_bytes = None
        self._raw_frame = None
        self._frame_count = 0
        self._encode_time_ms = 0

        # ========================================================
        # Circular frame buffer for slow-motion playback
        # ========================================================
        self._frame_buffer = CircularFrameBuffer(FRAME_BUFFER_SIZE)

        # ========================================================
        # Playback speed control
        # ========================================================
        self._playback_speed = PLAYBACK_SPEED
        self._last_render_time = 0.0
        self._display_frame_count = 0
        self._display_fps_actual = 0.0
        self._display_fps_start = 0.0

        # ========================================================
        # Inference FPS tracking (how fast AI produces frames)
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
        # Frame repeat tracking for smooth slow motion
        # ========================================================
        self._last_read_entry = None  # last frame read from buffer
        self._repeat_count = 0        # how many times we've repeated it
        self._max_repeats = 10        # max repeats before forcing a new frame

        # ========================================================
        # Smoothing: track the last few display intervals
        # to prevent jitter from variable inference FPS
        # ========================================================
        self._interval_history = []
        self._max_interval_history = 10

        # ========================================================
        # FIX: Track if we've ever received a frame
        # Used to yield the first frame immediately
        # ========================================================
        self._has_received_frame = False
        self._first_frame_yielded = False

    # -------------------------------------------------------
    # Playback Speed Configuration
    # -------------------------------------------------------

    def set_playback_speed(self, speed: float) -> None:
        """
        Set the playback speed multiplier.
        Clamped to the nearest valid option.
        """
        # Find the closest valid option
        closest = min(PLAYBACK_SPEED_OPTIONS, key=lambda x: abs(x - speed))
        self._playback_speed = closest
        print(f"[Streaming] Playback speed set to {closest}x")

    def get_playback_speed(self) -> float:
        """Get the current playback speed multiplier."""
        return self._playback_speed

    def get_playback_speed_options(self) -> list:
        """Get the list of available playback speed options."""
        return list(PLAYBACK_SPEED_OPTIONS)

    # -------------------------------------------------------
    # Frame Update (called from inference pipeline)
    # -------------------------------------------------------

    def update_frame(self, frame):
        """
        Update the current frame and encode JPEG once.
        Called from the processing pipeline.

        The frame is:
        1. JPEG-encoded once (shared across all clients)
        2. Pushed into the circular buffer for delayed playback
        3. Also cached as the latest raw frame for WebSocket clients
        """
        if frame is None:
            return

        now = time.perf_counter()
        self._raw_frame = frame

        # Track inference FPS
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
        start = time.perf_counter()
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        elapsed = (time.perf_counter() - start) * 1000
        self._encode_time_ms = elapsed

        if success:
            jpeg_bytes = buffer.tobytes()

            # Cache the latest JPEG for direct access (WebSocket, etc.)
            with self._lock:
                self._jpeg_buffer = buffer
                self._jpeg_bytes = jpeg_bytes
                self._has_received_frame = True

            # Push into circular buffer for slow-motion playback
            self._frame_count += 1
            self._frame_buffer.push(jpeg_bytes, self._frame_count)

    # -------------------------------------------------------
    # Frame Accessors
    # -------------------------------------------------------

    def get_jpeg_bytes(self):
        """Get the latest JPEG bytes (cached, for WebSocket)."""
        with self._lock:
            return self._jpeg_bytes

    def get_raw_frame(self):
        """Get the latest raw frame (for WebSocket)."""
        return self._raw_frame

    # -------------------------------------------------------
    # MJPEG Generator with Slow-Motion Playback
    # -------------------------------------------------------

    def generate_mjpeg_frames(self):
        """
        Generator for MJPEG streaming with slow-motion playback.

        FIX: First frame appears IMMEDIATELY.
        FIX: Falls back to latest cached JPEG when buffer is empty.
        FIX: Never shows black screen.

        Playback scheduler logic:
        1. Calculate display interval based on inference FPS and playback speed
           display_interval = 1.0 / (inference_fps * playback_speed)
        2. Read frames from the circular buffer at this interval
        3. If buffer is empty (startup), use the latest cached JPEG immediately
        4. If buffer has frames, read and yield them
        5. Frame repeating: if no new frame is available, repeat the last one
           (this creates smooth slow motion without frame skipping)
        6. Smoothing: average the last few intervals to prevent jitter
        """
        self._last_render_time = 0.0
        self._display_frame_count = 0
        self._display_fps_start = time.perf_counter()
        self._last_read_entry = None
        self._repeat_count = 0
        self._interval_history = []
        self._first_frame_yielded = False

        print("[Streaming] MJPEG generator started")

        while True:
            now = time.perf_counter()

            # ====================================================
            # FIX: Yield the first available frame IMMEDIATELY
            # without waiting for the buffer or timing interval.
            # ====================================================
            if not self._first_frame_yielded:
                # Check if we have a cached JPEG
                cached = self.get_jpeg_bytes()
                if cached is not None:
                    self._first_frame_yielded = True
                    self._last_render_time = now
                    self._display_frame_count += 1
                    print(f"[Streaming] Yielding FIRST frame immediately ({len(cached)} bytes)")
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + cached
                        + b'\r\n'
                    )
                    continue
                else:
                    # No frame yet — brief sleep and retry
                    time.sleep(0.005)
                    continue

            # ====================================================
            # Calculate display interval
            # display_interval = 1.0 / (inference_fps * playback_speed)
            #
            # FIX: Use a minimum of 10 FPS at startup so the first
            # few frames don't take 2 seconds to appear.
            # ====================================================
            inf_fps = max(self._inference_fps_actual, 10.0)  # FIX: was 1.0, now 10.0
            display_interval = 1.0 / (inf_fps * self._playback_speed)

            # Clamp to minimum interval to prevent excessive frame rates
            display_interval = max(display_interval, MIN_DISPLAY_INTERVAL)

            # ====================================================
            # Smooth the display interval using a moving average
            # This prevents jitter from variable inference FPS
            # ====================================================
            self._interval_history.append(display_interval)
            if len(self._interval_history) > self._max_interval_history:
                self._interval_history.pop(0)
            smoothed_interval = sum(self._interval_history) / len(self._interval_history)

            # ====================================================
            # Timestamp check: is it time to render?
            # ====================================================
            if self._last_render_time > 0:
                elapsed_since_render = now - self._last_render_time
                if elapsed_since_render < smoothed_interval:
                    # Not yet time — brief yield to avoid busy-wait
                    remaining = smoothed_interval - elapsed_since_render
                    sleep_time = min(remaining * 0.5, 0.005)  # Max 5ms sleep
                    time.sleep(sleep_time)
                    continue

            # ====================================================
            # Try to read a new frame from the circular buffer
            # ====================================================
            entry = self._frame_buffer.read()

            if entry is not None:
                # New frame available — use it
                self._last_read_entry = entry
                self._repeat_count = 0
            elif self._last_read_entry is not None:
                # No new frame — repeat the last one for smooth slow motion
                self._repeat_count += 1
                entry = self._last_read_entry

                # If we've repeated too many times, try to read the latest
                # frame to catch up (prevents infinite stalling)
                if self._repeat_count > self._max_repeats:
                    catch_up_entry = self._frame_buffer.read_latest()
                    if catch_up_entry is not None:
                        entry = catch_up_entry
                        self._last_read_entry = catch_up_entry
                        self._repeat_count = 0
                    else:
                        # FIX: Fallback to latest cached JPEG when buffer is empty
                        cached = self.get_jpeg_bytes()
                        if cached is not None:
                            # Create a temporary entry from cached JPEG
                            entry = FrameEntry(cached, -1, time.perf_counter())
                            self._repeat_count = 0
                        else:
                            time.sleep(0.005)
                            continue
            else:
                # ====================================================
                # FIX: No frames at all — fallback to latest cached JPEG
                # instead of sleeping. Never show black screen.
                # ====================================================
                cached = self.get_jpeg_bytes()
                if cached is not None:
                    entry = FrameEntry(cached, -1, time.perf_counter())
                    self._last_read_entry = entry
                    self._repeat_count = 0
                else:
                    # Truly no frames yet — brief sleep
                    time.sleep(0.005)
                    continue

            # ====================================================
            # Render this frame
            # ====================================================
            self._last_render_time = now

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
                + entry.jpeg_bytes
                + b'\r\n'
            )

    # -------------------------------------------------------
    # Stats
    # -------------------------------------------------------

    def get_stats(self):
        """Return streaming performance metrics."""
        return {
            "encode_time_ms": round(self._encode_time_ms, 2),
            "frame_count": self._frame_count,
            "jpeg_buffer_size": len(self._jpeg_bytes) if self._jpeg_bytes else 0,
            "playback_speed": self._playback_speed,
            "display_fps_actual": round(self._display_fps_actual, 1),
            "inference_fps": round(self._inference_fps_actual, 1),
            "latency_ms": round(self._current_latency_ms, 1),
            "buffer_size": self._frame_buffer.size(),
            "buffer_capacity": self._frame_buffer.capacity(),
        }

    def get_display_metrics(self):
        """
        Get display-specific metrics for the FPS overlay.
        Returns dict with camera_fps, inference_fps, display_fps,
        playback_speed, buffer_size, latency_ms.
        """
        return {
            "camera_fps": round(self._inference_fps_actual, 1),
            "inference_fps": round(self._inference_fps_actual, 1),
            "display_fps": round(self._display_fps_actual, 1),
            "playback_speed": self._playback_speed,
            "buffer_size": self._frame_buffer.size(),
            "buffer_capacity": self._frame_buffer.capacity(),
            "latency_ms": round(self._current_latency_ms, 1),
        }

    # -------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------

    def clear(self):
        """Clear all buffers and reset state."""
        with self._lock:
            self._jpeg_buffer = None
            self._jpeg_bytes = None
            self._raw_frame = None
            self._frame_count = 0
            self._has_received_frame = False
        self._frame_buffer.clear()
        self._last_render_time = 0.0
        self._display_frame_count = 0
        self._display_fps_actual = 0.0
        self._inference_frame_count = 0
        self._inference_fps_actual = 0.0
        self._last_read_entry = None
        self._repeat_count = 0
        self._interval_history = []
        self._first_frame_yielded = False