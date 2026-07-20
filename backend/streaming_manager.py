"""
============================================================
StreamingManager
Optimized MJPEG streaming with JPEG caching.
Encodes JPEG only ONCE per frame.
Reuses encoded bytes.
Separate thread for streaming.
============================================================
"""

import cv2
import time
import threading
from backend.config import JPEG_QUALITY


class StreamingManager:
    """
    Manages MJPEG streaming with:
    - JPEG encoding cache (encode once, serve many)
    - Separate streaming thread
    - Frame buffer for WebSocket updates
    - Performance logging
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg_buffer = None
        self._jpeg_bytes = None
        self._raw_frame = None
        self._frame_count = 0
        self._encode_time_ms = 0
        self._running = False

    def update_frame(self, frame):
        """
        Update the current frame and encode JPEG once.
        This is called from the processing pipeline.
        Only encodes if frame has changed.
        """
        if frame is None:
            return

        self._raw_frame = frame

        start = time.perf_counter()
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        elapsed = (time.perf_counter() - start) * 1000
        self._encode_time_ms = elapsed

        if success:
            with self._lock:
                self._jpeg_buffer = buffer
                self._jpeg_bytes = buffer.tobytes()
            self._frame_count += 1

    def get_jpeg_bytes(self):
        """Get the latest JPEG bytes (cached)."""
        with self._lock:
            return self._jpeg_bytes

    def get_raw_frame(self):
        """Get the latest raw frame."""
        return self._raw_frame

    def generate_mjpeg_frames(self):
        """
        Generator for MJPEG streaming.
        Yields frames as multipart/x-mixed-replace.
        Blocks until next frame is available.
        """
        last_frame_count = -1
        while True:
            current_count = self._frame_count
            if current_count == last_frame_count:
                # No new frame yet, sleep briefly
                time.sleep(0.016)  # ~60fps polling
                continue

            bytes_data = self.get_jpeg_bytes()
            if bytes_data is None:
                time.sleep(0.016)
                continue

            last_frame_count = current_count

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + bytes_data
                + b'\r\n'
            )

    def get_stats(self):
        return {
            "encode_time_ms": round(self._encode_time_ms, 2),
            "frame_count": self._frame_count,
            "jpeg_buffer_size": len(self._jpeg_bytes) if self._jpeg_bytes else 0,
        }

    def clear(self):
        """Clear the frame buffer."""
        with self._lock:
            self._jpeg_buffer = None
            self._jpeg_bytes = None
            self._raw_frame = None
            self._frame_count = 0