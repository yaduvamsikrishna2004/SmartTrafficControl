"""
============================================================
CameraManager
Opens VideoCapture ONCE and reuses it.
Auto-reconnects on failure.
Provides latest frame instantly without blocking.
============================================================
"""

import cv2
import time
import threading
from collections import deque
from backend.config import VIDEO_PATH


class CameraManager:
    """
    Manages a single VideoCapture instance.
    - Opens camera ONCE at start_monitoring
    - Never reopens unless explicitly needed
    - Auto-reconnects on disconnect
    - Maintains latest frame buffer
    - Thread-safe frame reading
    """

    def __init__(self, video_path=VIDEO_PATH):
        self.video_path = video_path
        self.cap = None
        self.is_open = False
        self._lock = threading.Lock()
        self._frame_buffer = None
        self._frame_count = 0
        self._open_time_ms = 0
        self._read_time_ms = 0
        self._reconnect_attempts = 0

    def open(self):
        """Open the video capture. Returns True if successful."""
        with self._lock:
            if self.cap is not None:
                self.release()

            start = time.perf_counter()
            self.cap = cv2.VideoCapture(self.video_path)

            # Set buffer size to minimum for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            elapsed = (time.perf_counter() - start) * 1000
            self._open_time_ms = elapsed

            if self.cap.isOpened():
                self.is_open = True
                self._reconnect_attempts = 0
                print(f"[Camera] Opened in {elapsed:.1f}ms | Path: {self.video_path}")
                return True
            else:
                self.is_open = False
                print(f"[Camera] FAILED to open in {elapsed:.1f}ms")
                return False

    def read(self):
        """
        Read the latest frame.
        Returns (ret, frame) where ret is True if successful.
        Non-blocking - returns immediately with latest frame.
        """
        with self._lock:
            if self.cap is None or not self.is_open:
                return False, None

            start = time.perf_counter()
            ret, frame = self.cap.read()
            elapsed = (time.perf_counter() - start) * 1000
            self._read_time_ms = elapsed

            if ret:
                self._frame_count += 1
                self._frame_buffer = frame
                return True, frame
            else:
                # Auto-reconnect on failure
                print(f"[Camera] Read failed after {self._frame_count} frames. Reconnecting...")
                self._reconnect_attempts += 1
                self._try_reconnect()
                return False, None

    def get_latest_frame(self):
        """Get the latest buffered frame without reading."""
        return self._frame_buffer

    def _try_reconnect(self):
        """Attempt to reconnect the camera."""
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cap.isOpened():
                self.is_open = True
                print(f"[Camera] Reconnected successfully (attempt {self._reconnect_attempts})")
            else:
                self.is_open = False
                print(f"[Camera] Reconnect failed (attempt {self._reconnect_attempts})")
        except Exception as e:
            self.is_open = False
            print(f"[Camera] Reconnect error: {e}")

    def release(self):
        """Release the camera."""
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_open = False
            self._frame_buffer = None
            self._frame_count = 0
            print("[Camera] Released")

    @property
    def fps(self):
        """Get the camera's native FPS."""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0

    @property
    def frame_width(self):
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    @property
    def frame_height(self):
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0

    def get_stats(self):
        return {
            "open_time_ms": round(self._open_time_ms, 1),
            "read_time_ms": round(self._read_time_ms, 1),
            "frame_count": self._frame_count,
            "reconnect_attempts": self._reconnect_attempts,
            "is_open": self.is_open,
        }