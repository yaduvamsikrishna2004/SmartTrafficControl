"""
Application Configuration
"""

import os

# Support both relative paths (for legacy scripts running from SmartTrafficProject/)
# and absolute paths (for scripts running from parent directory)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _resolve(path):
    """Resolve model/data paths to absolute paths for reliability."""
    abspath = os.path.join(_BASE_DIR, path)
    return abspath

# -----------------------------
# Model & Data
# -----------------------------

MODEL_PATH = _resolve("models/best.pt")
EMERGENCY_MODEL_PATH = _resolve("models/emergency_best.pt")
VIDEO_PATH = _resolve("videos/cam4.mp4")
LANE_CONFIG = _resolve("config/lanes.json")

# -----------------------------
# Emergency Detection
# -----------------------------
EMERGENCY_CONF = 0.25
EMERGENCY_IOU = 0.45
EMERGENCY_GREEN = 60

# -----------------------------
# Signal Timing
# -----------------------------

MIN_GREEN = 15
MAX_GREEN = 60
GREEN_FACTOR = 2

# -----------------------------
# Processing
# -----------------------------

FRAME_SKIP = 2
API_REFRESH_MS = 1000
JPEG_QUALITY = 80