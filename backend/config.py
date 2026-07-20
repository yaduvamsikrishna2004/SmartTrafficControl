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
# Detection Confidence Thresholds
# -----------------------------
# Vehicle model: lower threshold to catch all vehicles
VEHICLE_CONF = 0.35
VEHICLE_IOU = 0.45

# Emergency model: even lower because emergency vehicles are rare
EMERGENCY_CONF = 0.20
EMERGENCY_IOU = 0.45

# If emergency model confidence > this value, ALWAYS override vehicle class
EMERGENCY_FORCE_OVERRIDE_CONF = 0.40

# -----------------------------
# Merge / IoU Thresholds
# -----------------------------
# IoU threshold for considering two detections as the same object
MERGE_IOU_THRESHOLD = 0.5

# -----------------------------
# Temporal Confirmation
# -----------------------------
# Number of consecutive frames emergency must be detected before locking
TEMPORAL_CONFIRM_FRAMES = 2
# Number of frames to keep emergency class locked after confirmation
TEMPORAL_LOCK_FRAMES = 15

# -----------------------------
# Emergency Signal Override
# -----------------------------
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

# -----------------------------
# Debug Mode
# -----------------------------
# Set to True to draw debug boxes:
#   Blue   = vehicle model detections
#   Red    = emergency model detections
#   Green  = final merged output
DEBUG_MODE = True

# -----------------------------
# Emergency Priority Order
# -----------------------------
# Higher value = higher priority
EMERGENCY_PRIORITY_ORDER = {
    "ambulance": 100,
    "fire_truck": 90,
    "police": 80,
}
