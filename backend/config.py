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

# Lower IoU threshold for emergency override (catch partial overlaps)
EMERGENCY_OVERLAP_IOU = 0.3

# -----------------------------
# Temporal Confirmation
# -----------------------------
# Number of consecutive frames emergency must be detected before locking
TEMPORAL_CONFIRM_FRAMES = 2
# Number of frames to keep emergency class locked after confirmation
TEMPORAL_LOCK_FRAMES = 15

# Vehicle temporal confirmation: frames needed to confirm a new vehicle
VEHICLE_CONFIRM_FRAMES = 2
# Frames a vehicle can be missing before removal
VEHICLE_MISSING_TIMEOUT = 10

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
# Playback / Display Speed Control
# -----------------------------
# Base display frame rate for the MJPEG stream.
# The actual display FPS = DISPLAY_FPS_BASE * PLAYBACK_SPEED.
# At 1.0x speed → 20 FPS display (smooth CCTV playback)
# At 0.5x speed → 10 FPS display (slow motion)
# Inference runs at full speed regardless.
DISPLAY_FPS_BASE = 20

# Playback speed multiplier for the displayed video.
# AI inference runs at full speed regardless.
# 1.0  = real-time (no slow motion)
# 0.85 = ~15% slower (professional CCTV demo look)
# 0.75 = 25% slower
# 0.5  = half speed (each frame held 2x longer)
PLAYBACK_SPEED = 0.8

# Circular buffer capacity for processed frames.
# Larger values allow longer delays but use more memory.
# At 30 FPS inference: 60 frames = ~2 seconds of delay at 0.5x playback
FRAME_BUFFER_SIZE = 60

# Available playback speed options for the UI controls
PLAYBACK_SPEED_OPTIONS = [0.5, 0.75, 0.85, 1.0]

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

# -----------------------------
# Double Counting Prevention
# -----------------------------
# Number of frames a counted ID stays in the set before being eligible for recount
# Set to a high value to prevent double counting during the entire video
COUNTED_ID_TIMEOUT_FRAMES = 999999  # Effectively never re-count

# -----------------------------
# Streaming Performance
# -----------------------------
# Maximum time (seconds) to wait for a new frame before repeating the last one
STREAMING_FRAME_TIMEOUT = 0.5
# Minimum display interval (seconds) to prevent excessive frame rates
MIN_DISPLAY_INTERVAL = 0.016  # ~60 FPS max display