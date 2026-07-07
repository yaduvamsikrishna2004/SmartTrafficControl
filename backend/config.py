"""
Application Configuration
"""

# -----------------------------
# Model & Data
# -----------------------------

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/cam4.mp4"
LANE_CONFIG = "config/lanes.json"

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