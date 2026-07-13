"""
============================================================
AI Smart Traffic Signal Control System
FastAPI Backend (Real-Time)

Author : Vamsi Krishna
============================================================
"""

import cv2
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.traffic_engine import TrafficEngine
from backend.config import *

# ==========================================================
# FastAPI
# ==========================================================

app = FastAPI(
    title="Smart Traffic Signal Control API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Traffic Engine
# ==========================================================

engine = TrafficEngine(
    MODEL_PATH,
    LANE_CONFIG
)

# ==========================================================
# Global State
# ==========================================================

app_state = {

    "running": False,

    "thread": None,

    "fps": 0

}

# ==========================================================
# Background Video Processing
# ==========================================================

def process_video():

    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_count = 0

    start = time.time()

    while app_state["running"]:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Frame Skip
        if frame_count % FRAME_SKIP != 0:
            continue

        engine.process_frame(frame)

    cap.release()

    elapsed = time.time() - start

    if elapsed > 0:

        app_state["fps"] = round(frame_count / elapsed, 2)

    app_state["running"] = False

# ==========================================================
# API
# ==========================================================

@app.get("/")
def home():

    return {

        "project": "TrafficIQ",

        "version": "1.0",

        "status": "Running"

    }

# ==========================================================

@app.get("/status")
def status():

    return {

        "backend": "Running",

        "processing": app_state["running"],

        "fps": engine.fps or app_state["fps"],

        "health": "OK" if app_state["running"] else "IDLE",

        "camera": "Running" if app_state["running"] else "Stopped"

    }


# ==========================================================

@app.get("/health")
def health():

    return {

        "status": "OK",

        "backend": "Running",

        "processing": app_state["running"],

        "fps": engine.fps or app_state["fps"],

        "camera": "Running" if app_state["running"] else "Stopped"

    }


# ==========================================================

@app.get("/lanes")
def lanes():

    dashboard = engine.get_dashboard_data()

    return JSONResponse(

        dashboard.get("lane_data", {})

    )


# ==========================================================

@app.get("/camera")
def camera():

    return {

        "status": "running" if app_state["running"] else "stopped",

        "feed_url": "/video-feed",

        "backend": "TrafficIQ"

    }

# ==========================================================

@app.post("/start-video")
def start_video():

    if app_state["running"]:

        return {

            "message": "Video already running"

        }

    engine.reset()

    app_state["running"] = True

    thread = threading.Thread(
        target=process_video,
        daemon=True
    )

    app_state["thread"] = thread

    thread.start()

    return {

        "message": "Video processing started"

    }

# ==========================================================

@app.get("/traffic")
def traffic():

    return JSONResponse(

        engine.get_counter()

    )

# ==========================================================

@app.get("/density")
def density():

    return JSONResponse(

        engine.get_density()

    )

# ==========================================================

@app.get("/signals")
def signals():

    return JSONResponse(

        engine.get_signals()

    )

# ==========================================================

@app.get("/analytics")
def analytics():

    return JSONResponse(

        engine.get_statistics()

    )

# ==========================================================
# Statistics Alias
# ==========================================================

@app.get("/statistics")
def statistics():

    return JSONResponse(

        engine.get_statistics()

    )

# ==========================================================
# Emergency Status
# ==========================================================

@app.get("/emergency")
def emergency():

    dashboard = engine.get_dashboard_data()

    return JSONResponse(

        {
            "emergency": dashboard.get("emergency", {"active": False}),
            "summary": dashboard.get("emergency_summary", {
                "current_count": 0,
                "total_count": 0,
                "per_lane_count": {},
                "per_vehicle_count": {}
            })
        }

    )

# ==========================================================
# Video Streaming
# ==========================================================

def generate_frames():

    while True:

        frame = engine.get_latest_frame()

        if frame is None:

            time.sleep(0.05)

            continue

        _, buffer = cv2.imencode(

            ".jpg",

            frame,

            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        )

        yield (

            b'--frame\r\n'

            b'Content-Type: image/jpeg\r\n\r\n'

            + buffer.tobytes()

            + b'\r\n'

        )

# ==========================================================

@app.get("/video-feed")
def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type="multipart/x-mixed-replace; boundary=frame"

    )

# ==========================================================
# Dashboard API
# ==========================================================



@app.get("/dashboard")
def dashboard():

    # Use engine.fps for LIVE FPS during processing, fallback to app_state
    live_fps = engine.fps if engine.fps > 0 else (app_state["fps"] if app_state["fps"] > 0 else 0)

    return {

        "system": {

            "project": "TrafficIQ",

            "version": "1.0",

            "backend": "Running",

            "processing": app_state["running"],

            "fps": live_fps

        },

        "dashboard": engine.get_dashboard_data()

    }
