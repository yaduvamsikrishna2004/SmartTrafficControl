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

        "project": "AI Smart Traffic Signal Control",

        "status": "Running"

    }

# ==========================================================

@app.get("/status")
def status():

    return {

        "backend": "Running",

        "processing": app_state["running"],

        "fps": app_state["fps"]

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

@app.post("/stop-video")
def stop_video():

    app_state["running"] = False

    return {

        "message": "Processing stopped"

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