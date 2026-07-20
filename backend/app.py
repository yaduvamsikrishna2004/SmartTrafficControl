"""
============================================================
TrafficIQ - AI Smart Traffic Intelligence Platform
FastAPI Backend (OPTIMIZED)
============================================================

OPTIMIZATIONS:
1. Models loaded ONCE at startup via ModelManager
2. Camera opened ONCE, reused, auto-reconnect
3. First frame streams IMMEDIATELY (no AI wait)
4. Multi-threaded pipeline (camera / inference / streaming)
5. Frame buffer drops stale frames automatically
6. JPEG encoded ONCE, cached, reused
7. Models stay on GPU permanently
8. Stop/Restart never reloads models
9. Performance logging every 30 frames
10. WebSocket stays alive, no reconnection
11. WebSocket broadcast with proper async handling
12. Error recovery: auto-reconnect camera, graceful degradation
============================================================
"""

import cv2
import sys
import os
import time
import json
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.model_manager import ModelManager
from backend.inference_pipeline import InferencePipeline
from backend.config import *

# ==========================================================
# Initialize models ONCE at module load time
# ==========================================================
print("=" * 60)
print("TrafficIQ - Loading AI Models...")
print("=" * 60)
model_manager = ModelManager()
model_manager.load_models()  # Loads both models, warms them up, moves to GPU
print("[Startup] AI Models ready in memory")

# ==========================================================
# FastAPI
# ==========================================================
app = FastAPI(
    title="TrafficIQ - AI Smart Traffic Intelligence",
    version="3.0",
    description="Optimized real-time AI traffic monitoring with sub-second startup"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Pipeline Instance (created on start, destroyed on stop)
# Models are NOT re-created - they're shared from ModelManager
# ==========================================================
pipeline: InferencePipeline = None
app_state = {
    "running": False,
    "start_time_ms": 0,
    "last_error": None,
}

# ==========================================================
# WebSocket connections
# ==========================================================
active_websockets = set()
_ws_lock = threading.Lock()


@app.on_event("startup")
async def startup_event():
    """FastAPI startup: models already loaded above."""
    print("[Startup] FastAPI server started")
    print(f"[Startup] Models pre-loaded and ready on {model_manager.get_device()}")
    print(f"[Startup] Use /start-video to begin monitoring")


# ==========================================================
# Root
# ==========================================================
@app.get("/")
def home():
    return {
        "project": "TrafficIQ",
        "version": "3.0",
        "status": "Running",
        "models_loaded": ModelManager.is_loaded(),
        "device": model_manager.get_device(),
        "monitoring": app_state["running"],
    }


# ==========================================================
# Status
# ==========================================================
@app.get("/status")
def status():
    global pipeline
    fps = pipeline.fps if pipeline else 0
    return {
        "backend": "Running",
        "processing": app_state["running"],
        "fps": fps,
        "health": "OK" if app_state["running"] else "IDLE",
        "camera": "Running" if app_state["running"] else "Stopped",
        "models_loaded": ModelManager.is_loaded(),
        "device": model_manager.get_device(),
    }


@app.get("/health")
def health():
    return status()


# ==========================================================
# START MONITORING (Optimized: models already loaded)
# ==========================================================
@app.post("/start-video")
def start_video():
    global pipeline, app_state

    if app_state["running"]:
        return {"message": "Video already running", "fps": pipeline.fps if pipeline else 0}

    try:
        start = time.perf_counter()

        # Create pipeline with pre-loaded models
        pipeline = InferencePipeline(model_manager, LANE_CONFIG)

        # Start pipeline (opens camera, starts threads)
        success = pipeline.start()

        elapsed_ms = (time.perf_counter() - start) * 1000
        app_state["running"] = success
        app_state["start_time_ms"] = elapsed_ms
        app_state["last_error"] = None

        if success:
            print(f"[Startup] Monitoring started in {elapsed_ms:.1f}ms")
            # Notify websockets
            _broadcast({"type": "status", "data": "running"})
            return {
                "message": "Video processing started",
                "startup_ms": round(elapsed_ms, 1),
                "target": "< 1000ms",
            }
        else:
            app_state["last_error"] = "Camera failed to open"
            return JSONResponse(
                status_code=500,
                content={"message": "Failed to open camera", "error": "Camera not available"}
            )

    except Exception as e:
        app_state["last_error"] = str(e)
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to start monitoring", "error": str(e)}
        )


# ==========================================================
# STOP MONITORING (Releases camera, but NOT models)
# ==========================================================
@app.post("/stop-video")
def stop_video():
    global pipeline, app_state

    if not app_state["running"]:
        return {"message": "Video not running"}

    try:
        start = time.perf_counter()
        if pipeline:
            pipeline.stop()
        app_state["running"] = False
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[Stop] Monitoring stopped in {elapsed_ms:.1f}ms")

        # Notify websockets
        _broadcast({"type": "status", "data": "stopped"})

        return {"message": "Video processing stopped", "stop_time_ms": round(elapsed_ms, 1)}

    except Exception as e:
        app_state["last_error"] = str(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to stop", "error": str(e)}
        )


# ==========================================================
# RESTART (Does NOT reload models - reuses from ModelManager)
# ==========================================================
@app.post("/restart-video")
def restart_video():
    """Restart monitoring without reloading AI models."""
    global pipeline, app_state

    try:
        # Stop if running
        if app_state["running"]:
            if pipeline:
                pipeline.stop()
            app_state["running"] = False
            print("[Restart] Stopped previous session")

        # Start fresh (models still in memory from ModelManager)
        start = time.perf_counter()
        pipeline = InferencePipeline(model_manager, LANE_CONFIG)
        success = pipeline.start()
        elapsed_ms = (time.perf_counter() - start) * 1000
        app_state["running"] = success
        app_state["start_time_ms"] = elapsed_ms
        app_state["last_error"] = None

        # Notify websockets
        _broadcast({"type": "status", "data": "running"})

        return {
            "message": "Video restarted successfully",
            "startup_ms": round(elapsed_ms, 1),
            "models_reloaded": False,  # Never reload models!
        }

    except Exception as e:
        app_state["last_error"] = str(e)
        return JSONResponse(
            status_code=500,
            content={"message": "Failed to restart", "error": str(e)}
        )


# ==========================================================
# Camera Status
# ==========================================================
@app.get("/camera")
def camera():
    return {
        "status": "running" if app_state["running"] else "stopped",
        "feed_url": "/video-feed",
        "backend": "TrafficIQ",
        "startup_ms": app_state.get("start_time_ms", 0),
    }


# ==========================================================
# LANES
# ==========================================================
@app.get("/lanes")
def lanes():
    global pipeline
    if pipeline and app_state["running"]:
        dashboard = pipeline.get_dashboard_data()
        return JSONResponse(dashboard.get("lane_data", {}))
    return JSONResponse({})


# ==========================================================
# Traffic Counter
# ==========================================================
@app.get("/traffic")
def traffic():
    global pipeline
    if pipeline and app_state["running"]:
        return JSONResponse(pipeline.get_counter())
    return JSONResponse({})


@app.get("/density")
def density():
    global pipeline
    if pipeline and app_state["running"]:
        return JSONResponse(pipeline.get_density())
    return JSONResponse({})


@app.get("/signals")
def signals():
    global pipeline
    if pipeline and app_state["running"]:
        return JSONResponse(pipeline.get_signals())
    return JSONResponse({})


# ==========================================================
# Analytics
# ==========================================================
@app.get("/analytics")
def analytics():
    return statistics()


@app.get("/statistics")
def statistics():
    global pipeline
    if pipeline and app_state["running"]:
        return JSONResponse(pipeline.get_statistics())
    return JSONResponse({
        "total_vehicles": 0, "cars": 0, "bus": 0,
        "van": 0, "others": 0, "emergency_vehicles": 0
    })


# ==========================================================
# Emergency
# ==========================================================
@app.get("/emergency")
def emergency():
    global pipeline
    if pipeline and app_state["running"]:
        return JSONResponse(pipeline.get_emergency_status())
    return JSONResponse({
        "emergency": {"active": False},
        "summary": {"current_count": 0, "total_count": 0, "per_lane_count": {}, "per_vehicle_count": {}}
    })


# ==========================================================
# OPTIMIZED MJPEG STREAMING
# Uses cached JPEG bytes from StreamingManager.
# No repeated encoding.
# ==========================================================
@app.get("/video-feed")
def video_feed():
    global pipeline

    if not pipeline or not app_state["running"]:
        # Return a static frame if not running
        return StreamingResponse(
            _static_fallback_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    return StreamingResponse(
        pipeline.streaming.generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def _static_fallback_frames():
    """Generate a static 'Camera Stopped' frame when not monitoring."""
    import numpy as np
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Stopped", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1)


# ==========================================================
# Dashboard
# ==========================================================
@app.get("/dashboard")
def dashboard():
    global pipeline
    live_fps = pipeline.fps if pipeline else 0

    result = {
        "system": {
            "project": "TrafficIQ",
            "version": "3.0",
            "backend": "Running",
            "processing": app_state["running"],
            "fps": live_fps,
            "device": model_manager.get_device(),
            "models_loaded": ModelManager.is_loaded(),
        },
        "dashboard": pipeline.get_dashboard_data() if pipeline and app_state["running"] else {},
    }

    return result


# ==========================================================
# Playback Speed Control
# ==========================================================
@app.get("/playback-speed")
def get_playback_speed():
    """Get the current playback speed setting."""
    global pipeline
    if pipeline and app_state["running"]:
        return {
            "playback_speed": pipeline.streaming.get_playback_speed(),
            "options": pipeline.streaming.get_playback_speed_options(),
        }
    return {"playback_speed": 1.0, "options": [0.25, 0.5, 0.75, 1.0]}


@app.post("/playback-speed")
def set_playback_speed(speed: float = 0.5):
    """Set the playback speed. Options: 0.25, 0.5, 0.75, 1.0"""
    global pipeline
    if pipeline and app_state["running"]:
        pipeline.streaming.set_playback_speed(speed)
        return {
            "message": f"Playback speed set to {pipeline.streaming.get_playback_speed()}x",
            "playback_speed": pipeline.streaming.get_playback_speed(),
        }
    return {"message": "Pipeline not running", "playback_speed": 1.0}


# ==========================================================
# Performance Stats
# ==========================================================
@app.get("/perf")
def perf_stats():
    """Return detailed performance metrics."""
    global pipeline
    if pipeline and app_state["running"]:
        s = pipeline.stats
        return {
            "camera_read_ms": round(s.camera_read_ms, 2),
            "vehicle_inference_ms": round(s.vehicle_inference_ms, 2),
            "emergency_inference_ms": round(s.emergency_inference_ms, 2),
            "merge_ms": round(s.merge_ms, 2),
            "lane_assign_ms": round(s.lane_assign_ms, 2),
            "signal_ms": round(s.signal_ms, 2),
            "draw_ms": round(s.draw_ms, 2),
            "jpeg_encode_ms": round(s.jpeg_encode_ms, 2),
            "total_pipeline_ms": round(s.total_pipeline_ms, 2),
            "e2e_latency_ms": round(s.e2e_latency_ms, 2),
            "fps": round(s.fps, 1),
            "frame_count": s.frame_count,
        }
    return {"status": "idle", "fps": 0}


# ==========================================================
# WebSocket for real-time data
# ==========================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with _ws_lock:
        active_websockets.add(websocket)
    print(f"[WebSocket] Client connected. Total: {len(active_websockets)}")

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif data == "status":
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": "running" if app_state["running"] else "stopped"
                }))
            elif data == "dashboard":
                # Send dashboard data
                global pipeline
                if pipeline and app_state["running"]:
                    dashboard_data = pipeline.get_dashboard_data()
                    await websocket.send_text(json.dumps({
                        "type": "dashboard",
                        "data": dashboard_data
                    }))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        with _ws_lock:
            active_websockets.discard(websocket)
        print(f"[WebSocket] Client disconnected. Total: {len(active_websockets)}")


def _broadcast(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    with _ws_lock:
        stale = set()
        for ws in active_websockets:
            try:
                # Use a background task approach for async send
                loop = asyncio.new_event_loop()
                loop.run_until_complete(ws.send_text(json.dumps(message)))
                loop.close()
            except Exception:
                stale.add(ws)
        active_websockets.difference_update(stale)


# ==========================================================
# WebSocket streaming of frames (alternative to MJPEG)
# ==========================================================
@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """Stream base64-encoded frames over WebSocket."""
    await websocket.accept()
    print("[WebSocket Video] Client connected")

    try:
        last_frame_id = -1
        while True:
            if not app_state["running"] or not pipeline:
                await asyncio.sleep(0.1)
                continue

            frame = pipeline.get_latest_frame()
            if frame is None:
                await asyncio.sleep(0.016)
                continue

            # Encode JPEG (for WS clients we need to encode per client)
            # This is acceptable for WebSocket as it's per-client
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            import base64
            b64 = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_text(json.dumps({
                "type": "frame",
                "data": b64,
                "timestamp": time.time(),
            }))

            # Control frame rate for WebSocket
            await asyncio.sleep(0.033)  # ~30fps

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Video] Error: {e}")
    finally:
        print("[WebSocket Video] Client disconnected")


# ==========================================================
# Entry Point
# ==========================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("TrafficIQ v3.0 - Optimized")
    print("=" * 60)
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # No reload in production
        log_level="info",
    )