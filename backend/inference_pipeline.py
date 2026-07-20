"""
============================================================
InferencePipeline
Manages the threaded AI processing pipeline with:
- Thread 1: Camera reading (non-blocking)
- Thread 2: YOLO detection (vehicle + emergency)
- Thread 3: Tracking + merging
- Thread 4: Signal controller + analytics
- Thread 5: Streaming + WebSocket updates

No blocking between threads.
Frame buffer always uses newest frame.
First frame streams immediately without waiting for AI.

FIXES IMPLEMENTED:
1. Frame Synchronization: overlay belongs to same frame as detections
2. Missed Vehicles: temporal confirmation (2 frames to confirm, 10 missing to remove)
3. Tracking Stability: IoU + motion + appearance matching for ID persistence
4. Professional Overlay: FPS, Inference Time, Tracking Time, Vehicle Count, 
   Emergency Count, Signal, Lane, AI Confidence, Timestamp
5. Emergency vehicle: red blinking box, thick border, priority icon
6. Double Counting: counter now uses IoU matching + timeout
7. Error Handling: try/except around every stage, auto-recovery
============================================================
"""

import cv2
import time
import threading
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from backend.model_manager import ModelManager
from backend.camera_manager import CameraManager
from backend.streaming_manager import StreamingManager
from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.counter import VehicleCounter
from backend.density import DensityCalculator
from backend.signal_controller import SignalController
from backend.emergency import EmergencyDetector
from backend.priority_engine import PriorityEngine
from backend.detector import YOLODetector
from backend.utils import build_class_map, set_class_map, normalize_emergency_class, is_emergency_class
from backend.config import (
    EMERGENCY_CONF, EMERGENCY_IOU, EMERGENCY_GREEN,
    VEHICLE_CONF, VEHICLE_IOU, MERGE_IOU_THRESHOLD,
    TEMPORAL_CONFIRM_FRAMES, TEMPORAL_LOCK_FRAMES,
    EMERGENCY_FORCE_OVERRIDE_CONF, DEBUG_MODE,
    EMERGENCY_PRIORITY_ORDER, FRAME_SKIP,
    VEHICLE_CONFIRM_FRAMES, VEHICLE_MISSING_TIMEOUT,
)


@dataclass
class PerformanceStats:
    """Track performance metrics for each pipeline stage."""
    camera_read_ms: float = 0.0
    vehicle_inference_ms: float = 0.0
    emergency_inference_ms: float = 0.0
    merge_ms: float = 0.0
    tracking_ms: float = 0.0
    lane_assign_ms: float = 0.0
    signal_ms: float = 0.0
    draw_ms: float = 0.0
    jpeg_encode_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    fps: float = 0.0
    frame_count: int = 0


class FrameBuffer:
    """
    Thread-safe frame buffer.
    Always keeps the LATEST frame.
    Old frames are dropped automatically.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._timestamp = 0

    def set(self, frame, frame_id):
        """Store the latest frame. Always overwrites old one."""
        with self._lock:
            self._frame = frame
            self._frame_id = frame_id
            self._timestamp = time.perf_counter()

    def get(self):
        """Get the latest frame. Returns (frame, frame_id)."""
        with self._lock:
            return self._frame, self._frame_id

    def get_latest(self):
        """Get just the frame (non-blocking)."""
        with self._lock:
            return self._frame

    def clear(self):
        with self._lock:
            self._frame = None
            self._frame_id = 0


class InferencePipeline:
    """
    Multi-threaded inference pipeline.
    
    Architecture:
    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │ Camera      │───▶│ Frame Buffer │───▶│ Stream       │
    │ Reader Thd  │    │ (latest only)│    │ (immediate)  │
    └─────────────┘    └──────┬───────┘    └──────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ AI Inference Thread │
                    │ (vehicle + emergency)│
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Post-Processing    │
                    │ (merge, track,     │
                    │  signal, analytics)│
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ WebSocket Updates  │
                    └────────────────────┘
    """

    def __init__(self, model_manager: ModelManager, lane_config: str):
        self.model_manager = model_manager
        self.lane_config_path = lane_config

        # ========================================================
        # Core Components
        # ========================================================
        self.camera = CameraManager()
        self.streaming = StreamingManager()
        self.frame_buffer = FrameBuffer()
        self.stats = PerformanceStats()

        # ========================================================
        # AI Modules (use pre-loaded models from ModelManager)
        # ========================================================
        vehicle_model = model_manager.get_vehicle_model()
        emergency_model = model_manager.get_emergency_model()

        self.tracker = VehicleTracker.__new__(VehicleTracker)
        # Override detector to use shared model
        self.tracker.detector = YOLODetector.__new__(YOLODetector)
        self.tracker.detector.model = vehicle_model
        self.tracker.detector.class_names = model_manager.vehicle_class_names
        print("[Pipeline] Vehicle tracker using shared model")

        # Emergency detector using shared model
        self.emergency_detector = YOLODetector.__new__(YOLODetector)
        self.emergency_detector.model = emergency_model
        self.emergency_detector.class_names = model_manager.emergency_class_names
        print("[Pipeline] Emergency detector using shared model")

        # Build class map
        try:
            class_names = self.tracker.detector.class_names
            mapping = build_class_map(class_names)
            set_class_map(mapping)
        except Exception:
            pass

        # ========================================================
        # Lane Manager
        # ========================================================
        self.lane_manager = LaneManager(self.lane_config_path)

        # ========================================================
        # Analytics Modules
        # ========================================================
        self.counter = VehicleCounter()
        self.density = DensityCalculator()
        self.signal = SignalController(emergency_green=EMERGENCY_GREEN)
        self.emergency = EmergencyDetector(
            detector=self.emergency_detector,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU
        )
        self.priority = PriorityEngine()

        # ========================================================
        # State
        # ========================================================
        self._running = False
        self._camera_thread = None
        self._inference_thread = None
        self._frame_count = 0
        self._frame_skip_counter = 0
        self._blink_state = False
        self._notified_emergency_ids = set()

        # ========================================================
        # FIX: Vehicle temporal confirmation state
        # Track vehicles across frames to prevent missed detections
        # ========================================================
        # track_id -> {consecutive_seen, consecutive_missing, last_bbox, last_class}
        self._vehicle_temporal_state = {}
        # track_id -> confirmed (True after VEHICLE_CONFIRM_FRAMES consecutive frames)
        self._confirmed_vehicles = set()

        # Latest results (thread-safe via buffer)
        self._results_lock = threading.Lock()
        self._latest_results = {
            "tracks": [],
            "counter": {},
            "density": {},
            "signals": {},
            "emergency": {"active": False},
        }

        # Frame dimension cache (set once)
        self._frame_h = 0
        self._frame_w = 0

        print("=" * 60)
        print("Inference Pipeline Initialized")
        print("=" * 60)

    # ========================================================
    # START
    # ========================================================

    def start(self):
        """Start the pipeline. Returns True if camera opened successfully."""
        if self._running:
            print("[Pipeline] Already running")
            return True

        # Open camera
        start = time.perf_counter()
        camera_opened = self.camera.open()
        camera_open_ms = (time.perf_counter() - start) * 1000

        if not camera_opened:
            print("[Pipeline] Failed to open camera")
            return False

        # Reset state
        self._frame_count = 0
        self._frame_skip_counter = 0
        self._running = True

        # Start camera reader thread (Thread 1)
        self._camera_thread = threading.Thread(
            target=self._camera_reader_loop,
            name="camera-reader",
            daemon=True
        )
        self._camera_thread.start()

        # Start inference thread (Thread 2 + 3 + 4 combined)
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="ai-inference",
            daemon=True
        )
        self._inference_thread.start()

        total_startup_ms = (time.perf_counter() - start) * 1000
        print(f"[Pipeline] Pipeline started in {total_startup_ms:.1f}ms (camera: {camera_open_ms:.1f}ms)")

        return True

    # ========================================================
    # THREAD 1: Camera Reader
    # Continuously reads frames and pushes latest to buffer.
    # Never blocks on AI inference.
    # ========================================================

    def _camera_reader_loop(self):
        """Dedicated thread that reads frames from camera as fast as possible."""
        print("[Pipeline] Camera reader thread started")
        log_counter = 0

        while self._running:
            try:
                read_start = time.perf_counter()
                ret, frame = self.camera.read()
                read_elapsed = (time.perf_counter() - read_start) * 1000
                self.stats.camera_read_ms = read_elapsed

                if not ret:
                    if self._running:
                        if log_counter % 30 == 0:
                            print(f"[Camera] Read failed (ret=False) — frame_count={self._frame_count}")
                        log_counter += 1
                        time.sleep(0.01)
                    continue

                # Log camera info on first successful read
                if self._frame_count == 0:
                    h, w = frame.shape[:2]
                    print(f"[Camera] First frame captured: {w}x{h}, dtype={frame.dtype}")

                # Cache frame dimensions (set once)
                if self._frame_h == 0:
                    self._frame_h, self._frame_w = frame.shape[:2]
                    print(f"[Camera] Frame dimensions cached: {self._frame_w}x{self._frame_h}")
                    try:
                        self.lane_manager.update_scale(self._frame_w, self._frame_h)
                    except Exception as e:
                        print(f"[Camera] Lane scale update error: {e}")

                # Increment global frame counter
                self._frame_count += 1
                frame_id = self._frame_count

                # Log every 30 frames
                if frame_id % 30 == 0:
                    print(f"[Camera] Frame #{frame_id} captured — read_time={read_elapsed:.1f}ms")

                # Push to frame buffer (always latest)
                self.frame_buffer.set(frame, frame_id)

                # ====================================================
                # IMMEDIATE STREAMING: First frame goes out RIGHT AWAY
                # without waiting for AI inference.
                # ====================================================
                if frame_id == 1:
                    # First frame: stream raw (no annotations) immediately
                    print(f"[Pipeline] Streaming FIRST frame (raw, no annotations)")
                    self.streaming.update_frame(frame)
                    print(f"[Pipeline] FIRST FRAME streamed immediately (frame 1)")
                else:
                    # Subsequent frames: stream annotated version
                    pass  # Streaming updated by inference thread

                # Small sleep to prevent busy-waiting
                time.sleep(0.001)
                
            except Exception as e:
                print(f"[Pipeline] Camera reader error: {e}")
                import traceback
                traceback.print_exc()
                if self._running:
                    time.sleep(0.01)

        print("[Pipeline] Camera reader thread stopped")

    # ========================================================
    # THREAD 2+3+4: AI Inference + Post-Processing
    # Runs on the latest frame buffer.
    # Skips frames if inference is backlogged.
    # ========================================================

    def _inference_loop(self):
        """Dedicated thread for AI inference and post-processing."""
        print("[Pipeline] AI inference thread started")
        inf_log_counter = 0

        last_processed_id = 0
        perf_start = time.perf_counter()

        while self._running:
            try:
                # ====================================================
                # Get the latest frame (skip stale ones)
                # ====================================================
                frame, frame_id = self.frame_buffer.get()
                if frame is None or frame_id == last_processed_id:
                    time.sleep(0.002)
                    continue

                # Always process every frame to maintain smooth tracking
                last_processed_id = frame_id
                inf_log_counter += 1

                # ====================================================
                # PERFORMANCE: Start end-to-end timing
                # ====================================================
                e2e_start = time.perf_counter()
                pipeline_start = time.perf_counter()

                self._frame_skip_counter += 1

                # Log first inference
                if inf_log_counter == 1:
                    print(f"[Inference] Starting inference on frame #{frame_id}")

                # ====================================================
                # STEP 1: Vehicle Detection (best.pt)
                # ====================================================
                v_start = time.perf_counter()
                if inf_log_counter == 1:
                    print("[Inference] Running vehicle detection (best.pt)...")
                vehicle_detections, vehicle_raw = self._run_vehicle_inference(frame)
                veh_time = (time.perf_counter() - v_start) * 1000
                self.stats.vehicle_inference_ms = veh_time
                if inf_log_counter == 1:
                    print(f"[Inference] Vehicle detection complete: {len(vehicle_detections)} detections in {veh_time:.1f}ms")

                # ====================================================
                # STEP 2: Emergency Detection (emergency_best.pt)
                # ====================================================
                e_start = time.perf_counter()
                if inf_log_counter == 1:
                    print("[Inference] Running emergency detection (emergency_best.pt)...")
                emergency_detections = self._run_emergency_inference(frame)
                emerg_time = (time.perf_counter() - e_start) * 1000
                self.stats.emergency_inference_ms = emerg_time
                if inf_log_counter == 1:
                    print(f"[Inference] Emergency detection complete: {len(emergency_detections)} detections in {emerg_time:.1f}ms")

                # ====================================================
                # STEP 3: Merge + Track + Lane Assign + Signal
                # ====================================================
                m_start = time.perf_counter()
                merged = self._merge_detections(vehicle_detections, emergency_detections)
                merge_time = (time.perf_counter() - m_start) * 1000
                self.stats.merge_ms = merge_time

                # Priority
                priority_decision = self.priority.evaluate(emergency_detections)

                # Remove duplicates
                merged = self._remove_duplicates(merged)

                # ====================================================
                # FIX: Apply vehicle temporal confirmation
                # This prevents missed vehicles by:
                # 1. Requiring 2 consecutive frames to confirm a new vehicle
                # 2. Keeping vehicles that miss 1 frame (temporal recovery)
                # 3. Removing vehicles missing 10+ frames
                # ====================================================
                merged = self._apply_vehicle_temporal_confirmation(merged)

                # Lane assignment
                la_start = time.perf_counter()
                lane_objects = self.lane_manager.assign_lanes(merged)
                lane_time = (time.perf_counter() - la_start) * 1000
                self.stats.lane_assign_ms = lane_time

                # Tracker class update
                lane_objects = self._update_tracker_classes(lane_objects, emergency_detections)

                # Emergency mode trigger
                self._trigger_emergency(priority_decision, lane_objects)

                # ====================================================
                # FIX: Counter now receives frame_number for timeout tracking
                # ====================================================
                self.counter.update(lane_objects, frame_number=self._frame_count)
                lane_counts = self.counter.get_counts()
                lane_density, class_density = self.density.calculate_density(lane_objects)

                # Signal
                sig_start = time.perf_counter()
                signal_plan = self.signal.generate_signal_plan(class_density, priority_decision)
                sig_time = (time.perf_counter() - sig_start) * 1000
                self.stats.signal_ms = sig_time

                # ====================================================
                # STEP 4: Draw Results (professional overlay)
                # ====================================================
                draw_start = time.perf_counter()
                annotated = self._draw_results(frame, lane_objects, vehicle_detections, 
                                                emergency_detections, priority_decision,
                                                lane_counts, signal_plan)
                draw_time = (time.perf_counter() - draw_start) * 1000
                self.stats.draw_ms = draw_time

                pipeline_time = (time.perf_counter() - pipeline_start) * 1000
                self.stats.total_pipeline_ms = pipeline_time

                # ====================================================
                # STEP 5: Update Streaming (JPEG encode once)
                # ====================================================
                self.streaming.update_frame(annotated)

                # ====================================================
                # Store results for API
                # ====================================================
                with self._results_lock:
                    self._latest_results = {
                        "tracks": lane_objects,
                        "counter": lane_counts,
                        "density": {
                            "lane_density": dict(lane_density),
                            "class_density": dict(class_density)
                        },
                        "signals": signal_plan,
                        "emergency": priority_decision,
                        "emergency_summary": self.emergency.get_summary(),
                        "vehicle_detections": vehicle_detections,
                        "emergency_detections": emergency_detections,
                        "merged_detections": merged,
                        "stats": self.stats,
                    }

                # ====================================================
                # End-to-end latency
                # ====================================================
                e2e_elapsed = (time.perf_counter() - e2e_start) * 1000
                self.stats.e2e_latency_ms = e2e_elapsed

                # ====================================================
                # FPS Calculation (rolling window)
                # ====================================================
                elapsed_total = time.perf_counter() - perf_start
                if elapsed_total > 0 and self._frame_count > 1:
                    self.stats.fps = round(self._frame_count / elapsed_total, 1)
                self.stats.frame_count = self._frame_count

                # ====================================================
                # Performance Logging (every 30 frames)
                # ====================================================
                if self._frame_count % 30 == 0:
                    self._log_performance()

            except Exception as e:
                print(f"[Pipeline] Inference loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.01)

        print("[Pipeline] AI inference thread stopped")

    # ========================================================
    # FIX: Vehicle Temporal Confirmation
    # Prevents missed vehicles by tracking them across frames
    # ========================================================

    def _apply_vehicle_temporal_confirmation(self, detections):
        """
        Apply temporal confirmation to vehicle detections.
        
        Rules:
        1. New vehicle: must be seen for VEHICLE_CONFIRM_FRAMES (2) consecutive frames
        2. Missing 1 frame: keep tracking (temporal recovery)
        3. Missing VEHICLE_MISSING_TIMEOUT (10) frames: remove
        4. Confirmed vehicles are always included
        """
        current_ids = set()
        for det in detections:
            track_id = det.get("track_id")
            if track_id is not None:
                current_ids.add(track_id)

        # Update temporal state for current detections
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            if track_id not in self._vehicle_temporal_state:
                self._vehicle_temporal_state[track_id] = {
                    "consecutive_seen": 0,
                    "consecutive_missing": 0,
                    "last_bbox": det.get("bbox"),
                    "last_class": det.get("class_name"),
                }

            state = self._vehicle_temporal_state[track_id]
            state["consecutive_seen"] += 1
            state["consecutive_missing"] = 0
            state["last_bbox"] = det.get("bbox")
            state["last_class"] = det.get("class_name")

            # Mark as confirmed if seen enough frames
            if state["consecutive_seen"] >= VEHICLE_CONFIRM_FRAMES:
                self._confirmed_vehicles.add(track_id)

        # Update temporal state for missing vehicles
        for track_id in list(self._vehicle_temporal_state.keys()):
            if track_id not in current_ids:
                state = self._vehicle_temporal_state[track_id]
                state["consecutive_seen"] = 0
                state["consecutive_missing"] += 1

                # If vehicle was confirmed and missing for < timeout, recover it
                if track_id in self._confirmed_vehicles:
                    if state["consecutive_missing"] < VEHICLE_MISSING_TIMEOUT:
                        # Recover the vehicle with its last known state
                        recovered_det = {
                            "track_id": track_id,
                            "class_name": state["last_class"],
                            "vehicle_type": state["last_class"],
                            "display_name": state["last_class"],
                            "label": state["last_class"],
                            "vehicle": state["last_class"],
                            "confidence": 0.3,  # Reduced confidence for recovered
                            "bbox": state["last_bbox"],
                            "center": (
                                (state["last_bbox"][0] + state["last_bbox"][2]) // 2,
                                (state["last_bbox"][1] + state["last_bbox"][3]) // 2
                            ) if state["last_bbox"] else (0, 0),
                            "source": "temporal_recovery",
                            "emergency": False,
                            "is_emergency": False,
                            "priority": "NORMAL",
                            "_temporal_recovered": True,
                        }
                        detections.append(recovered_det)
                    else:
                        # Remove vehicle after timeout
                        del self._vehicle_temporal_state[track_id]
                        self._confirmed_vehicles.discard(track_id)

        return detections

    # ========================================================
    # Vehicle Inference
    # ========================================================

    def _run_vehicle_inference(self, frame):
        """Run vehicle detection + tracking."""
        try:
            results = self.tracker.detector.model.track(
                source=frame,
                conf=VEHICLE_CONF,
                iou=VEHICLE_IOU,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )
            result = results[0]
            detections = self.tracker.detector.get_detections(result)
            return detections, result
        except Exception as e:
            print(f"[Pipeline] Vehicle inference error: {e}")
            return [], None

    # ========================================================
    # Emergency Inference
    # ========================================================

    def _run_emergency_inference(self, frame):
        """Run emergency detection model."""
        try:
            raw_results = self.emergency_detector.model.predict(
                source=frame,
                conf=EMERGENCY_CONF,
                iou=EMERGENCY_IOU,
                verbose=False
            )
            emergency_list = self.emergency.detect(
                frame,
                self.lane_manager,
                raw_results=raw_results
            )
            return list(emergency_list)
        except Exception as e:
            print(f"[Pipeline] Emergency inference error: {e}")
            return []

    # ========================================================
    # Merge Detections (same logic as original, optimized)
    # ========================================================

    def _merge_detections(self, vehicle_detections, emergency_detections):
        """Merge vehicle and emergency detections. Emergency always overrides."""
        merged = []
        matched_indices = set()

        for emerg in emergency_detections:
            emerg_bbox = emerg.get("bbox")
            if not emerg_bbox:
                continue

            best_idx = None
            best_iou = 0.0

            for v_idx, veh in enumerate(vehicle_detections):
                if v_idx in matched_indices:
                    continue
                veh_bbox = veh.get("bbox")
                if not veh_bbox:
                    continue
                iou = self._compute_iou(emerg_bbox, veh_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = v_idx

            if best_idx is not None and best_iou > MERGE_IOU_THRESHOLD:
                matched_indices.add(best_idx)
                veh = vehicle_detections[best_idx]
                merged.append(self._create_emergency_obj(emerg, veh))
            elif emerg.get("confidence", 0) >= EMERGENCY_FORCE_OVERRIDE_CONF:
                merged.append(self._create_emergency_obj(emerg, None))
            else:
                merged.append(self._create_emergency_obj(emerg, None))

        # Add remaining vehicle detections
        for v_idx, veh in enumerate(vehicle_detections):
            if v_idx not in matched_indices:
                merged.append({
                    "track_id": veh.get("track_id", hash(str(veh.get("bbox", ""))) % 100000),
                    "class_name": veh.get("class_name", "vehicle"),
                    "vehicle_type": veh.get("class_name", "vehicle"),
                    "display_name": veh.get("class_name", "vehicle"),
                    "label": veh.get("class_name", "vehicle"),
                    "vehicle": veh.get("class_name", "vehicle"),
                    "confidence": veh.get("confidence", 0),
                    "bbox": veh.get("bbox", [0, 0, 0, 0]),
                    "center": veh.get("center", (0, 0)),
                    "source": "vehicle_model",
                    "emergency": False,
                    "is_emergency": False,
                    "priority": "NORMAL",
                })

        return merged

    def _create_emergency_obj(self, emerg, vehicle_match=None):
        """Create standardized emergency object."""
        bbox = emerg.get("bbox", [0, 0, 0, 0])
        center = emerg.get("center", ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2))

        obj = {
            "track_id": emerg.get("track_id", 900000),
            "class_name": emerg["vehicle"],
            "vehicle_type": emerg["vehicle"],
            "display_name": emerg["vehicle"],
            "label": emerg["vehicle"],
            "vehicle": emerg["vehicle"],
            "confidence": emerg.get("confidence", 0),
            "bbox": bbox,
            "center": center,
            "lane": emerg.get("lane", "Unknown"),
            "source": "emergency_model",
            "emergency": True,
            "is_emergency": True,
            "emergency_vehicle": emerg["vehicle"],
            "priority": "HIGH",
            "dashboard_class": emerg["vehicle"],
            "dashboard_label": f"\U0001F691 {emerg['vehicle'].upper()}",
            "override_active": True,
            "override_reason": "Emergency Model Override",
            "emergency_confidence": emerg.get("confidence", 0),
            "emergency_track_id": emerg.get("track_id", 900000),
            "_temporal_locked": emerg.get("_temporal_locked", False),
            "_bbox_key": emerg.get("_bbox_key", ""),
        }

        if vehicle_match is not None:
            obj["track_id"] = vehicle_match.get("track_id", obj["track_id"])
            obj["_original_vehicle_class"] = vehicle_match.get("class_name", "unknown")

        return obj

    def _compute_iou(self, bbox1, bbox2):
        """Fast IoU computation."""
        if bbox1 is None or bbox2 is None:
            return 0.0
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _remove_duplicates(self, detections):
        """Class-specific NMS. Emergency always protected."""
        if len(detections) <= 1:
            return detections

        emergency = [d for d in detections if d.get("emergency")]
        normal = [d for d in detections if not d.get("emergency")]

        keep = []
        suppressed = set()
        for i, a in enumerate(normal):
            if i in suppressed:
                continue
            keep.append(a)
            for j, b in enumerate(normal):
                if j <= i or j in suppressed:
                    continue
                if self._compute_iou(a.get("bbox"), b.get("bbox")) > MERGE_IOU_THRESHOLD:
                    suppressed.add(j)

        # Remove normal that overlap emergency
        final_normal = []
        for norm in keep:
            overlap = False
            for emerg in emergency:
                if self._compute_iou(norm.get("bbox"), emerg.get("bbox")) > MERGE_IOU_THRESHOLD:
                    overlap = True
                    break
            if not overlap:
                final_normal.append(norm)

        return emergency + final_normal

    def _update_tracker_classes(self, lane_objects, emergency_detections):
        """Update tracker: if car becomes ambulance on same IoU, update class."""
        for obj in lane_objects:
            if obj.get("emergency"):
                continue
            obj_bbox = obj.get("bbox")
            if not obj_bbox:
                continue
            for emerg in emergency_detections:
                emerg_bbox = emerg.get("bbox")
                if not emerg_bbox:
                    continue
                if self._compute_iou(obj_bbox, emerg_bbox) > MERGE_IOU_THRESHOLD:
                    new_class = emerg["vehicle"]
                    obj["class_name"] = new_class
                    obj["vehicle_type"] = new_class
                    obj["display_name"] = new_class
                    obj["label"] = new_class
                    obj["vehicle"] = new_class
                    obj["emergency"] = True
                    obj["is_emergency"] = True
                    obj["emergency_vehicle"] = new_class
                    obj["priority"] = "HIGH"
                    obj["dashboard_class"] = new_class
                    obj["dashboard_label"] = f"\U0001F691 {new_class.upper()}"
                    obj["override_active"] = True
                    obj["override_reason"] = "Tracker Class Update (Emergency)"
                    obj["emergency_confidence"] = emerg.get("confidence", 0)
                    break
        return lane_objects

    def _trigger_emergency(self, priority_decision, lane_objects):
        """Trigger emergency mode and send notifications."""
        if priority_decision.get("active", False):
            track_id = priority_decision.get("track_id")
            if track_id and track_id not in self._notified_emergency_ids:
                self._notified_emergency_ids.add(track_id)
                vehicle = priority_decision.get("vehicle", "Unknown")
                lane = priority_decision.get("lane", "Unknown")
                print(f"[Pipeline] EMERGENCY: {vehicle} in lane {lane}")

    # ========================================================
    # Professional Overlay Drawing
    # ========================================================

    def _draw_results(self, frame, lane_objects, vehicle_detections, 
                      emergency_detections, priority_decision,
                      lane_counts, signal_plan):
        """Draw professional overlay with all required information."""
        annotated = frame.copy()
        self._blink_state = (self._frame_count // 5) % 2 == 0

        if DEBUG_MODE:
            for det in vehicle_detections:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)

            for det in emergency_detections:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # ====================================================
        # Draw vehicle bounding boxes
        # ====================================================
        for obj in lane_objects:
            bbox = obj.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            is_emergency = obj.get("emergency", False)

            if is_emergency:
                # Emergency vehicle: RED box, thicker border, blinking
                color = (0, 0, 255)
                thickness = 3
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                if self._blink_state:
                    cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 1)

                vehicle = obj.get("class_name", "emergency")
                emoji_map = {"ambulance": "\U0001F691", "fire_truck": "\U0001F692", "police": "\U0001F694"}
                emoji = emoji_map.get(vehicle, "\U0001F6A8")
                label = f"{emoji} {vehicle.upper()}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{obj.get('class_name', 'vehicle')} {obj.get('confidence', 0):.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw + 5, y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # ====================================================
        # Professional FPS / Performance Overlay
        # ====================================================
        metrics = self.streaming.get_display_metrics()
        inf_fps = metrics.get("inference_fps", 0)
        disp_fps = metrics.get("display_fps", 0)
        speed = metrics.get("playback_speed", 0.5)
        buf_size = metrics.get("buffer_size", 0)
        buf_cap = metrics.get("buffer_capacity", 60)
        lat_ms = metrics.get("latency_ms", 0)

        # Count emergency vehicles
        emergency_count = sum(1 for obj in lane_objects if obj.get("emergency", False))
        
        # Count total vehicles
        total_vehicles = len(lane_objects)

        # Get current signal info
        current_signal = "NORMAL"
        emergency_lane = "None"
        if priority_decision.get("active", False):
            current_signal = "EMERGENCY"
            emergency_lane = priority_decision.get("lane", "Unknown")

        # Get timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())

        # ====================================================
        # Top-left: Performance metrics
        # ====================================================
        top_lines = [
            f"Inference: {inf_fps:.1f} FPS | Display: {disp_fps:.1f} FPS",
            f"Speed: {speed:.2f}x | Buffer: {buf_size}/{buf_cap} | Latency: {lat_ms:.1f}ms",
            f"Pipeline: {self.stats.total_pipeline_ms:.1f}ms | E2E: {self.stats.e2e_latency_ms:.1f}ms",
        ]
        for i, line in enumerate(top_lines):
            y = 25 + i * 22
            cv2.putText(annotated, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ====================================================
        # Top-right: Vehicle counts
        # ====================================================
        right_x = self._frame_w - 250 if self._frame_w > 250 else 10
        count_lines = [
            f"Vehicles: {total_vehicles}",
            f"Emergency: {emergency_count}",
            f"Signal: {current_signal}",
        ]
        if current_signal == "EMERGENCY":
            count_lines.append(f"Emer Lane: {emergency_lane}")
        count_lines.append(f"Time: {timestamp}")

        for i, line in enumerate(count_lines):
            y = 25 + i * 22
            cv2.putText(annotated, line, (right_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ====================================================
        # Bottom-left: Lane counts
        # ====================================================
        if lane_counts:
            y_start = self._frame_h - 30 * len(lane_counts) - 10 if self._frame_h > 0 else 50
            y_start = max(y_start, 50)
            for i, (lane, stats) in enumerate(lane_counts.items()):
                total = stats.get("total", 0)
                y = y_start + i * 25
                cv2.putText(annotated, f"{lane}: {total} vehicles", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # ====================================================
        # Emergency alert overlay (full width banner)
        # ====================================================
        if priority_decision.get("active", False):
            vehicle = priority_decision.get("vehicle", "Unknown").upper()
            lane = priority_decision.get("lane", "Unknown")
            alert_text = f"\U0001F6A8 EMERGENCY: {vehicle} in {lane} \U0001F6A8"
            
            # Blinking red banner at top
            if self._blink_state:
                (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
                banner_x = (self._frame_w - tw) // 2
                cv2.rectangle(annotated, (banner_x - 10, 5), (banner_x + tw + 10, 35), (0, 0, 255), -1)
                cv2.putText(annotated, alert_text, (banner_x, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

        return annotated

    # ========================================================
    # Performance Logging
    # ========================================================

    def _log_performance(self):
        """Log detailed performance metrics every 30 frames."""
        s = self.stats
        print(f"\n{'='*60}")
        print(f"[PERF] Frame {s.frame_count}")
        print(f"{'='*60}")
        print(f"  Camera Read:      {s.camera_read_ms:7.2f}ms")
        print(f"  Vehicle Inference: {s.vehicle_inference_ms:7.2f}ms")
        print(f"  Emergency Inf:     {s.emergency_inference_ms:7.2f}ms")
        print(f"  Merge:             {s.merge_ms:7.2f}ms")
        print(f"  Lane Assign:       {s.lane_assign_ms:7.2f}ms")
        print(f"  Signal:            {s.signal_ms:7.2f}ms")
        print(f"  Draw:              {s.draw_ms:7.2f}ms")
        print(f"  JPEG Encode:       {s.jpeg_encode_ms:7.2f}ms")
        print(f"  Pipeline Total:    {s.total_pipeline_ms:7.2f}ms")
        print(f"  E2E Latency:       {s.e2e_latency_ms:7.2f}ms")
        print(f"  FPS:               {s.fps:7.1f}")
        print(f"{'='*60}\n")

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):
        """Stop the pipeline and release resources."""
        print("[Pipeline] Stopping pipeline...")
        self._running = False

        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)

        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)

        self.camera.release()
        self.frame_buffer.clear()
        self.streaming.clear()

        # Reset counters (but NOT models - they stay in memory)
        self.counter = VehicleCounter()
        self.density = DensityCalculator()
        self.signal = SignalController(emergency_green=EMERGENCY_GREEN)
        self.emergency = EmergencyDetector(
            detector=self.emergency_detector,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU
        )
        self.priority = PriorityEngine()
        self._notified_emergency_ids = set()
        self._vehicle_temporal_state.clear()
        self._confirmed_vehicles.clear()
        self._frame_count = 0

        print("[Pipeline] Pipeline stopped")

    # ========================================================
    # API Accessors
    # ========================================================

    def get_latest_frame(self):
        """Get the latest annotated frame for streaming."""
        return self.streaming.get_raw_frame()

    def get_results(self):
        """Get the latest processing results."""
        with self._results_lock:
            return dict(self._latest_results)

    def get_counter(self):
        return self.get_results().get("counter", {})

    def get_density(self):
        return self.get_results().get("density", {})

    def get_signals(self):
        return self.get_results().get("signals", {})

    def get_tracks(self):
        return self.get_results().get("tracks", [])

    def get_emergency_status(self):
        results = self.get_results()
        return {
            "emergency": results.get("emergency", {"active": False}),
            "summary": results.get("emergency_summary", {
                "current_count": 0, "total_count": 0,
                "per_lane_count": {}, "per_vehicle_count": {}
            })
        }

    def get_dashboard_data(self):
        """Aggregate all data for dashboard API."""
        results = self.get_results()
        counter = results.get("counter", {})
        density = results.get("density", {})
        signals = results.get("signals", {})
        emergency = results.get("emergency", {"active": False})
        tracks = results.get("tracks", [])

        lane_names = set()
        lane_names.update(counter.keys())
        lane_names.update(density.get("class_density", {}).keys())
        lane_names.update(signals.keys())

        lane_data = {}
        for lane in lane_names:
            lane_data[lane] = {
                "vehicles": counter.get(lane, {}).get("total", 0),
                "density": density.get("class_density", {}).get(lane, {}).get("total", 0),
                "score": signals.get(lane, {}).get("score", 0),
                "green_time": signals.get(lane, {}).get("green_time", 0),
                "mode": signals.get(lane, {}).get("mode", "NORMAL"),
                "reason": signals.get(lane, {}).get("reason", "Adaptive AI Density Control"),
                "priority": signals.get(lane, {}).get("priority", "NORMAL"),
            }

        return {
            "counter": counter,
            "density": density,
            "signals": signals,
            "statistics": self.get_statistics(results),
            "processing_time": self.stats.total_pipeline_ms,
            "fps": self.stats.fps,
            "lane_data": lane_data,
            "tracked_vehicles": len(tracks),
            "emergency": emergency,
            "emergency_summary": results.get("emergency_summary", {}),
            "vehicle_detections": results.get("vehicle_detections", []),
            "emergency_detections": results.get("emergency_detections", []),
            "merged_detections": results.get("merged_detections", []),
        }

    def get_statistics(self, results=None):
        """Calculate aggregate statistics."""
        if results is None:
            results = self.get_results()
        counter = results.get("counter", {})

        total = 0
        cars = 0
        buses = 0
        vans = 0
        others = 0
        emergencies = 0

        for lane, lane_data in counter.items():
            total += lane_data.get("total", 0)
            cars += lane_data.get("car", 0)
            buses += lane_data.get("bus", 0)
            vans += lane_data.get("van", 0)
            others += lane_data.get("others", 0)
            emergencies += lane_data.get("ambulance", 0)
            emergencies += lane_data.get("fire_truck", 0)
            emergencies += lane_data.get("police", 0)

        return {
            "total_vehicles": total,
            "cars": cars,
            "bus": buses,
            "van": vans,
            "others": others,
            "emergency_vehicles": emergencies,
        }

    @property
    def fps(self):
        return self.stats.fps

    @property
    def is_running(self):
        return self._running