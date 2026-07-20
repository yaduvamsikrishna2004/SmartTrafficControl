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
    EMERGENCY_PRIORITY_ORDER, FRAME_SKIP
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

        while self._running:
            read_start = time.perf_counter()
            ret, frame = self.camera.read()
            read_elapsed = (time.perf_counter() - read_start) * 1000
            self.stats.camera_read_ms = read_elapsed

            if not ret:
                if self._running:
                    time.sleep(0.01)
                continue

            # Cache frame dimensions (set once)
            if self._frame_h == 0:
                self._frame_h, self._frame_w = frame.shape[:2]
                try:
                    self.lane_manager.update_scale(self._frame_w, self._frame_h)
                except Exception:
                    pass

            # Increment global frame counter
            self._frame_count += 1
            frame_id = self._frame_count

            # Push to frame buffer (always latest)
            self.frame_buffer.set(frame, frame_id)

            # ====================================================
            # IMMEDIATE STREAMING: First frame goes out RIGHT AWAY
            # without waiting for AI inference.
            # ====================================================
            if frame_id == 1:
                # First frame: stream raw (no annotations) immediately
                self.streaming.update_frame(frame)
                print(f"[Pipeline] FIRST FRAME streamed immediately (frame 1)")
            else:
                # Subsequent frames: stream annotated version
                pass  # Streaming updated by inference thread

            # Small sleep to prevent busy-waiting
            time.sleep(0.001)

        print("[Pipeline] Camera reader thread stopped")

    # ========================================================
    # THREAD 2+3+4: AI Inference + Post-Processing
    # Runs on the latest frame buffer.
    # Skips frames if inference is backlogged.
    # ========================================================

    def _inference_loop(self):
        """Dedicated thread for AI inference and post-processing."""
        print("[Pipeline] AI inference thread started")

        last_processed_id = 0
        perf_start = time.perf_counter()

        while self._running:
            # ====================================================
            # Get the latest frame (skip stale ones)
            # ====================================================
            frame, frame_id = self.frame_buffer.get()
            if frame is None or frame_id == last_processed_id:
                time.sleep(0.002)
                continue

            # Skip if we're too far behind (frame_id gap > 2 means inference is slow)
            # Always process every frame to maintain smooth tracking
            last_processed_id = frame_id

            # ====================================================
            # PERFORMANCE: Start end-to-end timing
            # ====================================================
            e2e_start = time.perf_counter()
            pipeline_start = time.perf_counter()

            self._frame_skip_counter += 1

            # ====================================================
            # STEP 1: Vehicle Detection (best.pt)
            # ====================================================
            v_start = time.perf_counter()
            vehicle_detections, vehicle_raw = self._run_vehicle_inference(frame)
            veh_time = (time.perf_counter() - v_start) * 1000
            self.stats.vehicle_inference_ms = veh_time

            # ====================================================
            # STEP 2: Emergency Detection (emergency_best.pt)
            # ====================================================
            e_start = time.perf_counter()
            emergency_detections = self._run_emergency_inference(frame)
            emerg_time = (time.perf_counter() - e_start) * 1000
            self.stats.emergency_inference_ms = emerg_time

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

            # Lane assignment
            la_start = time.perf_counter()
            lane_objects = self.lane_manager.assign_lanes(merged)
            lane_time = (time.perf_counter() - la_start) * 1000
            self.stats.lane_assign_ms = lane_time

            # Tracker class update
            lane_objects = self._update_tracker_classes(lane_objects, emergency_detections)

            # Emergency mode trigger
            self._trigger_emergency(priority_decision, lane_objects)

            # Counter + Density
            self.counter.update(lane_objects)
            lane_counts = self.counter.get_counts()
            lane_density, class_density = self.density.calculate_density(lane_objects)

            # Signal
            sig_start = time.perf_counter()
            signal_plan = self.signal.generate_signal_plan(class_density, priority_decision)
            sig_time = (time.perf_counter() - sig_start) * 1000
            self.stats.signal_ms = sig_time

            # ====================================================
            # STEP 4: Draw Results
            # ====================================================
            draw_start = time.perf_counter()
            annotated = self._draw_results(frame, lane_objects, vehicle_detections, emergency_detections)
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

        print("[Pipeline] AI inference thread stopped")

    # ========================================================
    # Vehicle Inference
    # ========================================================

    def _run_vehicle_inference(self, frame):
        """Run vehicle detection + tracking."""
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

    # ========================================================
    # Emergency Inference
    # ========================================================

    def _run_emergency_inference(self, frame):
        """Run emergency detection model."""
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
    # Drawing
    # ========================================================

    def _draw_results(self, frame, lane_objects, vehicle_detections, emergency_detections):
        """Draw detection results on frame. Optimized to minimize copies."""
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

        for obj in lane_objects:
            bbox = obj.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            is_emergency = obj.get("emergency", False)

            if is_emergency:
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

        # FPS overlay
        fps_text = f"FPS: {self.stats.fps:.1f} | Frame: {self._frame_count}"
        cv2.putText(annotated, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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