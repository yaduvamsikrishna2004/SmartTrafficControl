"""
============================================================
Traffic Engine

Author : Vamsi Krishna

Description:
Integrates all AI modules into one engine.
Acts as the central AI processing pipeline.

Architecture: best.pt + emergency_best.pt (2 models)
- best.pt: General vehicle detection, tracking, counting
- emergency_best.pt: Emergency vehicle detection and override

Emergency Override Rule:
If emergency model predicts ambulance/firetruck/police
AND IoU with tracked vehicle > threshold (0.3)
THEN emergency class ALWAYS wins and overrides ALL fields.
============================================================
"""

import time
import numpy as np

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.counter import VehicleCounter
from backend.density import DensityCalculator
from backend.signal_controller import SignalController
from backend.emergency import EmergencyDetector
from backend.priority_engine import PriorityEngine
from backend.utils import build_class_map, set_class_map, normalize_emergency_class, is_emergency_class
from backend.config import EMERGENCY_MODEL_PATH, EMERGENCY_CONF, EMERGENCY_IOU, EMERGENCY_GREEN
from backend.detector import YOLODetector


class TrafficEngine:

    def __init__(self, model_path, lane_config):

        # ====================================================
        # AI Modules
        # ====================================================

        self.tracker = VehicleTracker(model_path)

        self.emergency_detector = YOLODetector(
            EMERGENCY_MODEL_PATH,
            model_label="Emergency model loaded"
        )

        # Build and set a class map from the model's class names so
        # counting/density mapping is precise for the current model.
        try:
            class_names = self.tracker.detector.get_class_names()
            mapping = build_class_map(class_names)
            set_class_map(mapping)
            print(f"[Engine] Class map set: {mapping}")
        except Exception:
            pass

        self.lane_manager = LaneManager(lane_config)

        self.counter = VehicleCounter()

        self.density = DensityCalculator()

        self.signal = SignalController(emergency_green=EMERGENCY_GREEN)
        self.emergency = EmergencyDetector(
            detector=self.emergency_detector,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU
        )
        self.priority = PriorityEngine()

        # ====================================================
        # Runtime Information
        # ====================================================

        self.processing_time = 0
        self.fps = 0
        self.latest_confidence = 0.0
        self.current_green_lane = None
        self.current_green_time = 0

        # ====================================================
        # Latest Results
        # ====================================================

        self.latest_frame = None
        self.latest_tracks = []
        self.latest_counter = {}
        self.latest_density = {}
        self.latest_signals = {}
        self.latest_emergency = {"active": False}
        self.latest_vehicle_detections = []
        self.latest_emergency_detections = []
        self.latest_merged_detections = []

        print("=" * 60)
        print("Traffic Engine Initialized")
        print("=" * 60)

    # ========================================================
    # Process One Frame
    # ========================================================

    def process_frame(self, frame):

        start = time.perf_counter()

        # ----------------------------------------------------
        # Step 1: Vehicle Detection via Tracker (YOLO + ByteTrack)
        # Run ONCE per frame — no duplicate inference
        # ----------------------------------------------------
        tracks, annotated, vehicle_track_result = self.tracker.track_frame(
            frame,
            conf=0.30,
            return_results=True
        )

        # Vehicle model detections for debug
        self.latest_vehicle_detections = self.tracker.detector.get_detections(vehicle_track_result)
        print(f"[Engine] Step 1 - Vehicle model output: {len(self.latest_vehicle_detections)} detections")
        for det in self.latest_vehicle_detections:
            print(f"  [Vehicle] {det['class_name']} | Conf: {det['confidence']} | BBox: {det['bbox']}")

        # ----------------------------------------------------
        # Step 2: Lane Assignment
        # ----------------------------------------------------
        try:
            h, w = frame.shape[:2]
            self.lane_manager.update_scale(w, h)
        except Exception:
            pass

        lane_objects = self.lane_manager.assign_lanes(tracks)
        print(f"[Engine] Step 2 - Lane assignment: {len(lane_objects)} objects assigned")
        for obj in lane_objects[:5]:
            print(f"  [Lane] Track {obj['track_id']} | {obj['class_name']} | Lane: {obj['lane']}")

        # ----------------------------------------------------
        # Step 3: Emergency Detection
        # Run emergency model ONCE per frame — no duplicate inference
        # ----------------------------------------------------
        print(f"[Engine] Step 3 - Running emergency model...")

        emergency_raw_results = self.emergency_detector.model.track(
            source=frame,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        emergency_list = self.emergency.detect(
            frame,
            self.lane_manager,
            raw_results=emergency_raw_results
        )

        self.latest_emergency_detections = list(emergency_list)
        print(f"[Engine] Step 3 - Emergency model output: {len(emergency_list)} detections")
        for emerg in emergency_list:
            print(f"  [Emergency] {emerg['vehicle']} | Conf: {emerg['confidence']} | Lane: {emerg['lane']} | BBox: {emerg['bbox']} | TrackID: {emerg['track_id']}")

        # ----------------------------------------------------
        # Step 4: CRITICAL - Emergency Override Logic
        # When emergency prediction matches a tracked vehicle,
        # override ALL class-related fields.
        #
        # RULE: Emergency model ALWAYS wins if IoU > threshold (0.3)
        # ----------------------------------------------------
        print(f"[Engine] Step 4 - Matching emergencies to tracked vehicles...")

        if len(emergency_list) > 0:
            for emergency in emergency_list:
                track = self._find_matching_track_bbox(emergency, lane_objects)

                if track is not None:
                    # Compute IoU between emergency bbox and track bbox
                    iou = self._compute_iou(emergency["bbox"], track.get("bbox"))

                    if iou >= 0.3:
                        # EMERGENCY OVERRIDE: Replace ALL class-related fields
                        emergency_class = normalize_emergency_class(emergency["vehicle"])

                        print(f"  [OVERRIDE] Track {track['track_id']}: {track['class_name']} -> {emergency_class} (IoU: {iou:.2f})")

                        # Update EVERY class-related field to prevent any downstream module
                        # from reading the old (wrong) class
                        track["class_name"] = emergency_class
                        track["vehicle_type"] = emergency_class
                        track["display_name"] = emergency_class
                        track["label"] = emergency_class
                        track["vehicle"] = emergency_class
                        track["emergency_vehicle"] = emergency_class
                        track["emergency"] = True
                        track["is_emergency"] = True
                        track["priority"] = "HIGH"
                        track["emergency_confidence"] = emergency["confidence"]
                        track["emergency_track_id"] = emergency["track_id"]
                        track["confidence"] = emergency["confidence"]  # Use emergency model's confidence
                        track["dashboard_class"] = emergency_class
                        track["dashboard_label"] = emergency_class.upper()
                        track["override_active"] = True
                        track["override_reason"] = "Emergency Model Override"

                        # Mark emergency as matched
                        emergency["matched_track_id"] = track["track_id"]
                        emergency["matched"] = True

                        print(f"  [OVERRIDE SUCCESS] Track {track['track_id']}: final class = {emergency_class}")
                    else:
                        print(f"  [SKIP] Emergency {emergency['vehicle']} IoU {iou:.2f} < 0.3 with Track {track['track_id']}, no override")
                        emergency["matched_track_id"] = None
                else:
                    print(f"  [UNMATCHED] Emergency {emergency['vehicle']} (Track {emergency.get('track_id', 'N/A')}) has no matching tracked vehicle")
                    emergency["matched_track_id"] = None

        # Log tracked objects after override
        for obj in lane_objects:
            if obj.get("emergency"):
                print(f"  [AFTER OVERRIDE] Track {obj['track_id']} | Class: {obj['class_name']} | Emergency: {obj.get('emergency_vehicle', 'N/A')}")

        # ----------------------------------------------------
        # Step 5: Merge Detections
        # Combine: tracked objects + unmatched emergencies
        # ----------------------------------------------------
        combined_objects = list(lane_objects)

        for emergency in emergency_list:
            if emergency.get("matched_track_id") is None:
                lane = emergency.get("lane", "Unknown")
                combined_objects.append({
                    "track_id": emergency["track_id"],
                    "class_name": emergency["class_name"],
                    "vehicle_type": emergency["class_name"],
                    "display_name": emergency["class_name"],
                    "label": emergency["class_name"],
                    "vehicle": emergency["vehicle"],
                    "confidence": emergency["confidence"],
                    "bbox": emergency["bbox"],
                    "center": emergency["center"],
                    "lane": lane,
                    "source": "emergency_model",
                    "emergency": True,
                    "is_emergency": True,
                    "emergency_vehicle": emergency["vehicle"],
                    "priority": "HIGH",
                    "dashboard_class": emergency["class_name"],
                    "dashboard_label": emergency["class_name"].upper(),
                    "override_active": True,
                    "override_reason": "Direct Emergency Detection (No matched track)",
                })
                print(f"  [MERGE] Added unmatched emergency {emergency['vehicle']} to lane {lane}")

        self.latest_merged_detections = combined_objects
        print(f"[Engine] Step 5 - After merge: {len(combined_objects)} total objects")
        for obj in combined_objects:
            cls = obj.get("class_name", "unknown")
            emerg = " [EMERGENCY]" if obj.get("emergency") else ""
            print(f"  [MERGED] Track {obj['track_id']} | {cls}{emerg} | Lane: {obj['lane']}")

        # ----------------------------------------------------
        # Step 6: Priority Engine
        # ----------------------------------------------------
        priority_decision = self.priority.evaluate(emergency_list)
        print(f"[Engine] Step 6 - Priority decision: {priority_decision}")

        # ----------------------------------------------------
        # Step 7: Vehicle Counter (counts once per unique track_id)
        # ----------------------------------------------------
        self.counter.update(combined_objects)
        lane_counts = self.counter.get_counts()
        print(f"[Engine] Step 7 - Vehicle counter: {lane_counts}")

        # ----------------------------------------------------
        # Step 8: Density Calculation (weighted for emergency vehicles)
        # ----------------------------------------------------
        lane_density, class_density = self.density.calculate_density(
            combined_objects
        )
        print(f"[Engine] Step 8 - Density: lane={dict(lane_density)}, class={dict(class_density)}")

        # ----------------------------------------------------
        # Step 9: Adaptive Signal Controller
        # ----------------------------------------------------
        signal_plan = self.signal.generate_signal_plan(
            class_density,
            priority_decision
        )
        print(f"[Engine] Step 9 - Signal plan: {signal_plan}")
        print(f"[Engine] Emergency active: {priority_decision.get('active', False)}")

        # ----------------------------------------------------
        # Save Latest Results
        # ----------------------------------------------------
        self.latest_frame = annotated
        self.latest_tracks = lane_objects
        self.latest_counter = lane_counts

        self.latest_density = {
            "lane_density": dict(lane_density),
            "class_density": dict(class_density)
        }

        self.latest_signals = signal_plan
        self.latest_emergency = priority_decision

        # Average confidence across all tracked objects
        self.latest_confidence = 0.0
        if len(lane_objects) > 0:
            self.latest_confidence = round(
                sum(obj.get("confidence", 0) for obj in lane_objects)
                / len(lane_objects),
                3
            )

        # Current green lane (first lane in signal plan)
        self.current_green_lane = None
        self.current_green_time = 0
        if len(signal_plan) > 0:
            first_lane = next(iter(signal_plan))
            self.current_green_lane = first_lane
            self.current_green_time = signal_plan[first_lane].get("green_time", 0)

        # ----------------------------------------------------
        # Processing Time & FPS
        # ----------------------------------------------------
        end = time.perf_counter()
        self.processing_time = round((end - start) * 1000, 2)
        self.fps = round(1000 / self.processing_time, 2) if self.processing_time > 0 else 0

        print(f"[Engine] Processing time: {self.processing_time}ms | FPS: {self.fps}")

        # ----------------------------------------------------
        # Return Complete Pipeline Result
        # ----------------------------------------------------
        return {
            "frame": annotated,
            "tracks": lane_objects,
            "counter": lane_counts,
            "density": self.latest_density,
            "signals": signal_plan,
            "processing_time": self.processing_time,
            "fps": self.fps,
            "emergency": priority_decision,
            "emergency_summary": self.emergency.get_summary(),
            "vehicle_detections": self.latest_vehicle_detections,
            "emergency_detections": self.latest_emergency_detections,
            "merged_detections": self.latest_merged_detections,
        }

    # ========================================================
    # IoU Computation
    # ========================================================

    def _compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bounding boxes."""
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection coordinates
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)

        # Areas of both boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        if area1 + area2 - inter_area <= 0:
            return 0.0

        return inter_area / (area1 + area2 - inter_area)

    # ========================================================
    # Find matching track by IoU (preferred)
    # ========================================================

    def _find_matching_track_bbox(self, emergency, tracked_objects):
        """
        Find the best matching tracked vehicle for an emergency detection.
        Uses IoU-based matching with fallback to center distance.
        """
        emergency_bbox = emergency.get("bbox")
        emergency_center = emergency.get("center")
        emergency_lane = emergency.get("lane")

        if emergency_bbox is None and emergency_center is None:
            return None

        best_match = None
        best_iou = 0.3  # Minimum IoU threshold

        for obj in tracked_objects:
            # Lane filter (if emergency has a known lane)
            if emergency_lane and emergency_lane != "Unknown":
                if obj.get("lane") != emergency_lane:
                    continue

            obj_bbox = obj.get("bbox")
            if obj_bbox and emergency_bbox:
                iou = self._compute_iou(emergency_bbox, obj_bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_match = obj
                    continue

            # Fallback: center distance
            obj_center = obj.get("center")
            if obj_center and emergency_center:
                ex, ey = emergency_center
                cx, cy = obj_center
                dist = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
                if dist < 50 and best_match is None:
                    best_match = obj

        return best_match

    # ========================================================
    # Legacy: Find matching track by center distance
    # ========================================================

    def _find_matching_track(self, emergency, tracked_objects):

        emergency_center = emergency.get("center")
        emergency_lane = emergency.get("lane")

        if emergency_center is None:
            return None

        ex, ey = emergency_center

        for obj in tracked_objects:

            # If emergency has a valid lane, try to match by lane first
            if emergency_lane and emergency_lane != "Unknown":
                if obj.get("lane") != emergency_lane:
                    continue

            bbox = obj.get("bbox")
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            # Check if emergency center falls within vehicle bbox
            if x1 <= ex <= x2 and y1 <= ey <= y2:
                # Update the vehicle's lane to the emergency's lane if needed
                if emergency_lane and emergency_lane != "Unknown" and obj.get("lane") != emergency_lane:
                    obj["lane"] = emergency_lane
                return obj

            # fallback: compare center distance
            center = obj.get("center")
            if center:
                cx, cy = center
                if abs(cx - ex) <= 50 and abs(cy - ey) <= 50:
                    return obj

        return None

    # ========================================================
    # Get Latest Counter
    # ========================================================

    def get_counter(self):
        return self.latest_counter

    # ========================================================
    # Get Density
    # ========================================================

    def get_density(self):
        return self.latest_density

    # ========================================================
    # Get Signals
    # ========================================================

    def get_signals(self):
        return self.latest_signals

    # ========================================================
    # Get Tracks
    # ========================================================

    def get_tracks(self):
        return self.latest_tracks

    # ========================================================
    # Get Latest Frame
    # ========================================================

    def get_latest_frame(self):
        return self.latest_frame

    # ========================================================
    # Analytics
    # ========================================================

    def get_statistics(self):

        total = 0
        cars = 0
        buses = 0
        vans = 0
        others = 0
        emergencies = 0

        for lane, lane_data in self.latest_counter.items():

            total += lane_data.get("total", 0)
            cars += lane_data.get("car", 0)
            buses += lane_data.get("bus", 0)
            vans += lane_data.get("van", 0)
            others += lane_data.get("others", 0)

            # Count emergency vehicles separately
            emergencies += lane_data.get("ambulance", 0)
            emergencies += lane_data.get("fire_truck", 0)
            emergencies += lane_data.get("police", 0)

        stats = {
            "total_vehicles": total,
            "cars": cars,
            "bus": buses,
            "van": vans,
            "others": others,
            "emergency_vehicles": emergencies,
        }

        if self.latest_confidence is not None:
            stats["confidence"] = f"{round(self.latest_confidence * 100)}%"

        return stats

    # ========================================================
    # Dashboard Data
    # ========================================================

    def get_dashboard_data(self):

        lane_names = set()
        lane_names.update(self.latest_counter.keys())
        lane_names.update(self.latest_density.get("class_density", {}).keys())
        lane_names.update(self.latest_signals.keys())

        lane_data = {}
        for lane in lane_names:
            lane_data[lane] = {
                "vehicles": self.latest_counter.get(lane, {}).get("total", 0),
                "density": self.latest_density.get("class_density", {}).get(lane, {}).get("total", 0),
                "score": self.latest_signals.get(lane, {}).get("score", 0),
                "green_time": self.latest_signals.get(lane, {}).get("green_time", 0),
                "mode": self.latest_signals.get(lane, {}).get("mode", "NORMAL"),
                "reason": self.latest_signals.get(lane, {}).get("reason", "Adaptive AI Density Control"),
                "priority": self.latest_signals.get(lane, {}).get("priority", "NORMAL"),
            }

        return {
            "counter": self.latest_counter,
            "density": self.latest_density,
            "signals": self.latest_signals,
            "statistics": self.get_statistics(),
            "processing_time": self.processing_time,
            "fps": self.fps,
            "confidence": self.latest_confidence,
            "current_green": {
                "lane": self.current_green_lane,
                "green_time": self.current_green_time,
            },
            "lane_data": lane_data,
            "tracked_vehicles": len(self.latest_tracks),
            "emergency": self.latest_emergency,
            "emergency_summary": self.emergency.get_summary(),
            "vehicle_detections": self.latest_vehicle_detections,
            "emergency_detections": self.latest_emergency_detections,
            "merged_detections": self.latest_merged_detections,
        }

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        self.counter = VehicleCounter()
        self.density = DensityCalculator()
        self.signal = SignalController(emergency_green=EMERGENCY_GREEN)

        self.emergency = EmergencyDetector(
            detector=self.emergency_detector,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU
        )

        self.processing_time = 0
        self.fps = 0
        self.latest_confidence = 0.0
        self.current_green_lane = None
        self.current_green_time = 0
        self.latest_frame = None
        self.latest_tracks = []
        self.latest_counter = {}
        self.latest_density = {}
        self.latest_signals = {}
        self.latest_emergency = {"active": False}
        self.latest_vehicle_detections = []
        self.latest_emergency_detections = []
        self.latest_merged_detections = []

        print("=" * 60)
        print("Traffic Engine Reset")
        print("=" * 60)