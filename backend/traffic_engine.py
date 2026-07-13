"""
============================================================
Traffic Engine

Author : Vamsi Krishna

Description:
Integrates all AI modules into one engine.
Acts as the central AI processing pipeline.
============================================================
"""

import time

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.counter import VehicleCounter
from backend.density import DensityCalculator
from backend.signal_controller import SignalController
from backend.emergency import EmergencyDetector
from backend.priority_engine import PriorityEngine
from backend.utils import build_class_map, set_class_map
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

        self.emergency_summary = {
            "current_count": 0,
            "total_count": 0,
            "per_lane_count": {},
            "per_vehicle_count": {}
        }

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

        self.latest_emergency = {
            "active": False
        }

        self.latest_vehicle_detections = []
        self.latest_emergency_detections = []
        self.latest_merged_detections = []

        self.emergency_summary = {
            "current_count": 0,
            "total_count": 0,
            "per_lane_count": {},
            "per_vehicle_count": {}
        }

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
        # ----------------------------------------------------
        # tracker.track_frame already calls model.track internally.
        # We do NOT need a separate model.predict() call.

        tracks, annotated, vehicle_track_result = self.tracker.track_frame(
            frame,
            conf=0.30,
            return_results=True
        )

        # Save raw vehicle detections for debugging
        self.latest_vehicle_detections = self.tracker.detector.get_detections(vehicle_track_result)
        print(f"[Engine] Vehicle detections: {len(self.latest_vehicle_detections)} objects")

        # ----------------------------------------------------
        # Step 2: Lane Assignment
        # ----------------------------------------------------

        try:
            h, w = frame.shape[:2]
            self.lane_manager.update_scale(w, h)
        except Exception:
            pass

        lane_objects = self.lane_manager.assign_lanes(tracks)
        print(f"[Engine] Lane assignment complete. Sample: {lane_objects[:3]}")

        # ----------------------------------------------------
        # Step 3: Emergency Detection
        # ----------------------------------------------------
        # Run emergency model separately for ambulance/firetruck/police

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
        print(f"[Engine] Emergency detections after lane/class normalization: {len(emergency_list)} objects")

        # ----------------------------------------------------
        # Step 4: Match emergency to tracked vehicles (BEFORE merging)
        # ----------------------------------------------------

        if len(emergency_list) > 0:
            for emergency in emergency_list:
                track = self._find_matching_track(emergency, lane_objects)
                if track is not None:
                    track["emergency"] = True
                    track["emergency_vehicle"] = emergency["vehicle"]
                    track["emergency_confidence"] = emergency["confidence"]
                    track["emergency_track_id"] = emergency["track_id"]
                    emergency["matched_track_id"] = track["track_id"]
                else:
                    emergency["matched_track_id"] = None

        # ----------------------------------------------------
        # Step 5: Merge Detections (AFTER matching)
        # ----------------------------------------------------

        combined_objects = list(lane_objects)

        for emergency in emergency_list:
            # Only add unmatched emergencies as new objects
            if emergency.get("matched_track_id") is None:
                lane = emergency.get("lane")
                combined_objects.append({
                    "track_id": emergency["track_id"],
                    "class_name": emergency["class_name"],
                    "confidence": emergency["confidence"],
                    "bbox": emergency["bbox"],
                    "center": emergency["center"],
                    "lane": lane,
                    "source": "emergency_model",
                    "emergency": True,
                    "vehicle": emergency["vehicle"]
                })
            else:
                print(f"[Engine] Emergency vehicle {emergency['vehicle']} matched to track {emergency['matched_track_id']}")

        self.latest_merged_detections = combined_objects
        print(f"[Engine] Merged detections: {len(combined_objects)} objects")

        # ----------------------------------------------------
        # Step 6: Priority Engine
        # ----------------------------------------------------

        priority_decision = self.priority.evaluate(emergency_list)
        print(f"[Engine] Priority decision: {priority_decision}")

        # ----------------------------------------------------
        # Step 7: Vehicle Counter (use merged objects to count emergency vehicles too)
        # ----------------------------------------------------

        self.counter.update(combined_objects)
        lane_counts = self.counter.get_counts()

        # ----------------------------------------------------
        # Step 8: Density Calculation (use merged objects for accurate density)
        # ----------------------------------------------------

        lane_density, class_density = self.density.calculate_density(
            combined_objects
        )

        # ----------------------------------------------------
        # Step 9: Adaptive Signal Controller
        # ----------------------------------------------------

        signal_plan = self.signal.generate_signal_plan(
            class_density,
            priority_decision
        )

        print(f"[Engine] Signal plan: {signal_plan}")
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
            "merged_detections": self.latest_merged_detections
        }

    # ========================================================
    # Emergency Matching Helpers
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

            # Count emergency vehicles too
            emergencies += lane_data.get("ambulance", 0)
            emergencies += lane_data.get("fire_truck", 0)
            emergencies += lane_data.get("police", 0)

        stats = {

            "total_vehicles": total,

            "cars": cars,

            "bus": buses,

            "van": vans,

            "others": others,

            "emergency_vehicles": emergencies

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
                "priority": self.latest_signals.get(lane, {}).get("priority", "NORMAL")
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
                "green_time": self.current_green_time
            },

            "lane_data": lane_data,

            "tracked_vehicles": len(self.latest_tracks),

            "emergency": self.latest_emergency,

            "emergency_summary": self.emergency.get_summary(),

            "vehicle_detections": self.latest_vehicle_detections,

            "emergency_detections": self.latest_emergency_detections,

            "merged_detections": self.latest_merged_detections

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

        self.latest_emergency = { "active": False }
        self.latest_vehicle_detections = []
        self.latest_emergency_detections = []
        self.latest_merged_detections = []
        self.emergency_summary = {
            "current_count": 0,
            "total_count": 0,
            "per_lane_count": {},
            "per_vehicle_count": {}
        }

        print("=" * 60)
        print("Traffic Engine Reset")
        print("=" * 60)