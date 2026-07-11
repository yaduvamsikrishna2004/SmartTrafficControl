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


class TrafficEngine:

    def __init__(self, model_path, lane_config):

        # ====================================================
        # AI Modules
        # ====================================================

        self.tracker = VehicleTracker(model_path)

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

        self.signal = SignalController()
        self.emergency = EmergencyDetector()
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

        self.latest_emergency = {

            "active": False

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
        # Vehicle Tracking
        # ----------------------------------------------------

        tracks, annotated = self.tracker.track_frame(frame)
        print(f"[Engine] Tracked {len(tracks)} objects")

        # ----------------------------------------------------
        # Lane Assignment
        # ----------------------------------------------------

        # update lane polygons to match current frame size
        try:
            h, w = frame.shape[:2]
            self.lane_manager.update_scale(w, h)
        except Exception:
            pass

        lane_objects = self.lane_manager.assign_lanes(tracks)
        print(f"[Engine] Lane assignment complete. Sample: {lane_objects[:3]}")

        # ----------------------------------------------------
        # Emergency Detection
        # ----------------------------------------------------

        emergency_list = self.emergency.detect(lane_objects)
        print(f"[Engine] Emergency detection found: {emergency_list}")

        # Evaluate priority decision (convert list -> decision dict)
        priority_decision = self.priority.evaluate(emergency_list)
        print(f"[Engine] Priority decision: {priority_decision}")

        # ----------------------------------------------------
        # Vehicle Counter
        # ----------------------------------------------------

        self.counter.update(lane_objects)

        lane_counts = self.counter.get_counts()

        # ----------------------------------------------------
        # Density Calculation
        # ----------------------------------------------------

        lane_density, class_density = self.density.calculate_density(
            lane_objects
        )

        # ----------------------------------------------------
        # Adaptive Signal Controller
        # ----------------------------------------------------

        # pass priority_decision (dict) to signal planner
        signal_plan = self.signal.generate_signal_plan(
            class_density,
            priority_decision
        )

        print(f"[Engine] Signal plan: {signal_plan}")

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

        self.latest_confidence = 0.0
        if len(lane_objects) > 0:
            self.latest_confidence = round(
                sum(obj.get("confidence", 0) for obj in lane_objects)
                / len(lane_objects),
                3
            )

        self.current_green_lane = None
        self.current_green_time = 0
        if len(signal_plan) > 0:
            first_lane = next(iter(signal_plan))
            self.current_green_lane = first_lane
            self.current_green_time = signal_plan[first_lane].get("green_time", 0)

        # ----------------------------------------------------
        # Processing Time
        # ----------------------------------------------------

        end = time.perf_counter()

        self.processing_time = round(

            (end - start) * 1000,

            2

        )

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

            "emergency": priority_decision

        }

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

        for lane in self.latest_counter.values():

            total += lane.get("total", 0)

            cars += lane.get("car", 0)

            buses += lane.get("bus", 0)

            vans += lane.get("van", 0)

            others += lane.get("others", 0)

        stats = {

            "total_vehicles": total,

            "cars": cars,

            "bus": buses,

            "van": vans,

            "others": others

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

            "emergency": self.latest_emergency

        }

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        self.counter = VehicleCounter()

        self.density = DensityCalculator()

        self.signal = SignalController()

        self.emergency = EmergencyDetector()

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

        self.latest_emergency = {

            "active": False

        }

        print("=" * 60)
        print("Traffic Engine Reset")
        print("=" * 60)