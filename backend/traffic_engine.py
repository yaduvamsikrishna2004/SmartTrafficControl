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


class TrafficEngine:

    def __init__(self, model_path, lane_config):

        # ====================================================
        # AI Modules
        # ====================================================

        self.tracker = VehicleTracker(model_path)

        self.lane_manager = LaneManager(lane_config)

        self.counter = VehicleCounter()

        self.density = DensityCalculator()

        self.signal = SignalController()

        # ====================================================
        # Runtime Information
        # ====================================================

        self.processing_time = 0

        self.fps = 0

        # ====================================================
        # Latest Results
        # ====================================================

        self.latest_frame = None

        self.latest_tracks = []

        self.latest_counter = {}

        self.latest_density = {}

        self.latest_signals = {}

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

        # ----------------------------------------------------
        # Lane Assignment
        # ----------------------------------------------------

        lane_objects = self.lane_manager.assign_lanes(tracks)

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

        signal_plan = self.signal.generate_signal_plan(
            class_density
        )

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

        # ----------------------------------------------------
        # Processing Time
        # ----------------------------------------------------

        end = time.perf_counter()

        self.processing_time = round(

            (end - start) * 1000,

            2

        )

        return {

            "frame": annotated,

            "tracks": lane_objects,

            "counter": lane_counts,

            "density": self.latest_density,

            "signals": signal_plan,

            "processing_time": self.processing_time

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
    # Get Frame
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

        return {

            "total_vehicles": total,

            "cars": cars,

            "bus": buses,

            "van": vans,

            "others": others

        }

    # ========================================================
    # Dashboard Data
    # ========================================================

    def get_dashboard_data(self):

        return {

            "counter": self.latest_counter,

            "density": self.latest_density,

            "signals": self.latest_signals,

            "statistics": self.get_statistics(),

            "processing_time": self.processing_time,

            "tracked_vehicles": len(self.latest_tracks)

        }

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        self.counter = VehicleCounter()

        self.density = DensityCalculator()

        self.signal = SignalController()

        self.processing_time = 0

        self.latest_frame = None

        self.latest_tracks = []

        self.latest_counter = {}

        self.latest_density = {}

        self.latest_signals = {}

        print("=" * 60)
        print("Traffic Engine Reset")
        print("=" * 60)