"""
============================================================
Traffic Engine

Author : Vamsi Krishna

Description:
Integrates all AI modules into one engine.
============================================================
"""

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.counter import VehicleCounter
from backend.density import DensityCalculator
from backend.signal_controller import SignalController


class TrafficEngine:

    def __init__(
        self,
        model_path,
        lane_config
    ):

        self.tracker = VehicleTracker(model_path)

        self.lane_manager = LaneManager(lane_config)

        self.counter = VehicleCounter()

        self.density = DensityCalculator()

        self.signal = SignalController()

        print("=" * 60)
        print("Traffic Engine Initialized")
        print("=" * 60)

    # -----------------------------------------------------

    def process_frame(self, frame):

        tracks, annotated = self.tracker.track_frame(frame)

        lane_objects = self.lane_manager.assign_lanes(tracks)

        self.counter.update(lane_objects)

        lane_counts = self.counter.get_counts()

        _, class_density = self.density.calculate_density(
            lane_objects
        )

        signal_plan = self.signal.generate_signal_plan(
            class_density
        )

        return {

            "frame": annotated,

            "tracks": lane_objects,

            "counter": lane_counts,

            "density": class_density,

            "signals": signal_plan

        }