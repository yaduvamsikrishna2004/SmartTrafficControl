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

    def __init__(self, model_path, lane_config):

        # -------------------------------
        # AI Modules
        # -------------------------------

        self.tracker = VehicleTracker(model_path)

        self.lane_manager = LaneManager(lane_config)

        self.counter = VehicleCounter()

        self.density = DensityCalculator()

        self.signal = SignalController()

        # -------------------------------
        # Latest Results
        # -------------------------------

        self.latest_frame = None

        self.latest_tracks = []

        self.latest_counter = {}

        self.latest_density = {}

        self.latest_signals = {}

        print("=" * 60)
        print("Traffic Engine Initialized")
        print("=" * 60)

    # =========================================================

    def process_frame(self, frame):

        """
        Process one frame through the complete AI pipeline.
        """

        # -------------------------------
        # Tracking
        # -------------------------------

        tracks, annotated = self.tracker.track_frame(frame)

        # -------------------------------
        # Lane Assignment
        # -------------------------------

        lane_objects = self.lane_manager.assign_lanes(tracks)

        # -------------------------------
        # Vehicle Counter
        # -------------------------------

        self.counter.update(lane_objects)

        lane_counts = self.counter.get_counts()

        # -------------------------------
        # Density
        # -------------------------------

        _, class_density = self.density.calculate_density(
            lane_objects
        )

        # -------------------------------
        # Signal Controller
        # -------------------------------

        signal_plan = self.signal.generate_signal_plan(
            class_density
        )

        # -------------------------------
        # Save Latest Results
        # -------------------------------

        self.latest_frame = annotated

        self.latest_tracks = lane_objects

        self.latest_counter = lane_counts

        self.latest_density = class_density

        self.latest_signals = signal_plan

        return {

            "frame": annotated,

            "tracks": lane_objects,

            "counter": lane_counts,

            "density": class_density,

            "signals": signal_plan

        }

    # =========================================================

    def get_counter(self):

        """
        Returns cumulative vehicle counts.
        """

        return self.latest_counter

    # =========================================================

    def get_density(self):

        """
        Returns current lane density.
        """

        return self.latest_density

    # =========================================================

    def get_signals(self):

        """
        Returns adaptive signal timings.
        """

        return self.latest_signals

    # =========================================================

    def get_tracks(self):

        """
        Returns latest tracked vehicles.
        """

        return self.latest_tracks

    # =========================================================

    def get_latest_frame(self):

        """
        Returns latest annotated frame.
        """

        return self.latest_frame

    # =========================================================

    def get_statistics(self):

        """
        Returns everything required by the frontend.
        """

        return {

            "counter": self.latest_counter,

            "density": self.latest_density,

            "signals": self.latest_signals,

            "tracks": self.latest_tracks

        }

    # =========================================================

    def reset(self):

        """
        Reset all statistics.
        """

        self.counter = VehicleCounter()

        self.latest_frame = None

        self.latest_tracks = []

        self.latest_counter = {}

        self.latest_density = {}

        self.latest_signals = {}

        print("=" * 60)
        print("Traffic Engine Reset")
        print("=" * 60)