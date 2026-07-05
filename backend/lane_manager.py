"""
============================================================
Lane Manager

Author : Vamsi Krishna

Description:
Assign tracked vehicles to traffic lanes using polygons.
Each vehicle is assigned to a lane only once.
============================================================
"""

import json
from pathlib import Path

from shapely.geometry import Point
from shapely.geometry import Polygon


class LaneManager:

    def __init__(self, lane_config_path):

        lane_config_path = Path(lane_config_path)

        if not lane_config_path.exists():
            raise FileNotFoundError(
                f"Lane configuration not found:\n{lane_config_path}"
            )

        with open(lane_config_path, "r") as f:
            lane_data = json.load(f)

        self.lanes = {}

        for lane_name, points in lane_data.items():

            self.lanes[lane_name] = Polygon(points)

        # Store lane assignment permanently
        self.vehicle_lane = {}

        print("=" * 60)
        print("Lane Manager Initialized")
        print("=" * 60)

        for lane in self.lanes:
            print(lane)

        print("=" * 60)

    # ---------------------------------------------------------

    def get_lane(self, center):

        point = Point(center)

        for lane_name, polygon in self.lanes.items():

            if polygon.contains(point):
                return lane_name

        return None

    # ---------------------------------------------------------

    def assign_lanes(self, tracked_objects):

        results = []

        for obj in tracked_objects:

            vehicle_id = obj["track_id"]

            # Vehicle already assigned
            if vehicle_id in self.vehicle_lane:

                obj["lane"] = self.vehicle_lane[vehicle_id]

            else:

                lane = self.get_lane(obj["center"])

                obj["lane"] = lane

                if lane is not None:
                    self.vehicle_lane[vehicle_id] = lane

            results.append(obj)

        return results

    # ---------------------------------------------------------

    def print_lane_assignments(self, tracked_objects):

        print("=" * 80)
        print("Lane Assignment")
        print("=" * 80)

        for obj in tracked_objects:

            print(
                f"ID:{obj['track_id']:3d}"
                f" | {obj['class_name']:8s}"
                f" | Lane:{obj['lane']}"
            )

    # ---------------------------------------------------------

    def get_lane_statistics(self):

        stats = {}

        for lane in self.vehicle_lane.values():

            stats[lane] = stats.get(lane, 0) + 1

        return stats