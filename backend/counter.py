"""
============================================================
Vehicle Counter

Author : Vamsi Krishna

Description:
Counts unique vehicles in each lane.
============================================================
"""

from collections import defaultdict


class VehicleCounter:

    def __init__(self):

        # vehicle IDs already counted
        self.counted_ids = set()

        # statistics
        self.lane_counts = defaultdict(

            lambda: {

                "car": 0,
                "bus": 0,
                "van": 0,
                "others": 0,
                "total": 0

            }

        )

        print("=" * 60)
        print("Vehicle Counter Initialized")
        print("=" * 60)

    # -----------------------------------------------------

    def update(self, tracked_objects):

        """
        Count vehicles only once.
        """

        for obj in tracked_objects:

            lane = obj["lane"]

            if lane is None:
                continue

            vehicle_id = obj["track_id"]

            if vehicle_id in self.counted_ids:
                continue

            self.counted_ids.add(vehicle_id)

            vehicle_class = obj["class_name"]

            self.lane_counts[lane][vehicle_class] += 1

            self.lane_counts[lane]["total"] += 1

    # -----------------------------------------------------

    def get_counts(self):

        return self.lane_counts

    # -----------------------------------------------------

    def print_summary(self):

        print("=" * 60)

        print("Vehicle Count Summary")

        print("=" * 60)

        for lane, stats in self.lane_counts.items():

            print()

            print(lane)

            print("-" * 20)

            print(f"Cars   : {stats['car']}")

            print(f"Bus    : {stats['bus']}")

            print(f"Van    : {stats['van']}")

            print(f"Others : {stats['others']}")

            print(f"Total  : {stats['total']}")