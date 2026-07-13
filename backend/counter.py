"""
============================================================
Vehicle Counter

Author : Vamsi Krishna

Description:
Counts unique vehicles in each lane.
============================================================
"""

from collections import defaultdict
from backend.utils import map_vehicle_class


class VehicleCounter:

    def __init__(self):

        # vehicle IDs already counted
        self.counted_ids = set()

        # statistics - dynamically handle any vehicle type
        self.lane_counts = defaultdict(
            lambda: defaultdict(int)
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

            # normalize class names to canonical keys
            raw_class = obj.get("class_name", "others")
            vehicle_class = map_vehicle_class(raw_class)

            # ensure key exists
            if vehicle_class not in self.lane_counts[lane]:
                self.lane_counts[lane][vehicle_class] = 0

            self.lane_counts[lane][vehicle_class] += 1

            self.lane_counts[lane]["total"] += 1

        # Debug: print a brief update
        print(f"[Counter] Total lanes counted: {len(self.lane_counts)}")

    # -----------------------------------------------------

    def get_counts(self):

        return self.lane_counts

    # -----------------------------------------------------

    def get_total(self):
        """Return total count across all lanes."""
        total = 0
        for lane, stats in self.lane_counts.items():
            total += stats.get("total", 0)
        return total

    def print_summary(self):

        print("=" * 60)

        print("Vehicle Count Summary")

        print("=" * 60)

        for lane, stats in self.lane_counts.items():

            print()

            print(lane)

            print("-" * 20)

            for vtype, count in sorted(stats.items()):
                if vtype != "total" and count > 0:
                    print(f"{vtype.capitalize():6s}: {count}")

            print(f"Total  : {stats.get('total', 0)}")
