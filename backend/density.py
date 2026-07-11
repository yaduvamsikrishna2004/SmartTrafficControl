"""
============================================================
Traffic Density Calculator

Author : Vamsi Krishna

Description:
Calculates LIVE vehicle density in each lane.
============================================================
"""

from collections import defaultdict
from backend.utils import map_vehicle_class


class DensityCalculator:

    def __init__(self):

        print("=" * 60)
        print("Traffic Density Calculator Initialized")
        print("=" * 60)

        # Store latest density for dashboard API
        self.last_density = {}

    # =========================================================

    def calculate_density(self, tracked_objects):

        lane_density = defaultdict(int)

        class_density = defaultdict(

            lambda: {

                "car": 0,
                "bus": 0,
                "van": 0,
                "others": 0,
                "total": 0

            }

        )

        for obj in tracked_objects:

            lane = obj["lane"]

            if lane is None:
                continue

            # normalize class name to ensure keys exist
            raw_class = obj.get("class_name", "others")
            vehicle_class = map_vehicle_class(raw_class)

            lane_density[lane] += 1

            # ensure key exists
            if vehicle_class not in class_density[lane]:
                class_density[lane][vehicle_class] = 0

            class_density[lane][vehicle_class] += 1

            class_density[lane]["total"] += 1

        # Debug: save a quick printout
        print(f"[Density] lane_density={dict(lane_density)}")

        # Save latest density
        self.last_density = {

            "lane_density": dict(lane_density),

            "class_density": dict(class_density)

        }

        return lane_density, class_density

    # =========================================================

    def get_density_level(self, total):

        if total <= 5:

            return "LOW"

        elif total <= 10:

            return "MEDIUM"

        elif total <= 20:

            return "HIGH"

        else:

            return "VERY HIGH"

    # =========================================================

    def print_density(self, class_density):

        print("=" * 70)
        print("LIVE TRAFFIC DENSITY")
        print("=" * 70)

        if len(class_density) == 0:

            print("No vehicles detected.")
            return

        for lane, stats in class_density.items():

            level = self.get_density_level(stats["total"])

            print()

            print(lane)
            print("-" * 25)

            print(f"Cars    : {stats['car']}")
            print(f"Bus     : {stats['bus']}")
            print(f"Van     : {stats['van']}")
            print(f"Others  : {stats['others']}")
            print(f"Total   : {stats['total']}")
            print(f"Density : {level}")