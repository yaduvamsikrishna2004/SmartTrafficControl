"""
============================================================
Traffic Density Calculator

Author : Vamsi Krishna

Description:
Calculates LIVE vehicle density in each lane.
============================================================
"""

from collections import defaultdict


class DensityCalculator:

    def __init__(self):

        print("=" * 60)
        print("Traffic Density Calculator Initialized")
        print("=" * 60)

    # --------------------------------------------------------

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

            vehicle_class = obj["class_name"]

            lane_density[lane] += 1

            class_density[lane][vehicle_class] += 1

            class_density[lane]["total"] += 1

        return lane_density, class_density

    # --------------------------------------------------------

    def get_density_level(self, total):

        if total <= 5:

            return "LOW"

        elif total <= 10:

            return "MEDIUM"

        elif total <= 20:

            return "HIGH"

        else:

            return "VERY HIGH"

    # --------------------------------------------------------

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

            print("-"*25)

            print(f"Cars    : {stats['car']}")

            print(f"Bus     : {stats['bus']}")

            print(f"Van     : {stats['van']}")

            print(f"Others  : {stats['others']}")

            print(f"Total   : {stats['total']}")

            print(f"Density : {level}")