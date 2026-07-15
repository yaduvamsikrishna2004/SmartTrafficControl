"""
============================================================
Traffic Density Calculator

Author : Vamsi Krishna

Description:
Calculates LIVE vehicle density in each lane.
Emergency vehicles receive higher weight for density scoring.

Weights:
  Car:           1.0
  Bus:           2.0
  Van:           2.0
  Others:        1.0
  Ambulance:     5.0
  Fire Truck:    5.0
  Police Vehicle: 4.0
============================================================
"""

from collections import defaultdict
from backend.utils import map_vehicle_class, is_emergency_class


class DensityCalculator:

    def __init__(self):

        # Emergency vehicle density weights
        self.emergency_weights = {
            "ambulance": 5.0,
            "fire_truck": 5.0,
            "police": 4.0,
        }

        # General vehicle density weights
        self.vehicle_weights = {
            "car": 1.0,
            "bus": 2.0,
            "van": 2.0,
            "others": 1.0,
        }

        print("=" * 60)
        print("Traffic Density Calculator Initialized")
        print("=" * 60)

        # Store latest density for dashboard API
        self.last_density = {}

    # =========================================================

    def calculate_density(self, tracked_objects):

        lane_density = defaultdict(int)

        # Dynamically handle any vehicle type
        class_density = defaultdict(
            lambda: defaultdict(int)
        )

        # Weighted density for signal control decisions
        weighted_density = defaultdict(float)

        for obj in tracked_objects:

            lane = obj["lane"]

            if lane is None:
                continue

            # Normalize class name
            raw_class = obj.get("class_name", "others")
            vehicle_class = map_vehicle_class(raw_class)

            # Apply weight based on vehicle type
            if is_emergency_class(vehicle_class):
                weight = self.emergency_weights.get(vehicle_class, 5.0)
            else:
                weight = self.vehicle_weights.get(vehicle_class, 1.0)

            lane_density[lane] += 1
            weighted_density[lane] += weight

            class_density[lane][vehicle_class] += 1
            class_density[lane]["total"] += 1
            class_density[lane]["weighted"] = round(weighted_density[lane], 1)

        # Debug output
        print(f"[Density] Lane vehicle counts: {dict(lane_density)}")
        print(f"[Density] Weighted density: {dict(weighted_density)}")

        # Save latest density
        self.last_density = {
            "lane_density": dict(lane_density),
            "class_density": dict(class_density),
            "weighted_density": dict(weighted_density),
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

            level = self.get_density_level(stats.get("total", 0))

            print()
            print(lane)
            print("-" * 25)

            for vtype, count in sorted(stats.items()):
                if vtype not in ("total", "weighted") and count > 0:
                    print(f"{vtype.capitalize():6s}: {count}")

            print(f"Total    : {stats.get('total', 0)}")
            print(f"Weighted : {stats.get('weighted', 0)}")
            print(f"Density  : {level}")