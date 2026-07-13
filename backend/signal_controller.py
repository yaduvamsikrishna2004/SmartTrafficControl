"""
============================================================
Adaptive Signal Controller

Author : Vamsi Krishna

Description:
Calculates adaptive traffic signal timing.
Supports Emergency Vehicle Priority Override.
============================================================
"""


class SignalController:

    def __init__(
        self,
        min_green=15,
        max_green=60,
        factor=2,
        emergency_green=60
    ):

        # Latest signal plan
        self.last_signal = {}

        self.min_green = min_green
        self.max_green = max_green
        self.factor = factor

        # Emergency override duration
        self.emergency_green = emergency_green

        # Vehicle weights
        self.weights = {

            "car": 1.0,
            "van": 1.3,
            "bus": 2.5,
            "others": 1.0

        }

        print("=" * 60)
        print("Adaptive Signal Controller Initialized")
        print("=" * 60)

    # =====================================================
    # Calculate Weighted Density Score
    # =====================================================

    def calculate_score(self, lane_stats):

        score = 0

        score += lane_stats.get("car", 0) * self.weights.get("car", 1.0)
        score += lane_stats.get("van", 0) * self.weights.get("van", 1.3)
        score += lane_stats.get("bus", 0) * self.weights.get("bus", 2.5)
        score += lane_stats.get("others", 0) * self.weights.get("others", 1.0)
        score += lane_stats.get("ambulance", 0) * 5.0
        score += lane_stats.get("fire_truck", 0) * 5.0
        score += lane_stats.get("police", 0) * 5.0

        return round(score, 2)

    # =====================================================
    # Calculate Green Time
    # =====================================================

    def calculate_green_time(self, score):

        green = self.min_green + score * self.factor

        green = min(green, self.max_green)

        return round(green)

    # =====================================================
    # Generate Signal Plan
    # =====================================================

    def generate_signal_plan(
        self,
        class_density,
        emergency=None
    ):

        plan = {}

        # =================================================
        # Emergency Override
        # =================================================

        if emergency and emergency.get("active", False):

            lane = emergency["lane"]

            plan[lane] = {

                "mode": "EMERGENCY",

                "vehicle": emergency["vehicle"],

                "score": 999,

                "green_time": self.emergency_green,

                "reason": "Emergency Vehicle Detected",

                "priority": "HIGH"

            }

            self.last_signal = plan

            return plan

        # =================================================
        # Normal Adaptive AI Logic
        # =================================================

        for lane, stats in class_density.items():

            score = self.calculate_score(stats)

            green = self.calculate_green_time(score)

            plan[lane] = {

                "mode": "NORMAL",

                "score": score,

                "green_time": green,

                "reason": "Adaptive AI Density Control",

                "priority": "NORMAL"

            }

        self.last_signal = plan

        return plan

    # =====================================================
    # Print Signal Plan
    # =====================================================

    def print_signal_plan(self, plan):

        print("=" * 70)
        print("SIGNAL PLAN")
        print("=" * 70)

        for lane, info in plan.items():

            print()

            print(lane)

            print("-" * 30)

            print(f"Mode        : {info['mode']}")

            if info["mode"] == "EMERGENCY":

                print(f"Vehicle     : {info['vehicle']}")

            print(f"Score       : {info['score']}")

            print(f"Green Time  : {info['green_time']} sec")

            print(f"Priority    : {info['priority']}")

            print(f"Reason      : {info['reason']}")