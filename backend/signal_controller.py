"""
============================================================
Adaptive Signal Controller

Author : Vamsi Krishna

Description:
Calculates adaptive green signal duration
based on weighted traffic density.
============================================================
"""


class SignalController:

    def __init__(
        self,
        min_green=15,
        max_green=60,
        factor=2
    ):

        self.min_green = min_green
        self.max_green = max_green
        self.factor = factor

        self.weights = {

            "car": 1.0,

            "van": 1.3,

            "bus": 2.5,

            "others": 1.0

        }

        print("=" * 60)
        print("Adaptive Signal Controller Initialized")
        print("=" * 60)

    # -----------------------------------------------------

    def calculate_score(self, lane_stats):

        score = 0

        score += lane_stats["car"] * self.weights["car"]
        score += lane_stats["van"] * self.weights["van"]
        score += lane_stats["bus"] * self.weights["bus"]
        score += lane_stats["others"] * self.weights["others"]

        return round(score, 2)

    # -----------------------------------------------------

    def calculate_green_time(self, score):

        green = self.min_green + score * self.factor

        green = min(green, self.max_green)

        return round(green)

    # -----------------------------------------------------

    def generate_signal_plan(self, class_density):

        plan = {}

        for lane, stats in class_density.items():

            score = self.calculate_score(stats)

            green = self.calculate_green_time(score)

            plan[lane] = {

                "score": score,

                "green_time": green

            }

        return plan

    # -----------------------------------------------------

    def print_signal_plan(self, plan):

        print("=" * 70)
        print("ADAPTIVE SIGNAL PLAN")
        print("=" * 70)

        for lane, info in plan.items():

            print()

            print(lane)

            print("-"*25)

            print(f"Density Score : {info['score']}")

            print(f"Green Time    : {info['green_time']} sec")