"""
============================================================
Adaptive Signal Controller

Author : Vamsi Krishna

Description:
Calculates adaptive traffic signal timing.
Supports Emergency Vehicle Priority Override.

Emergency Priority:
  Highest Priority: Ambulance (100)
  ↓
  Fire Truck (90)
  ↓
  Police Vehicle (80)
  ↓
  Normal Adaptive Signal

When emergency vehicle detected:
  - Immediately override adaptive timing
  - Green signal on emergency lane
  - All other lanes: Red
When emergency vehicle exits:
  - Return to adaptive mode
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

        # Emergency override duration (seconds)
        self.emergency_green = emergency_green

        # Vehicle weights for density scoring
        self.weights = {
            "car": 1.0,
            "van": 1.3,
            "bus": 2.5,
            "others": 1.0,
            "TwoWheelers": 0.5,
            "auto-rikshaw": 1.0,
        }

        # Emergency vehicle weights (high priority = high weight)
        self.emergency_weights = {
            "ambulance": 10.0,
            "fire_truck": 10.0,
            "police": 8.0,
        }

        print("=" * 60)
        print("Adaptive Signal Controller Initialized")
        print("=" * 60)

    # =====================================================
    # Calculate Weighted Density Score
    # =====================================================

    def calculate_score(self, lane_stats):

        score = 0

        # General vehicle weights
        for vtype, weight in self.weights.items():
            score += lane_stats.get(vtype, 0) * weight

        # Emergency vehicle weights (much higher)
        for etype, weight in self.emergency_weights.items():
            score += lane_stats.get(etype, 0) * weight

        # Also check for alternate emergency class names
        score += lane_stats.get("ambulance", 0) * 10.0
        score += lane_stats.get("fire_truck", 0) * 10.0
        score += lane_stats.get("police", 0) * 8.0

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
        # When emergency vehicle detected:
        #   - Green signal on emergency lane
        #   - All other lanes: Red (not included in plan)
        # =================================================

        if emergency and emergency.get("active", False):

            lane = emergency["lane"]
            vehicle = emergency.get("vehicle", "Unknown")
            priority = emergency.get("priority", 80)

            plan[lane] = {
                "mode": "EMERGENCY",
                "vehicle": vehicle,
                "score": 999,
                "green_time": self.emergency_green,
                "reason": f"Emergency Vehicle Detected: {vehicle}",
                "priority": "HIGH",
                "emergency_priority": priority,
                "all_other_lanes": "RED",
            }

            self.last_signal = plan
            print(f"[Signal] EMERGENCY OVERRIDE: Lane {lane} GREEN ({self.emergency_green}s) for {vehicle}")
            print(f"[Signal] All other lanes: RED")

            return plan

        # =================================================
        # Normal Adaptive AI Logic
        # =================================================

        for lane, stats in class_density.items():

            score = self.calculate_score(stats)
            green = self.calculate_green_time(score)

            # Determine priority level
            if score >= 50:
                priority = "HIGH"
            elif score >= 25:
                priority = "MEDIUM"
            else:
                priority = "NORMAL"

            plan[lane] = {
                "mode": "NORMAL",
                "score": score,
                "green_time": green,
                "reason": "Adaptive AI Density Control",
                "priority": priority,
            }

        self.last_signal = plan
        print(f"[Signal] Normal signal plan generated for {len(plan)} lanes")

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