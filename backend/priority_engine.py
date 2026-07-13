"""
============================================================
Emergency Priority Engine

Author : Vamsi Krishna

Description:
Determines which emergency vehicle should receive
highest signal priority.

Supports:
    • Ambulance
    • Fire Truck
    • Police Vehicle

Future Support:
    • ETA
    • Distance
    • Vehicle Speed
    • GPS
============================================================
"""


class PriorityEngine:

    def __init__(self):

        # Higher value = Higher priority
        self.priority_table = {

            "ambulance": 100,

            "fire_truck": 90,

            "police": 80

        }

        self.latest_decision = {

            "active": False

        }

        print("=" * 60)
        print("Priority Engine Initialized")
        print("=" * 60)

    # =====================================================

    def evaluate(self, emergency_objects):

        """
        Parameters
        ----------
        emergency_objects : list

        Example

        [

            {

                "vehicle":"ambulance",

                "lane":"Lane_B",

                "confidence":0.97,

                "track_id":12

            },

            {

                "vehicle":"fire_truck",

                "lane":"Lane_A",

                "confidence":0.91,

                "track_id":30

            }

        ]
        """

        # ------------------------------------------

        if len(emergency_objects) == 0:

            self.latest_decision = {

                "active": False

            }

            return self.latest_decision

        # ------------------------------------------

        best = None

        highest_priority = -1

        # ------------------------------------------

        for vehicle in emergency_objects:

            vehicle_type = vehicle["vehicle"]

            priority = self.priority_table.get(

                vehicle_type,

                0

            )

            if priority > highest_priority:

                highest_priority = priority

                best = vehicle

            elif priority == highest_priority and best is not None:

                if vehicle["confidence"] > best["confidence"]:

                    best = vehicle

        # ------------------------------------------

        decision = {

            "active": True,

            "vehicle": best["vehicle"],

            "lane": best["lane"],

            "track_id": best["track_id"],

            "confidence": best["confidence"],

            "priority": highest_priority,

            "override": True,

            "reason": "Highest Priority Emergency Vehicle"

        }

        self.latest_decision = decision

        return decision

    # =====================================================

    def get_latest_decision(self):

        return self.latest_decision

    # =====================================================

    def print_decision(self, decision):

        print("=" * 70)

        print("EMERGENCY PRIORITY ENGINE")

        print("=" * 70)

        if not decision["active"]:

            print("No emergency vehicle detected.")

            return

        print()

        print(f"Vehicle    : {decision['vehicle']}")

        print(f"Lane       : {decision['lane']}")

        print(f"Priority   : {decision['priority']}")

        print(f"Track ID   : {decision['track_id']}")

        print(f"Confidence : {decision['confidence']}")

        print(f"Reason     : {decision['reason']}")