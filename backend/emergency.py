"""
============================================================
Emergency Vehicle Detector

Author : Vamsi Krishna

Description:
Detects all emergency vehicles from tracked objects.

Supported Vehicles:
    • Ambulance
    • Fire Truck
    • Police Vehicle
============================================================
"""


class EmergencyDetector:

    def __init__(self):

        self.emergency_classes = {

            "ambulance",

            "fire_truck",

            "police"

        }

        self.latest_emergencies = []

        print("=" * 60)
        print("Emergency Detector Initialized")
        print("=" * 60)

    # =====================================================

    def detect(self, tracked_objects):

        """
        Detect all emergency vehicles.

        Parameters
        ----------
        tracked_objects : list

        Returns
        -------
        list

        Example

        [

            {

                "vehicle":"ambulance",

                "lane":"Lane_B",

                "confidence":0.97,

                "track_id":15

            }

        ]
        """

        emergencies = []

        for obj in tracked_objects:

            if obj["class_name"] in self.emergency_classes:

                emergencies.append({

                    "vehicle": obj["class_name"],

                    "lane": obj["lane"],

                    "confidence": obj["confidence"],

                    "track_id": obj["track_id"]

                })

        self.latest_emergencies = emergencies

        return emergencies

    # =====================================================

    def has_emergency(self):

        return len(self.latest_emergencies) > 0

    # =====================================================

    def get_latest(self):

        return self.latest_emergencies

    # =====================================================

    def print_emergencies(self):

        print("=" * 70)

        print("EMERGENCY VEHICLES")

        print("=" * 70)

        if len(self.latest_emergencies) == 0:

            print("No Emergency Vehicles Detected")

            return

        for vehicle in self.latest_emergencies:

            print()

            print(f"Vehicle    : {vehicle['vehicle']}")

            print(f"Lane       : {vehicle['lane']}")

            print(f"Track ID   : {vehicle['track_id']}")

            print(f"Confidence : {vehicle['confidence']}")