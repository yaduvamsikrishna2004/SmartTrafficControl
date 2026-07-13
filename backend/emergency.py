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


from collections import defaultdict


class EmergencyDetector:

    def __init__(self, detector=None, conf=0.30, iou=0.45):

        self.detector = detector
        self.conf = conf
        self.iou = iou

        self.emergency_classes = {
            "ambulance",
            "fire_truck",
            "police"
        }

        self._printed_names = False

        self.latest_emergencies = []
        self.current_count = 0
        self.total_count = 0
        self.seen_track_ids = set()
        self.per_lane_count = defaultdict(int)
        self.per_vehicle_count = defaultdict(int)

        print("=" * 60)
        print("Emergency Detector Initialized")
        print("=" * 60)

    # =====================================================

    def detect(self, frame, lane_manager, raw_results=None):

        """
        Detect emergency vehicles in the current frame.

        Parameters
        ----------
        frame : np.ndarray
        lane_manager : LaneManager
        raw_results : YOLO results object, optional

        Returns
        -------
        list
        """

        emergencies = []

        if self.detector is None:
            self._update_counts(emergencies)
            self.latest_emergencies = emergencies
            return emergencies

        try:
            if raw_results is None:
                results = self.detector.model.track(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            else:
                results = raw_results

            result = results[0]

            if not self._printed_names:
                print("[Emergency] emergency_results.names:", self.detector.class_names)
                self._printed_names = True

            if result.boxes is None:
                self._update_counts(emergencies)
                self.latest_emergencies = emergencies
                return emergencies

            for box in result.boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width // 2
                center_y = y1 + height // 2
                center = (center_x, center_y)
                lane = lane_manager.get_lane(center)

                # If lane polygon doesn't contain center, try using bbox center approach
                if lane is None:
                    # Still record the emergency, assign a lane name based on x-position
                    # or mark it as "Unknown" - still count it for emergency detection
                    lane = "Unknown"

                raw_class_name = self.detector.class_names[cls]
                class_name = self._normalize_class_name(raw_class_name)
                if class_name is None:
                    continue

                emergencies.append({
                    "vehicle": class_name,
                    "class_id": cls,
                    "class_name": class_name,
                    "lane": lane,
                    "confidence": round(confidence, 3),
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "center": center,
                    "source_model": "emergency_model"
                })

        except Exception as exc:
            print(f"[Emergency] Detection failed: {exc}")
            emergencies = []

        self._update_counts(emergencies)
        self.latest_emergencies = emergencies
        return emergencies

    # =====================================================

    def _normalize_class_name(self, class_name):

        if not class_name:
            return None

        label = str(class_name).lower().replace("-", " ").replace("_", " ").strip()

        if label == "ambulance":
            return "ambulance"

        if label in ("firetruck", "fire truck", "fire_truck"):
            return "fire_truck"

        if label in ("police", "police vehicle", "police car", "police_car", "policevehicle"):
            return "police"

        return None

    # =====================================================

    def _update_counts(self, emergencies):

        self.current_count = len(emergencies)

        for emergency in emergencies:

            track_id = emergency["track_id"]

            if track_id in self.seen_track_ids:
                continue

            self.seen_track_ids.add(track_id)
            self.total_count += 1
            self.per_lane_count[emergency["lane"]] += 1
            self.per_vehicle_count[emergency["vehicle"]] += 1

    # =====================================================

    def get_summary(self):

        return {
            "current_count": self.current_count,
            "total_count": self.total_count,
            "per_lane_count": dict(self.per_lane_count),
            "per_vehicle_count": dict(self.per_vehicle_count)
        }

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