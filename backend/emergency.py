"""
============================================================
Emergency Vehicle Detector

Author : Vamsi Krishna

Description:
Detects all emergency vehicles from tracked objects.
Now uses model.predict() instead of model.track() to avoid
tracker ID conflicts with the vehicle model.

Supported Vehicles:
    • Ambulance
    • Fire Truck
    • Police Vehicle
============================================================
"""


from collections import defaultdict


class EmergencyDetector:

    def __init__(self, detector=None, conf=0.20, iou=0.45):

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

        # =====================================================
        # Temporal confirmation state
        # =====================================================
        # Tracks consecutive frames each emergency bbox has been seen
        # Key: (frame_bbox_key) -> consecutive_count
        self._temporal_counts = defaultdict(int)
        # Frames remaining in locked state for each bbox key
        self._lock_remaining = defaultdict(int)
        # Whether a bbox key is currently locked as emergency
        self._locked_emergencies = {}

        print("=" * 60)
        print("Emergency Detector Initialized")
        print("=" * 60)

    # =====================================================

    def detect(self, frame, lane_manager, raw_results=None):

        """
        Detect emergency vehicles in the current frame.
        Uses model.predict() NOT model.track() to prevent tracker ID conflicts.

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
            # Use predict() instead of track() to avoid tracker ID conflicts
            # with the vehicle model's ByteTrack instance
            if raw_results is None:
                results = self.detector.model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False
                )
            else:
                results = raw_results

            result = results[0]

            if not self._printed_names:
                print("[Emergency] Emergency class names:", self.detector.class_names)
                self._printed_names = True

            if result.boxes is None:
                self._update_counts(emergencies)
                self.latest_emergencies = emergencies
                return emergencies

            # ----- Collect raw detections from emergency model -----
            raw_detections = []
            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width // 2
                center_y = y1 + height // 2
                center = (center_x, center_y)
                lane = lane_manager.get_lane(center)

                if lane is None:
                    lane = "Unknown"

                raw_class_name = self.detector.class_names[cls]
                class_name = self._normalize_class_name(raw_class_name)
                if class_name is None:
                    continue

                # Create a bbox hash for temporal tracking
                # Use quantized bbox to allow small movements
                bbox_key = f"{x1//10}_{y1//10}_{x2//10}_{y2//10}"

                raw_detections.append({
                    "vehicle": class_name,
                    "class_id": cls,
                    "class_name": class_name,
                    "lane": lane,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": center,
                    "source_model": "emergency_model",
                    "_bbox_key": bbox_key,
                })

            # ----- Apply temporal confirmation -----
            # Decrease all lock counters
            to_delete = []
            for key in self._lock_remaining:
                self._lock_remaining[key] -= 1
                if self._lock_remaining[key] <= 0:
                    to_delete.append(key)
            for key in to_delete:
                del self._lock_remaining[key]
                if key in self._locked_emergencies:
                    del self._locked_emergencies[key]

            # Process this frame's detections
            current_keys = set()
            for det in raw_detections:
                key = det["_bbox_key"]
                current_keys.add(key)

                if key in self._locked_emergencies:
                    # Already locked as emergency - keep it
                    det["_temporal_locked"] = True
                    continue

                # Increment consecutive count
                self._temporal_counts[key] += 1
                if self._temporal_counts[key] >= 2:
                    # Lock this emergency for 15 frames
                    self._lock_remaining[key] = 15
                    self._locked_emergencies[key] = det
                    det["_temporal_locked"] = True
                    print(f"[Emergency] TEMPORAL LOCK: {det['vehicle']} at {key} locked for 15 frames")
                else:
                    det["_temporal_locked"] = False

            # Clear temporal counts for keys not seen this frame
            stale_keys = [k for k in self._temporal_counts if k not in current_keys]
            for k in stale_keys:
                del self._temporal_counts[k]

            # Add locked emergencies that weren't detected this frame
            for key, locked_det in self._locked_emergencies.items():
                if key not in current_keys:
                    # Re-add the locked detection (with slightly reduced confidence as a signal)
                    re_add = dict(locked_det)
                    re_add["_temporal_locked"] = True
                    re_add["_temporal_recovered"] = True
                    raw_detections.append(re_add)
                    print(f"[Emergency] TEMPORAL RECOVER: {re_add['vehicle']} at {key} (re-added from lock)")

            # ----- Assign surrogate track_ids based on locked state -----
            # We use a hash of the bbox as a surrogate track_id since we're not
            # using the emergency model's tracker
            for idx, det in enumerate(raw_detections):
                # Use a deterministic hash based on bbox for tracking
                det["track_id"] = abs(hash(det["_bbox_key"])) % (10 ** 8)
                # Ensure we don't conflict with vehicle tracker IDs
                det["track_id"] = 900000 + (det["track_id"] % 100000)

            emergencies = raw_detections

        except Exception as exc:
            print(f"[Emergency] Detection failed: {exc}")
            import traceback
            traceback.print_exc()
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

        self.current_count = len([e for e in emergencies if e.get("_temporal_locked", False)])

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

    def reset_temporal(self):
        """Reset temporal confirmation state (called on engine reset)."""
        self._temporal_counts.clear()
        self._lock_remaining.clear()
        self._locked_emergencies.clear()

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
            locked = vehicle.get("_temporal_locked", False)
            print(f"Locked     : {locked}")