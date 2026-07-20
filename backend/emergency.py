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

FIXES:
1. Lower confidence threshold (0.20) to never miss emergency vehicles
2. Temporal confirmation: 2 consecutive frames → lock for 15 frames
3. Object missing 1 frame → keep tracking (temporal recovery)
4. Object missing 10 frames → remove
5. Emergency overlap IoU threshold lowered to 0.3 for partial occlusions
6. Better bbox key generation for temporal tracking
============================================================
"""


from collections import defaultdict
from backend.config import (
    TEMPORAL_CONFIRM_FRAMES,
    TEMPORAL_LOCK_FRAMES,
    EMERGENCY_OVERLAP_IOU,
    MERGE_IOU_THRESHOLD,
)


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

        # =====================================================
        # FIX: Track missing frames for each locked emergency
        # If missing for more than VEHICLE_MISSING_TIMEOUT frames,
        # remove the lock
        # =====================================================
        self._missing_frames = defaultdict(int)
        self._max_missing_frames = 10  # Remove after 10 consecutive missing frames

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
                # ====================================================
                # FIX: Even if no detections this frame, check if we
                # have locked emergencies that should be recovered
                # ====================================================
                recovered = self._recover_locked_emergencies(lane_manager)
                if recovered:
                    emergencies = recovered
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

                # ====================================================
                # FIX: Better bbox key - use smaller quantization
                # to handle small movements between frames
                # ====================================================
                bbox_key = f"{x1//5}_{y1//5}_{x2//5}_{y2//5}"

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

            # ====================================================
            # FIX: Apply temporal confirmation with missing frame tracking
            # ====================================================
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
                if key in self._missing_frames:
                    del self._missing_frames[key]

            # Process this frame's detections
            current_keys = set()
            for det in raw_detections:
                key = det["_bbox_key"]
                current_keys.add(key)

                # Reset missing frame counter for this key
                if key in self._missing_frames:
                    del self._missing_frames[key]

                if key in self._locked_emergencies:
                    # Already locked as emergency - keep it
                    det["_temporal_locked"] = True
                    # Update the bbox in the locked entry
                    self._locked_emergencies[key]["bbox"] = det["bbox"]
                    self._locked_emergencies[key]["center"] = det["center"]
                    self._locked_emergencies[key]["confidence"] = det["confidence"]
                    continue

                # Increment consecutive count
                self._temporal_counts[key] += 1
                if self._temporal_counts[key] >= TEMPORAL_CONFIRM_FRAMES:
                    # Lock this emergency for configured frames
                    self._lock_remaining[key] = TEMPORAL_LOCK_FRAMES
                    self._locked_emergencies[key] = det
                    det["_temporal_locked"] = True
                    print(f"[Emergency] TEMPORAL LOCK: {det['vehicle']} at {key} locked for {TEMPORAL_LOCK_FRAMES} frames")
                else:
                    det["_temporal_locked"] = False

            # Clear temporal counts for keys not seen this frame
            stale_keys = [k for k in self._temporal_counts if k not in current_keys]
            for k in stale_keys:
                del self._temporal_counts[k]

            # ====================================================
            # FIX: Track missing frames for locked emergencies
            # ====================================================
            for key in list(self._locked_emergencies.keys()):
                if key not in current_keys:
                    self._missing_frames[key] += 1
                    if self._missing_frames[key] > self._max_missing_frames:
                        # Remove this lock - vehicle has left the scene
                        print(f"[Emergency] REMOVING LOCK: {self._locked_emergencies[key]['vehicle']} at {key} "
                              f"(missing {self._missing_frames[key]} frames)")
                        del self._locked_emergencies[key]
                        if key in self._lock_remaining:
                            del self._lock_remaining[key]
                        if key in self._missing_frames:
                            del self._missing_frames[key]

            # Add locked emergencies that weren't detected this frame
            for key, locked_det in self._locked_emergencies.items():
                if key not in current_keys:
                    # Re-add the locked detection (with slightly reduced confidence as a signal)
                    re_add = dict(locked_det)
                    re_add["_temporal_locked"] = True
                    re_add["_temporal_recovered"] = True
                    raw_detections.append(re_add)
                    print(f"[Emergency] TEMPORAL RECOVER: {re_add['vehicle']} at {key} (re-added from lock, "
                          f"missing {self._missing_frames.get(key, 0)} frames)")

            # ====================================================
            # FIX: Use IoU-based matching to detect if an emergency
            # vehicle was detected but with a slightly different bbox
            # (handles partial occlusions and small movements)
            # ====================================================
            # Check for IoU-based matches between new detections and locked emergencies
            for det in raw_detections:
                if det.get("_temporal_locked"):
                    continue
                det_bbox = det.get("bbox")
                if not det_bbox:
                    continue
                for key, locked_det in list(self._locked_emergencies.items()):
                    locked_bbox = locked_det.get("bbox")
                    if not locked_bbox:
                        continue
                    iou = self._compute_iou(det_bbox, locked_bbox)
                    if iou > EMERGENCY_OVERLAP_IOU:
                        # This detection matches a locked emergency - use the lock
                        det["_temporal_locked"] = True
                        det["_bbox_key"] = key  # Use the same key for consistency
                        # Update the locked entry with new bbox
                        self._locked_emergencies[key]["bbox"] = det["bbox"]
                        self._locked_emergencies[key]["center"] = det["center"]
                        self._locked_emergencies[key]["confidence"] = max(
                            self._locked_emergencies[key]["confidence"], det["confidence"]
                        )
                        # Reset missing frames
                        if key in self._missing_frames:
                            del self._missing_frames[key]
                        break

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

    def _recover_locked_emergencies(self, lane_manager):
        """
        When no detections are found, check if we have locked emergencies
        that should still be reported.
        """
        recovered = []
        for key, locked_det in list(self._locked_emergencies.items()):
            self._missing_frames[key] += 1
            if self._missing_frames[key] > self._max_missing_frames:
                print(f"[Emergency] REMOVING LOCK (no detections): {locked_det['vehicle']} at {key}")
                del self._locked_emergencies[key]
                if key in self._lock_remaining:
                    del self._lock_remaining[key]
                if key in self._missing_frames:
                    del self._missing_frames[key]
                continue

            re_add = dict(locked_det)
            re_add["_temporal_locked"] = True
            re_add["_temporal_recovered"] = True
            recovered.append(re_add)

        if recovered:
            print(f"[Emergency] Recovered {len(recovered)} locked emergencies (no new detections)")

        return recovered

    # =====================================================

    def _compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bounding boxes."""
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection coordinates
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)

        # Areas of both boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        if area1 + area2 - inter_area <= 0:
            return 0.0

        return inter_area / (area1 + area2 - inter_area)

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
        self._missing_frames.clear()

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