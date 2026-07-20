"""
============================================================
Vehicle Counter

Author : Vamsi Krishna

Description:
Counts unique vehicles in each lane.
Prevents double counting using:
1. Track ID set (immediate dedup)
2. IoU-based matching (if track ID changes, check if same vehicle)
3. Timeout-based removal (for vehicles that leave ROI)
============================================================
"""

from collections import defaultdict
from backend.utils import map_vehicle_class
from backend.config import COUNTED_ID_TIMEOUT_FRAMES, MERGE_IOU_THRESHOLD


class VehicleCounter:

    def __init__(self):

        # vehicle IDs already counted
        self.counted_ids = set()

        # Track ID -> frame number when counted (for timeout-based removal)
        self.counted_id_frames = {}

        # Track ID -> last known bbox (for IoU matching when ID changes)
        self.counted_id_bboxes = {}

        # statistics - dynamically handle any vehicle type
        self.lane_counts = defaultdict(
            lambda: defaultdict(int)
        )

        # Global frame counter (incremented externally)
        self._frame_number = 0

        print("=" * 60)
        print("Vehicle Counter Initialized (Double-Count Protected)")
        print("=" * 60)

    # -----------------------------------------------------

    def update(self, tracked_objects, frame_number=None):
        """
        Count vehicles only once.
        
        Fixes for double counting:
        1. Track ID set: never count same ID twice
        2. IoU matching: if a "new" ID overlaps significantly with a 
           previously counted vehicle, it's the same vehicle - skip it
        3. Timeout: only remove IDs from the counted set after timeout
        
        Parameters
        ----------
        tracked_objects : list
            List of tracked vehicle objects with track_id, bbox, lane, class_name
        frame_number : int, optional
            Current frame number for timeout tracking
        """
        if frame_number is not None:
            self._frame_number = frame_number
        else:
            self._frame_number += 1

        for obj in tracked_objects:

            lane = obj.get("lane")

            if lane is None:
                continue

            vehicle_id = obj.get("track_id")
            if vehicle_id is None:
                continue

            # ====================================================
            # FIX 1: Direct ID check - if already counted, skip
            # ====================================================
            if vehicle_id in self.counted_ids:
                # Update the bbox for this ID (helps with IoU matching)
                bbox = obj.get("bbox")
                if bbox:
                    self.counted_id_bboxes[vehicle_id] = bbox
                continue

            # ====================================================
            # FIX 2: IoU-based matching - check if this "new" ID
            # actually overlaps with a previously counted vehicle
            # that may have gotten a new track ID from ByteTrack
            # ====================================================
            bbox = obj.get("bbox")
            if bbox:
                is_duplicate = self._check_iou_duplicate(vehicle_id, bbox)
                if is_duplicate:
                    # This is the same vehicle with a new ID - skip counting
                    # But add the new ID to the counted set so we don't check again
                    self.counted_ids.add(vehicle_id)
                    self.counted_id_frames[vehicle_id] = self._frame_number
                    self.counted_id_bboxes[vehicle_id] = bbox
                    continue

            # ====================================================
            # FIX 3: Count this vehicle (first time seeing it)
            # ====================================================
            self.counted_ids.add(vehicle_id)
            self.counted_id_frames[vehicle_id] = self._frame_number
            if bbox:
                self.counted_id_bboxes[vehicle_id] = bbox

            # Normalize class names to canonical keys
            raw_class = obj.get("class_name", "others")
            vehicle_class = map_vehicle_class(raw_class)

            # Ensure key exists
            if vehicle_class not in self.lane_counts[lane]:
                self.lane_counts[lane][vehicle_class] = 0

            self.lane_counts[lane][vehicle_class] += 1
            self.lane_counts[lane]["total"] += 1

        # ====================================================
        # FIX 4: Periodic cleanup of old counted IDs
        # Only remove IDs that haven't been seen for a long time
        # This prevents unbounded memory growth while still
        # preventing double counting of active vehicles
        # ====================================================
        if self._frame_number % 100 == 0:  # Cleanup every 100 frames
            self._cleanup_old_ids()

    # -----------------------------------------------------

    def _check_iou_duplicate(self, new_id, new_bbox):
        """
        Check if a vehicle with a new track ID is actually the same
        as a previously counted vehicle by comparing IoU.
        
        This handles the case where ByteTrack assigns a new ID to a
        vehicle that was temporarily occluded or lost.
        """
        if not new_bbox:
            return False

        for counted_id, counted_bbox in list(self.counted_id_bboxes.items()):
            if counted_id == new_id:
                continue
            if counted_bbox is None:
                continue

            iou = self._compute_iou(new_bbox, counted_bbox)
            if iou > MERGE_IOU_THRESHOLD:
                # Same vehicle - don't count again
                return True

        return False

    # -----------------------------------------------------

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

    # -----------------------------------------------------

    def _cleanup_old_ids(self):
        """
        Remove IDs that haven't been seen for COUNTED_ID_TIMEOUT_FRAMES.
        This prevents memory leaks while keeping active vehicles protected.
        """
        threshold = self._frame_number - COUNTED_ID_TIMEOUT_FRAMES
        stale_ids = [
            vid for vid, frame in self.counted_id_frames.items()
            if frame < threshold
        ]
        for vid in stale_ids:
            self.counted_ids.discard(vid)
            self.counted_id_frames.pop(vid, None)
            self.counted_id_bboxes.pop(vid, None)

        if stale_ids:
            print(f"[Counter] Cleaned up {len(stale_ids)} stale tracked IDs")

    # -----------------------------------------------------

    def get_counts(self):
        return self.lane_counts

    # -----------------------------------------------------

    def get_total(self):
        """Return total count across all lanes."""
        total = 0
        for lane, stats in self.lane_counts.items():
            total += stats.get("total", 0)
        return total

    def print_summary(self):
        print("=" * 60)
        print("Vehicle Count Summary")
        print("=" * 60)
        for lane, stats in self.lane_counts.items():
            print()
            print(lane)
            print("-" * 20)
            for vtype, count in sorted(stats.items()):
                if vtype != "total" and count > 0:
                    print(f"{vtype.capitalize():6s}: {count}")
            print(f"Total  : {stats.get('total', 0)}")