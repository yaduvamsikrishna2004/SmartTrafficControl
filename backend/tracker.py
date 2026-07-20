"""
============================================================
AI Smart Traffic Signal Control System
Tracker Module

Author : Vamsi Krishna

Description:
    Performs multi-object tracking using
    YOLOv8 + ByteTrack.
============================================================
"""

import cv2
from backend.detector import YOLODetector


class VehicleTracker:

    def __init__(self, model_path):

        self.detector = YOLODetector(
            model_path,
            model_label="Vehicle model loaded"
        )

        print("Vehicle Tracker Initialized")

    # -----------------------------------------------------

    def track_frame(
        self,
        frame,
        conf=0.30,
        iou=0.45,
        return_results=False
    ):

        """
        Track vehicles in a single frame.

        If return_results is True, returns a tuple of
        (tracked_objects, annotated_frame, raw_results).
        """

        results = self.detector.model.track(

            source=frame,

            conf=conf,

            iou=iou,

            persist=True,

            tracker="bytetrack.yaml",

            verbose=False

        )

        result = results[0]

        tracked_objects = []

        if result.boxes is None:

            if return_results:
                return tracked_objects, frame, result

            return tracked_objects, frame

        annotated = result.plot()

        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1

            height = y2 - y1

            center_x = x1 + width // 2

            center_y = y1 + height // 2

            tracked_objects.append({

                "track_id": track_id,

                "class_id": cls,

                "class_name": self.detector.class_names[cls],

                "confidence": round(conf, 3),

                "bbox": [x1, y1, x2, y2],

                "center": (center_x, center_y),

                "width": width,

                "height": height

            })

        print(f"[Tracker] Returning {len(tracked_objects)} tracked objects")

        if return_results:
            return tracked_objects, annotated, result

        return tracked_objects, annotated

    # -----------------------------------------------------

    def print_tracks(self, tracks):

        print("=" * 80)

        print("Tracked Vehicles")

        print("=" * 80)

        if len(tracks) == 0:

            print("No vehicles tracked.")

            return

        for obj in tracks:

            print(

                f"ID: {obj['track_id']:3d} | "

                f"{obj['class_name']:8s} | "

                f"Conf: {obj['confidence']:.2f} | "

                f"Center: {obj['center']}"

            )
