"""
============================================================
AI Smart Traffic Signal Control System
Detector Module

Author  : Vamsi Krishna
Model   : YOLOv8 Custom Model

Description:
    Loads the trained YOLOv8 model and performs
    object detection on images, frames and videos.
============================================================
"""

import os
import cv2
from ultralytics import YOLO


class YOLODetector:
    """
    YOLO Detector Class
    """

    def __init__(self, model_path: str):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        print("=" * 60)
        print("Loading YOLO Model...")
        print("=" * 60)

        self.model = YOLO(model_path)
        self.class_names = self.model.names

        print("✅ Model Loaded Successfully\n")

        print("Classes")
        for idx, name in self.class_names.items():
            print(f"{idx} -> {name}")

        print("=" * 60)

    # ======================================================

    def detect_frame(
        self,
        frame,
        conf=0.30,
        iou=0.45
    ):

        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False
        )

        return results[0]

    # ======================================================

    def detect_image(
        self,
        image_path,
        conf=0.30,
        save=False
    ):

        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save
        )

        return results[0]

    # ======================================================

    def detect_video(
        self,
        video_path,
        conf=0.30,
        save=True
    ):

        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        self.model.predict(
            source=video_path,
            conf=conf,
            save=save
        )

    # ======================================================

    def draw_boxes(
        self,
        frame,
        results
    ):

        return results.plot()

    # ======================================================

    def get_detections(self, results):

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            center_x = x1 + width // 2
            center_y = y1 + height // 2

            detections.append({

                "class_id": cls,

                "class_name": self.class_names[cls],

                "confidence": round(conf, 3),

                "bbox": [x1, y1, x2, y2],

                "center": (center_x, center_y),

                "width": width,

                "height": height

            })

        return detections

    # ======================================================

    def print_detections(self, detections):

        print("=" * 80)
        print("Detected Objects")
        print("=" * 80)

        if len(detections) == 0:
            print("No detections.")
            return

        for index, obj in enumerate(detections, start=1):

            print(f"Object {index}")

            print(f"Class      : {obj['class_name']}")

            print(f"Confidence : {obj['confidence']}")

            print(f"BBox       : {obj['bbox']}")

            print(f"Center     : {obj['center']}")

            print(f"Width      : {obj['width']}")

            print(f"Height     : {obj['height']}")

            print("-" * 80)

    # ======================================================

    def get_class_names(self):

        return self.class_names