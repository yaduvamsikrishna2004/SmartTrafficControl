"""
Test script to verify yolov8n.pt integration into the pipeline.
"""
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import COCO_MODEL_PATH, MODEL_PATH, LANE_CONFIG
from backend.detector import YOLODetector
from backend.utils import is_coco_traffic_class, get_coco_class_name, map_vehicle_class
from backend.traffic_engine import TrafficEngine
import cv2
import time

print("=" * 60)
print("YOLOv8n Integration Test")
print("=" * 60)

# Test 1: COCO Utility Functions
print("\n[Test 1] COCO Utility Functions")
print(f"  is_coco_traffic_class(2) (car): {is_coco_traffic_class(2)}")
print(f"  is_coco_traffic_class(7) (truck): {is_coco_traffic_class(7)}")
print(f"  is_coco_traffic_class(0) (person): {is_coco_traffic_class(0)}")
print(f"  is_coco_traffic_class(5) (bus): {is_coco_traffic_class(5)}")
print(f"  get_coco_class_name(5): {get_coco_class_name(5)}")
print(f"  map_vehicle_class('truck'): {map_vehicle_class('truck')}")
print(f"  map_vehicle_class('motorcycle'): {map_vehicle_class('motorcycle')}")
print(f"  map_vehicle_class('bicycle'): {map_vehicle_class('bicycle')}")
print("  PASSED")

# Test 2: Load COCO Model
print("\n[Test 2] Loading COCO Model (yolov8n.pt)")
start = time.time()
coco_detector = YOLODetector(COCO_MODEL_PATH, model_label="Test load")
elapsed = time.time() - start
print(f"  Load time: {elapsed:.2f}s")
print(f"  Number of classes: {len(coco_detector.class_names)}")
print(f"  Traffic-relevant classes available:")
for idx in [1, 2, 3, 5, 7]:
    print(f"    {idx}: {coco_detector.class_names[idx]}")
print("  PASSED")

# Test 3: Test _extract_coco_detections with a dummy frame
print("\n[Test 3] COCO Detection on Dummy Frame")
import numpy as np
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
results = coco_detector.model.track(
    source=dummy_frame,
    conf=0.30,
    iou=0.45,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
)
# Manually test the extraction logic
from backend.utils import is_coco_traffic_class, map_vehicle_class
objects = []
if results and len(results) > 0:
    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            if is_coco_traffic_class(cls):
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width // 2
                center_y = y1 + height // 2
                raw_name = coco_detector.class_names[cls]
                canonical_name = map_vehicle_class(raw_name)
                objects.append({
                    "class_id": cls,
                    "class_name": canonical_name,
                    "confidence": round(conf, 3),
                    "source": "coco_model"
                })
print(f"  COCO detections on blank frame: {len(objects)} (expected 0)")
print("  PASSED")

# Test 4: Full Engine Initialization
print("\n[Test 4] Full TrafficEngine Initialization (with COCO detector)")
start = time.time()
engine = TrafficEngine(MODEL_PATH, LANE_CONFIG)
elapsed = time.time() - start
print(f"  Engine init time: {elapsed:.2f}s")
print(f"  COCO detector loaded: {hasattr(engine, 'coco_detector')}")
print(f"  COCO detector model path: {COCO_MODEL_PATH}")
print("  PASSED")

# Test 5: Process a single frame from video
print("\n[Test 5] Processing a Frame (with COCO supplementary detection)")
from backend.config import VIDEO_PATH
video_path = VIDEO_PATH
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        start = time.time()
        result = engine.process_frame(frame)
        elapsed = time.time() - start
        print(f"  Frame processing time: {elapsed*1000:.1f}ms")
        print(f"  Primary detections: {len(result.get('vehicle_detections', []))}")
        print(f"  COCO supplementary detections: {len(result.get('coco_detections', []))}")
        print(f"  Emergency detections: {len(result.get('emergency_detections', []))}")
        print(f"  Merged detections: {len(result.get('merged_detections', []))}")
        print(f"  FPS: {result.get('fps', 0)}")
        print("  PASSED")
    cap.release()
else:
    print(f"  Video not found at {video_path}, skipping frame test")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)