import cv2

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.counter import VehicleCounter

MODEL_PATH = "models/best.pt"

VIDEO_PATH = "videos/cam4.mp4"

LANE_CONFIG = "config/lanes.json"

tracker = VehicleTracker(MODEL_PATH)

lane_manager = LaneManager(LANE_CONFIG)

counter = VehicleCounter()

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    tracks, annotated = tracker.track_frame(frame)

    lane_objects = lane_manager.assign_lanes(tracks)

    counter.update(lane_objects)

    if frame_count % 100 == 0:

        print()

        print(f"Frame : {frame_count}")

        counter.print_summary()

    cv2.imshow("Vehicle Counter", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()

print()

print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

counter.print_summary()

