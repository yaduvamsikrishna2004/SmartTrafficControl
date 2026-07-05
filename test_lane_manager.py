import cv2

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/cam4.mp4"
LANE_CONFIG = "config/lanes.json"

tracker = VehicleTracker(MODEL_PATH)
lane_manager = LaneManager(LANE_CONFIG)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    tracks, annotated = tracker.track_frame(frame)

    lane_objects = lane_manager.assign_lanes(tracks)

    # Print only every 30 frames
    if frame_count % 30 == 0:
        lane_manager.print_lane_assignments(lane_objects)

    # Draw lane name on each vehicle
    for obj in lane_objects:

        x1, y1, x2, y2 = obj["bbox"]

        lane = obj["lane"]

        if lane is not None:

            cv2.putText(
                annotated,
                lane,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    cv2.imshow("Lane Assignment", annotated)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()