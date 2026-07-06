import cv2

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.density import DensityCalculator
from backend.signal_controller import SignalController

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/cam4.mp4"
LANE_CONFIG = "config/lanes.json"

tracker = VehicleTracker(MODEL_PATH)
lane_manager = LaneManager(LANE_CONFIG)
density = DensityCalculator()
signal = SignalController()

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    tracks, annotated = tracker.track_frame(frame)

    lane_objects = lane_manager.assign_lanes(tracks)

    _, class_density = density.calculate_density(lane_objects)

    if frame_count % 60 == 0:

        plan = signal.generate_signal_plan(class_density)

        print(f"\nFrame : {frame_count}")

        signal.print_signal_plan(plan)

    cv2.imshow("Adaptive Signal", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()