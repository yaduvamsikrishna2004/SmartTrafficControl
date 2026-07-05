import cv2

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.density import DensityCalculator

MODEL_PATH="models/best.pt"

VIDEO_PATH="videos/cam4.mp4"

LANE_CONFIG="config/lanes.json"

tracker=VehicleTracker(MODEL_PATH)

lane_manager=LaneManager(LANE_CONFIG)

density=DensityCalculator()

cap=cv2.VideoCapture(VIDEO_PATH)

frame_count=0

while True:

    ret,frame=cap.read()

    if not ret:
        break

    frame_count+=1

    tracks,annotated=tracker.track_frame(frame)

    lane_objects=lane_manager.assign_lanes(tracks)

    lane_density,class_density=density.calculate_density(lane_objects)

    if frame_count%60==0:

        print()

        print(f"Frame : {frame_count}")

        density.print_density(class_density)

    cv2.imshow("Traffic Density",annotated)

    if cv2.waitKey(1)==ord("q"):
        break

cap.release()

cv2.destroyAllWindows()