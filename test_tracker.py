from backend.tracker import VehicleTracker
import cv2

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/cam4.mp4"

tracker = VehicleTracker(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    tracks, annotated = tracker.track_frame(frame)

    tracker.print_tracks(tracks)

    cv2.imshow("Vehicle Tracking", annotated)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()