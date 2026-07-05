from backend.detector import YOLODetector
import cv2

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/cam4.mp4"

detector = YOLODetector(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()

if not ret:
    print("Cannot read video.")
    exit()

results = detector.detect_frame(frame)

detections = detector.get_detections(results)

detector.print_detections(detections)

annotated = detector.draw_boxes(frame, results)

cv2.imshow("YOLO Detection", annotated)

cv2.waitKey(0)

cv2.destroyAllWindows()

cap.release()