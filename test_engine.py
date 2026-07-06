import cv2

from backend.traffic_engine import TrafficEngine

MODEL = "models/best.pt"

VIDEO = "videos/cam4.mp4"

LANES = "config/lanes.json"

engine = TrafficEngine(MODEL, LANES)

cap = cv2.VideoCapture(VIDEO)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    output = engine.process_frame(frame)

    cv2.imshow(
        "Traffic Engine",
        output["frame"]
    )

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()

print("\nFinal Counter\n")

print(output["counter"])

print("\nSignal Plan\n")

print(output["signals"])