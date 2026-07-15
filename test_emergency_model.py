"""
============================================================
Test Emergency Model (Standalone)
============================================================
Loads the retrained emergency model and runs it on a video
to verify detections of ambulance, fire_truck, and police.
============================================================
"""

import cv2
import os
import sys
from ultralytics import YOLO

# Paths - relative to the SmartTrafficProject directory where this script lives
import pathlib
_SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
MODEL_PATH = str(_SCRIPT_DIR / "models" / "emergency_best.pt")
VIDEO_PATH = str(_SCRIPT_DIR / "videos" / "cam4.mp4")
CONF = 0.25
IOU = 0.45

def main():
    # ----------------------------------------------------------
    # 1. Load Model
    # ----------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("Loading Emergency Model...")
    print("=" * 60)

    model = YOLO(MODEL_PATH)
    class_names = model.names

    print(f"\nModel classes ({len(class_names)}):")
    for idx, name in class_names.items():
        print(f"  {idx} -> {name}")
    print("=" * 60)

    # ----------------------------------------------------------
    # 2. Open Video
    # ----------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nVideo: {VIDEO_PATH}")
    print(f"Frames: {total_frames} | FPS: {fps:.2f}\n")

    frame_count = 0
    emergency_total = 0
    seen_track_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for speed (process every 3rd frame)
        if frame_count % 3 != 0:
            continue

        # ----------------------------------------------------------
        # 3. Run Detection
        # ----------------------------------------------------------
        results = model.track(
            source=frame,
            conf=CONF,
            iou=IOU,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            continue

        frame_emergencies = []

        for box in result.boxes:
            track_id = int(box.id[0])
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            frame_emergencies.append({
                "class_name": class_name,
                "confidence": confidence,
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2]
            })

            if track_id not in seen_track_ids:
                seen_track_ids.add(track_id)
                emergency_total += 1

        # ----------------------------------------------------------
        # 4. Print detections for this frame
        # ----------------------------------------------------------
        if frame_emergencies:
            print(f"\n--- Frame {frame_count} ---")
            for det in frame_emergencies:
                print(f"  [{det['class_name']}] "
                      f"conf={det['confidence']:.3f} "
                      f"track_id={det['track_id']} "
                      f"bbox={det['bbox']}")

        # ----------------------------------------------------------
        # 5. Annotate and display (optional, press 'q' to quit)
        # ----------------------------------------------------------
        annotated = result.plot()
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Emergencies: {len(frame_emergencies)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Emergency Model Test", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # ----------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("EMERGENCY MODEL TEST COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Unique emergency vehicles detected: {emergency_total}")
    print(f"Unique track IDs: {sorted(seen_track_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    main()