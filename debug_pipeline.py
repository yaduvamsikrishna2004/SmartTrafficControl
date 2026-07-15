"""
============================================================
COMPLETE PIPELINE DEBUG
============================================================
Investigates why ambulance is shown as bus/van in final output.

Steps:
1. Run emergency model only → outputs/emergency_only.mp4
2. Run vehicle model only → outputs/vehicle_only.mp4
3. Frame-by-frame comparison with IoU
4. Verify merge logic
5. Print final merged detections
6. Verify priority engine
7. Root cause determination
============================================================
"""

import cv2
import os
import sys
import json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

import pathlib
_SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()

EMERGENCY_MODEL_PATH = str(_SCRIPT_DIR / "models" / "emergency_best.pt")
VEHICLE_MODEL_PATH = str(_SCRIPT_DIR / "models" / "best.pt")
VIDEO_PATH = str(_SCRIPT_DIR / "videos" / "cam4.mp4")
OUTPUT_DIR = str(_SCRIPT_DIR / "outputs")
LANE_CONFIG_PATH = str(_SCRIPT_DIR / "config" / "lanes.json")

EMERGENCY_CONF = 0.25
VEHICLE_CONF = 0.30
IOU_THRESH = 0.45

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emergency classes that should override vehicle classes
EMERGENCY_CLASSES = {"ambulance", "firetruck", "police vehicle", "fire_truck", "police"}
EMERGENCY_CANONICAL = {
    "ambulance": "ambulance",
    "firetruck": "fire_truck",
    "fire_truck": "fire_truck",
    "police vehicle": "police",
    "police": "police",
    "police_car": "police",
}

# ==========================================================
# Helper: compute IoU
# ==========================================================
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# ==========================================================
# Helper: load lane config
# ==========================================================
def load_lane_config(path):
    with open(path) as f:
        return json.load(f)

# ==========================================================
# Helper: get lane for a point
# ==========================================================
def get_lane_for_point(center, lane_config):
    x, y = center
    for lane_name, polygon in lane_config.items():
        pts = np.array(polygon, dtype=np.int32)
        if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
            return lane_name
    return "Unknown"

# ==========================================================
# STEP 1: Run emergency model only
# ==========================================================
def step1_emergency_only():
    print("\n" + "=" * 70)
    print("STEP 1: EMERGENCY MODEL ONLY")
    print("=" * 70)

    model = YOLO(EMERGENCY_MODEL_PATH)
    class_names = model.names
    print(f"Emergency model classes: {class_names}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "emergency_only.mp4"), fourcc, fps, (w, h))

    frame_count = 0
    emergency_detections = []  # list of (frame, class, conf, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(
            source=frame, conf=EMERGENCY_CONF, iou=IOU_THRESH,
            persist=True, tracker="bytetrack.yaml", verbose=False
        )
        result = results[0]
        annotated = result.plot()

        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls]
                bbox = list(map(int, box.xyxy[0]))
                emergency_detections.append((frame_count, class_name, conf, bbox, track_id))

                # Print every detection
                print(f"Frame {frame_count}")
                print(f"  Class      : {class_name}")
                print(f"  Confidence : {conf:.3f}")
                print(f"  BBox       : {bbox}")
                print(f"  Track ID   : {track_id}")
                print()

                # Draw on frame
                cv2.putText(annotated, f"{class_name} {conf:.2f}", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(annotated)

    cap.release()
    out.release()
    print(f"\n[STEP 1] Emergency model processed {frame_count} frames")
    print(f"[STEP 1] Total emergency detections: {len(emergency_detections)}")
    print(f"[STEP 1] Saved to outputs/emergency_only.mp4")
    return emergency_detections

# ==========================================================
# STEP 2: Run vehicle model only
# ==========================================================
def step2_vehicle_only():
    print("\n" + "=" * 70)
    print("STEP 2: VEHICLE MODEL ONLY")
    print("=" * 70)

    model = YOLO(VEHICLE_MODEL_PATH)
    class_names = model.names
    print(f"Vehicle model classes: {class_names}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "vehicle_only.mp4"), fourcc, fps, (w, h))

    frame_count = 0
    vehicle_detections = []  # list of (frame, class, conf, bbox, track_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(
            source=frame, conf=VEHICLE_CONF, iou=IOU_THRESH,
            persist=True, tracker="bytetrack.yaml", verbose=False
        )
        result = results[0]
        annotated = result.plot()

        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls]
                bbox = list(map(int, box.xyxy[0]))
                vehicle_detections.append((frame_count, class_name, conf, bbox, track_id))

                # Print every detection
                print(f"Frame {frame_count}")
                print(f"  Class      : {class_name}")
                print(f"  Confidence : {conf:.3f}")
                print(f"  BBox       : {bbox}")
                print(f"  Track ID   : {track_id}")
                print()

                cv2.putText(annotated, f"{class_name} {conf:.2f}", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out.write(annotated)

    cap.release()
    out.release()
    print(f"\n[STEP 2] Vehicle model processed {frame_count} frames")
    print(f"[STEP 2] Total vehicle detections: {len(vehicle_detections)}")
    print(f"[STEP 2] Saved to outputs/vehicle_only.mp4")
    return vehicle_detections

# ==========================================================
# STEP 3: Frame-by-frame comparison
# ==========================================================
def step3_comparison(emergency_dets, vehicle_dets):
    print("\n" + "=" * 70)
    print("STEP 3: FRAME-BY-FRAME COMPARISON")
    print("=" * 70)

    # Group by frame
    em_by_frame = defaultdict(list)
    for f, c, conf, bb, tid in emergency_dets:
        em_by_frame[f].append((c, conf, bb, tid))

    veh_by_frame = defaultdict(list)
    for f, c, conf, bb, tid in vehicle_dets:
        veh_by_frame[f].append((c, conf, bb, tid))

    all_frames = sorted(set(em_by_frame.keys()) | set(veh_by_frame.keys()))

    comparison_results = []

    for frame_num in all_frames:
        em_objs = em_by_frame.get(frame_num, [])
        veh_objs = veh_by_frame.get(frame_num, [])

        if not em_objs or not veh_objs:
            continue

        for em_obj in em_objs:
            em_class, em_conf, em_bb, em_tid = em_obj
            for veh_obj in veh_objs:
                veh_class, veh_conf, veh_bb, veh_tid = veh_obj
                iou = compute_iou(em_bb, veh_bb)

                if iou > 0.1:  # Report any significant overlap
                    comparison_results.append({
                        "frame": frame_num,
                        "vehicle_class": veh_class,
                        "vehicle_conf": veh_conf,
                        "vehicle_bbox": veh_bb,
                        "vehicle_track_id": veh_tid,
                        "emergency_class": em_class,
                        "emergency_conf": em_conf,
                        "emergency_bbox": em_bb,
                        "emergency_track_id": em_tid,
                        "iou": iou
                    })

                    print(f"Frame {frame_num}")
                    print(f"  Vehicle Model:")
                    print(f"    Class      : {veh_class}")
                    print(f"    Confidence : {veh_conf:.3f}")
                    print(f"    BBox       : {veh_bb}")
                    print(f"    Track ID   : {veh_tid}")
                    print(f"  Emergency Model:")
                    print(f"    Class      : {em_class}")
                    print(f"    Confidence : {em_conf:.3f}")
                    print(f"    BBox       : {em_bb}")
                    print(f"    Track ID   : {em_tid}")
                    print(f"  IoU         : {iou:.3f}")
                    print()

    print(f"[STEP 3] Found {len(comparison_results)} overlapping detections")
    return comparison_results

# ==========================================================
# STEP 4 & 5: Verify merge logic + final merged detections
# ==========================================================
def step4_5_merge_and_final(emergency_dets, vehicle_dets):
    print("\n" + "=" * 70)
    print("STEP 4 & 5: MERGE LOGIC VERIFICATION + FINAL DETECTIONS")
    print("=" * 70)

    lane_config = load_lane_config(LANE_CONFIG_PATH)

    # Group by frame
    em_by_frame = defaultdict(list)
    for f, c, conf, bb, tid in emergency_dets:
        em_by_frame[f].append((c, conf, bb, tid))

    veh_by_frame = defaultdict(list)
    for f, c, conf, bb, tid in vehicle_dets:
        veh_by_frame[f].append((c, conf, bb, tid))

    all_frames = sorted(set(em_by_frame.keys()) | set(veh_by_frame.keys()))

    # Simulate the merge logic from traffic_engine.py
    # For each frame, we have vehicle tracked objects and emergency detections
    # The merge logic in traffic_engine.py:
    # 1. Start with lane_objects (from vehicle tracker)
    # 2. For each emergency detection, try to find matching track
    # 3. If matched, set emergency flags on the track
    # 4. If unmatched, add as new object
    # 5. The final class_name comes from the vehicle model unless overridden

    # The critical question: does the merge logic override the class_name?

    merged_results = []

    for frame_num in all_frames:
        em_objs = em_by_frame.get(frame_num, [])
        veh_objs = veh_by_frame.get(frame_num, [])

        # Build tracked objects (from vehicle model) - same as lane_objects
        tracked_objects = []
        for veh_class, veh_conf, veh_bb, veh_tid in veh_objs:
            center_x = (veh_bb[0] + veh_bb[2]) // 2
            center_y = (veh_bb[1] + veh_bb[3]) // 2
            lane = get_lane_for_point((center_x, center_y), lane_config)
            tracked_objects.append({
                "track_id": veh_tid,
                "class_name": veh_class,
                "confidence": veh_conf,
                "bbox": veh_bb,
                "center": (center_x, center_y),
                "lane": lane,
                "source": "vehicle_model",
                "emergency": False
            })

        # Process emergency detections (same as emergency.detect())
        emergency_list = []
        for em_class, em_conf, em_bb, em_tid in em_objs:
            center_x = (em_bb[0] + em_bb[2]) // 2
            center_y = (em_bb[1] + em_bb[3]) // 2
            lane = get_lane_for_point((center_x, center_y), lane_config)

            # Normalize class name (same as _normalize_class_name in emergency.py)
            normalized = None
            label = em_class.lower().replace("-", " ").replace("_", " ").strip()
            if label == "ambulance":
                normalized = "ambulance"
            elif label in ("firetruck", "fire truck", "fire_truck"):
                normalized = "fire_truck"
            elif label in ("police", "police vehicle", "police car", "police_car", "policevehicle"):
                normalized = "police"

            if normalized:
                emergency_list.append({
                    "vehicle": normalized,
                    "class_name": normalized,
                    "lane": lane,
                    "confidence": em_conf,
                    "track_id": em_tid,
                    "bbox": em_bb,
                    "center": (center_x, center_y),
                    "source_model": "emergency_model"
                })

        # --- MERGE LOGIC (replicating traffic_engine.py lines 184-218) ---

        # Step 4: Match emergency to tracked vehicles
        for emergency in emergency_list:
            track = _find_matching_track(emergency, tracked_objects)
            if track is not None:
                track["emergency"] = True
                track["emergency_vehicle"] = emergency["vehicle"]
                track["emergency_confidence"] = emergency["confidence"]
                track["emergency_track_id"] = emergency["track_id"]
                emergency["matched_track_id"] = track["track_id"]
            else:
                emergency["matched_track_id"] = None

        # Step 5: Merge detections
        combined_objects = list(tracked_objects)  # Start with vehicle objects

        for emergency in emergency_list:
            if emergency.get("matched_track_id") is None:
                lane = emergency.get("lane")
                combined_objects.append({
                    "track_id": emergency["track_id"],
                    "class_name": emergency["class_name"],
                    "confidence": emergency["confidence"],
                    "bbox": emergency["bbox"],
                    "center": emergency["center"],
                    "lane": lane,
                    "source": "emergency_model",
                    "emergency": True,
                    "vehicle": emergency["vehicle"]
                })

        # --- NOW PRINT THE MERGED RESULTS ---
        for obj in combined_objects:
            # Find matching emergency
            matching_em = None
            for em in emergency_list:
                if em.get("matched_track_id") == obj["track_id"]:
                    matching_em = em
                    break
                if em["track_id"] == obj["track_id"] and obj.get("source") == "emergency_model":
                    matching_em = em
                    break

            # Find matching vehicle
            matching_veh = None
            for veh_class, veh_conf, veh_bb, veh_tid in veh_objs:
                if veh_tid == obj["track_id"]:
                    matching_veh = (veh_class, veh_conf)
                    break

            veh_class_str = matching_veh[0] if matching_veh else "N/A"
            em_class_str = matching_em["vehicle"] if matching_em else "N/A"

            # Determine final class (this is what the pipeline outputs)
            if matching_em:
                final_class = matching_em["vehicle"]
            else:
                final_class = obj["class_name"]

            merged_results.append({
                "frame": frame_num,
                "track_id": obj["track_id"],
                "vehicle_class": veh_class_str,
                "emergency_class": em_class_str,
                "final_class": final_class,
                "lane": obj.get("lane", "Unknown"),
                "source": obj.get("source", "vehicle_model"),
                "emergency": obj.get("emergency", False)
            })

            # Print only when there's an emergency involved
            if matching_em or obj.get("emergency"):
                print(f"Frame {frame_num}")
                print(f"  Track ID        : {obj['track_id']}")
                print(f"  Vehicle Model   : {veh_class_str}")
                print(f"  Emergency Model : {em_class_str}")
                print(f"  Final Class     : {final_class}")
                print(f"  Lane            : {obj.get('lane', 'Unknown')}")
                print(f"  Source          : {obj.get('source', 'vehicle_model')}")
                print()

    print(f"[STEP 4&5] Total merged objects with emergency involvement: {len(merged_results)}")
    return merged_results

def _find_matching_track(emergency, tracked_objects):
    """Replicates traffic_engine.py _find_matching_track"""
    emergency_center = emergency.get("center")
    emergency_lane = emergency.get("lane")

    if emergency_center is None:
        return None

    ex, ey = emergency_center

    for obj in tracked_objects:
        if emergency_lane and emergency_lane != "Unknown":
            if obj.get("lane") != emergency_lane:
                continue

        bbox = obj.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        if x1 <= ex <= x2 and y1 <= ey <= y2:
            if emergency_lane and emergency_lane != "Unknown" and obj.get("lane") != emergency_lane:
                obj["lane"] = emergency_lane
            return obj

        center = obj.get("center")
        if center:
            cx, cy = center
            if abs(cx - ex) <= 50 and abs(cy - ey) <= 50:
                return obj

    return None

# ==========================================================
# STEP 6: Verify priority engine
# ==========================================================
def step6_priority_engine(merged_results, emergency_dets):
    print("\n" + "=" * 70)
    print("STEP 6: PRIORITY ENGINE VERIFICATION")
    print("=" * 70)

    # Group emergency detections by frame
    em_by_frame = defaultdict(list)
    for f, c, conf, bb, tid in emergency_dets:
        em_by_frame[f].append((c, conf, bb, tid))

    lane_config = load_lane_config(LANE_CONFIG_PATH)

    priority_table = {
        "ambulance": 100,
        "fire_truck": 90,
        "police": 80
    }

    for frame_num in sorted(em_by_frame.keys()):
        em_objs = em_by_frame[frame_num]

        # Build emergency objects like the pipeline does
        emergency_objects = []
        for em_class, em_conf, em_bb, em_tid in em_objs:
            center_x = (em_bb[0] + em_bb[2]) // 2
            center_y = (em_bb[1] + em_bb[3]) // 2
            lane = get_lane_for_point((center_x, center_y), lane_config)

            label = em_class.lower().replace("-", " ").replace("_", " ").strip()
            if label == "ambulance":
                normalized = "ambulance"
            elif label in ("firetruck", "fire truck", "fire_truck"):
                normalized = "fire_truck"
            elif label in ("police", "police vehicle", "police car", "police_car", "policevehicle"):
                normalized = "police"
            else:
                continue

            emergency_objects.append({
                "vehicle": normalized,
                "lane": lane,
                "confidence": em_conf,
                "track_id": em_tid
            })

        if not emergency_objects:
            continue

        # Evaluate priority (same as PriorityEngine.evaluate)
        best = None
        highest_priority = -1

        for vehicle in emergency_objects:
            vehicle_type = vehicle["vehicle"]
            priority = priority_table.get(vehicle_type, 0)
            if priority > highest_priority:
                highest_priority = priority
                best = vehicle
            elif priority == highest_priority and best is not None:
                if vehicle["confidence"] > best["confidence"]:
                    best = vehicle

        if best:
            decision = {
                "active": True,
                "vehicle": best["vehicle"],
                "lane": best["lane"],
                "track_id": best["track_id"],
                "confidence": best["confidence"],
                "priority": highest_priority,
                "override": True,
                "reason": "Highest Priority Emergency Vehicle"
            }

            print(f"Frame {frame_num}")
            print(f"  Emergency Detected : YES")
            print(f"  Vehicle            : {decision['vehicle']}")
            print(f"  Lane               : {decision['lane']}")
            print(f"  Priority           : {decision['priority']}")
            print(f"  Signal Override    : {decision['override']}")
            print(f"  Green Lane         : {decision['lane']}")
            print(f"  Reason             : {decision['reason']}")
            print()

    print("[STEP 6] Priority engine verification complete")

# ==========================================================
# STEP 7: Root cause analysis
# ==========================================================
def step7_root_cause(emergency_dets, vehicle_dets, comparison_results, merged_results):
    print("\n" + "=" * 70)
    print("STEP 7: ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # Check if emergency model ever predicts emergency classes
    em_classes = set()
    for f, c, conf, bb, tid in emergency_dets:
        em_classes.add(c.lower())
    print(f"Emergency model predicted classes: {em_classes}")

    # Check if emergency model predicts bus/van for the same object
    # that vehicle model predicts as ambulance
    for comp in comparison_results:
        em_class = comp["emergency_class"].lower()
        veh_class = comp["vehicle_class"].lower()

        # Check if emergency model says bus but vehicle model says ambulance
        if em_class in ("bus", "van", "car", "twowheelers", "auto-rikshaw") and \
           veh_class in ("ambulance", "firetruck", "police", "police vehicle", "fire_truck"):
            print(f"ISSUE: Emergency model predicts '{em_class}' but vehicle model predicts '{veh_class}'")
            print(f"  Frame: {comp['frame']}, IoU: {comp['iou']:.3f}")
            print(f"  Emergency bbox: {comp['emergency_bbox']}")
            print(f"  Vehicle bbox: {comp['vehicle_bbox']}")
            print()

        # Check if vehicle model says bus but emergency model says ambulance
        if veh_class in ("bus", "van", "car") and \
           em_class in ("ambulance", "firetruck", "police", "police vehicle", "fire_truck"):
            print(f"ISSUE: Vehicle model predicts '{veh_class}' but emergency model predicts '{em_class}'")
            print(f"  Frame: {comp['frame']}, IoU: {comp['iou']:.3f}")
            print(f"  Emergency bbox: {comp['emergency_bbox']}")
            print(f"  Vehicle bbox: {comp['vehicle_bbox']}")
            print()

    # Check merge results for class override failures
    for mr in merged_results:
        if mr["emergency_class"] != "N/A" and mr["emergency_class"] != mr["final_class"]:
            print(f"MERGE ISSUE: Emergency class '{mr['emergency_class']}' != Final class '{mr['final_class']}'")
            print(f"  Frame: {mr['frame']}, Track ID: {mr['track_id']}")
            print()

    # Summary
    print("\n" + "=" * 70)
    print("ROOT CAUSE SUMMARY")
    print("=" * 70)

    # Count how many times emergency model predicted each class
    em_class_counts = defaultdict(int)
    for f, c, conf, bb, tid in emergency_dets:
        em_class_counts[c] += 1
    print(f"Emergency model prediction counts: {dict(em_class_counts)}")

    # Count how many times vehicle model predicted each class
    veh_class_counts = defaultdict(int)
    for f, c, conf, bb, tid in vehicle_dets:
        veh_class_counts[c] += 1
    print(f"Vehicle model prediction counts: {dict(veh_class_counts)}")

    # Check if the emergency model is misclassifying
    emergency_emergency_count = sum(1 for f, c, conf, bb, tid in emergency_dets
                                     if c.lower() in ("ambulance", "firetruck", "police vehicle", "police", "fire_truck"))
    emergency_non_emergency_count = len(emergency_dets) - emergency_emergency_count

    print(f"\nEmergency model - Emergency class predictions: {emergency_emergency_count}")
    print(f"Emergency model - Non-emergency class predictions: {emergency_non_emergency_count}")

    if emergency_non_emergency_count > emergency_emergency_count:
        print("\n>>> ROOT CAUSE: The emergency model is predicting non-emergency classes")
        print("    more often than emergency classes. This indicates the model itself")
        print("    is not properly trained to distinguish emergency vehicles.")
    elif emergency_emergency_count > 0:
        print("\n>>> The emergency model IS detecting emergency vehicles correctly.")
        print("    The issue may be in the merge logic or class mapping.")
    else:
        print("\n>>> ROOT CAUSE: The emergency model NEVER predicted any emergency class.")
        print("    The model is completely failing to detect emergency vehicles.")

    # Check merge logic
    override_failures = sum(1 for mr in merged_results
                           if mr["emergency_class"] != "N/A" and mr["emergency_class"] != mr["final_class"])
    if override_failures > 0:
        print(f"\n>>> MERGE LOGIC ISSUE: {override_failures} cases where emergency class")
        print("    was not properly overriding the vehicle class.")

    print("\n" + "=" * 70)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE PIPELINE DEBUG INVESTIGATION")
    print("=" * 70)

    # Step 1
    emergency_dets = step1_emergency_only()

    # Step 2
    vehicle_dets = step2_vehicle_only()

    # Step 3
    comparison_results = step3_comparison(emergency_dets, vehicle_dets)

    # Step 4 & 5
    merged_results = step4_5_merge_and_final(emergency_dets, vehicle_dets)

    # Step 6
    step6_priority_engine(merged_results, emergency_dets)

    # Step 7
    step7_root_cause(emergency_dets, vehicle_dets, comparison_results, merged_results)

    print("\nDebug complete. Check outputs/ for annotated videos.")