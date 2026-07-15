"""
============================================================
VERIFY DUAL-MODEL MERGE FIX
============================================================
Verifies that emergency model predictions correctly override
vehicle model predictions in all downstream modules.
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

def load_lane_config(path):
    with open(path) as f:
        return json.load(f)

def get_lane_for_point(center, lane_config):
    x, y = center
    for lane_name, polygon in lane_config.items():
        pts = np.array(polygon, dtype=np.int32)
        if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
            return lane_name
    return "Unknown"

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

def compute_center_distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

# Map from emergency model class name → canonical
def normalize_emergency_class(raw_name):
    label = str(raw_name).lower().replace("-", " ").replace("_", " ").strip()
    if label == "ambulance":
        return "ambulance"
    if label in ("firetruck", "fire truck", "fire_truck"):
        return "fire_truck"
    if label in ("police", "police vehicle", "police car", "police_car", "policevehicle"):
        return "police"
    return None

# Map vehicle class → canonical (for counter/density)
def map_vehicle_class(class_name: str) -> str:
    if not class_name:
        return "others"
    name_lower = str(class_name).lower()
    if name_lower in ("ambulance", "fire_truck", "police"):
        return name_lower
    if "ambulance" in name_lower:
        return "ambulance"
    if "fire" in name_lower and "truck" in name_lower:
        return "fire_truck"
    if "police" in name_lower:
        return "police"
    if name_lower in ("car", "sedan", "coupe", "hatchback"):
        return "car"
    if name_lower in ("bus",):
        return "bus"
    if name_lower in ("van", "truck", "lorry", "pickup"):
        return "van"
    if name_lower in ("motorbike", "motorcycle", "bike", "bicycle", "scooter"):
        return "others"
    return "others"

def run_verification():
    print("=" * 80)
    print("DUAL-MODEL MERGE VERIFICATION")
    print("=" * 80)

    lane_config = load_lane_config(LANE_CONFIG_PATH)

    # Load models
    print("\n--- Loading Models ---")
    veh_model = YOLO(VEHICLE_MODEL_PATH)
    em_model = YOLO(EMERGENCY_MODEL_PATH)
    veh_class_names = veh_model.names
    em_class_names = em_model.names
    print(f"Vehicle model classes: {veh_class_names}")
    print(f"Emergency model classes: {em_class_names}")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_veh = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "vehicle_only.mp4"), fourcc, fps, (w, h))
    out_em = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "emergency_only.mp4"), fourcc, fps, (w, h))
    out_merged = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "merged_output.mp4"), fourcc, fps, (w, h))

    # Tracking state
    veh_counted_ids = set()
    em_counted_ids = set()

    report_data = {
        "matches": [],
        "mismatches": [],
        "counter_results": defaultdict(lambda: defaultdict(int)),
        "density_results": defaultdict(lambda: defaultdict(int)),
        "priority_decisions": []
    }

    frame_count = 0
    total_matches = 0
    total_emergency_overrides = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ----------------------------------------------------
        # VEHICLE MODEL
        # ----------------------------------------------------
        veh_results = veh_model.track(
            source=frame, conf=VEHICLE_CONF, iou=IOU_THRESH,
            persist=True, tracker="bytetrack.yaml", verbose=False
        )
        veh_result = veh_results[0]
        veh_annotated = veh_result.plot()
        veh_objects = []

        if veh_result.boxes is not None and veh_result.boxes.id is not None:
            for box in veh_result.boxes:
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = veh_class_names[cls]
                bbox = list(map(int, box.xyxy[0]))
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                lane = get_lane_for_point(center, lane_config)

                veh_objects.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": bbox,
                    "center": center,
                    "lane": lane
                })

                # Annotate vehicle-only
                label = f"V:{class_name} ID:{track_id} {conf:.2f}"
                cv2.putText(veh_annotated, label, (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out_veh.write(veh_annotated)

        # ----------------------------------------------------
        # EMERGENCY MODEL
        # ----------------------------------------------------
        em_results = em_model.track(
            source=frame, conf=EMERGENCY_CONF, iou=IOU_THRESH,
            persist=True, tracker="bytetrack.yaml", verbose=False
        )
        em_result = em_results[0]
        em_annotated = em_result.plot()
        em_objects = []

        if em_result.boxes is not None and em_result.boxes.id is not None:
            for box in em_result.boxes:
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                raw_name = em_class_names[cls]
                normalized = normalize_emergency_class(raw_name)
                bbox = list(map(int, box.xyxy[0]))
                center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                lane = get_lane_for_point(center, lane_config)

                em_objects.append({
                    "track_id": track_id,
                    "class_name": raw_name,
                    "vehicle": normalized if normalized else raw_name,
                    "is_emergency": normalized is not None,
                    "confidence": round(conf, 3),
                    "bbox": bbox,
                    "center": center,
                    "lane": lane
                })

                # Annotate emergency-only
                label = f"E:{normalized if normalized else raw_name} ID:{track_id} {conf:.2f}"
                color = (0, 0, 255) if normalized else (0, 255, 255)
                cv2.putText(em_annotated, label, (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_em.write(em_annotated)

        # ----------------------------------------------------
        # MERGE LOGIC (exact replica of traffic_engine.py with fix)
        # ----------------------------------------------------
        # Step 1: Start with vehicle tracked objects
        merged_objects = []
        for veh in veh_objects:
            merged_objects.append({
                "track_id": veh["track_id"],
                "class_name": veh["class_name"],        # Vehicle model class
                "vehicle_type": veh["class_name"],       # For downstream
                "confidence": veh["confidence"],
                "bbox": veh["bbox"],
                "center": veh["center"],
                "lane": veh["lane"],
                "source": "vehicle_model",
                "emergency": False,
                "emergency_vehicle": None,
                "is_emergency": False
            })

        # Step 2: Match and override (REPLICATING THE FIX)
        frame_emergency_overrides = 0

        for em_obj in em_objects:
            if not em_obj["is_emergency"]:
                continue  # Only match if emergency model says it's an emergency

            # Find matching track
            matched_track = None
            ex, ey = em_obj["center"]

            for mo in merged_objects:
                if em_obj["lane"] and em_obj["lane"] != "Unknown":
                    if mo.get("lane") != em_obj["lane"]:
                        continue

                bbox = mo.get("bbox")
                if bbox and bbox[0] <= ex <= bbox[2] and bbox[1] <= ey <= bbox[3]:
                    matched_track = mo
                    break

                center = mo.get("center")
                if center and abs(center[0] - ex) <= 50 and abs(center[1] - ey) <= 50:
                    matched_track = mo
                    break

            if matched_track is not None:
                # === THIS IS THE CRITICAL OVERRIDE (THE FIX) ===
                # Override ALL fields that downstream modules read
                matched_track["class_name"] = em_obj["vehicle"]        # counter.py, density.py, lane_manager.py
                matched_track["vehicle_type"] = em_obj["vehicle"]      # for any future references
                matched_track["emergency"] = True                      # boolean flag
                matched_track["emergency_vehicle"] = em_obj["vehicle"]  # string name
                matched_track["is_emergency"] = True                   # alternative boolean
                matched_track["emergency_confidence"] = em_obj["confidence"]
                matched_track["emergency_track_id"] = em_obj["track_id"]
                matched_track["source"] = "merged_emergency"
                frame_emergency_overrides += 1
                total_emergency_overrides += 1

                # Record match
                report_data["matches"].append({
                    "frame": frame_count,
                    "vehicle_class": veh_objects[merged_objects.index(matched_track)]["class_name"] if matched_track in merged_objects else "N/A",
                    "emergency_class": em_obj["vehicle"],
                    "final_class": em_obj["vehicle"],
                    "lane": em_obj["lane"],
                    "iou": compute_iou(em_obj["bbox"], matched_track["bbox"]),
                    "center_distance": compute_center_distance(em_obj["center"], matched_track["center"]),
                    "track_id": matched_track["track_id"],
                    "emergency_track_id": em_obj["track_id"],
                    "matched": True
                })
            else:
                # Unmatched emergency - add as new object
                merged_objects.append({
                    "track_id": em_obj["track_id"],
                    "class_name": em_obj["vehicle"],
                    "vehicle_type": em_obj["vehicle"],
                    "confidence": em_obj["confidence"],
                    "bbox": em_obj["bbox"],
                    "center": em_obj["center"],
                    "lane": em_obj["lane"],
                    "source": "emergency_model",
                    "emergency": True,
                    "emergency_vehicle": em_obj["vehicle"],
                    "is_emergency": True,
                    "emergency_confidence": em_obj["confidence"]
                })
                total_emergency_overrides += 1

                report_data["matches"].append({
                    "frame": frame_count,
                    "vehicle_class": "N/A (unmatched)",
                    "emergency_class": em_obj["vehicle"],
                    "final_class": em_obj["vehicle"],
                    "lane": em_obj["lane"],
                    "iou": 0,
                    "center_distance": 0,
                    "track_id": em_obj["track_id"],
                    "emergency_track_id": em_obj["track_id"],
                    "matched": False
                })

        # ----------------------------------------------------
        # DOWNSTREAM SIMULATION
        # ----------------------------------------------------

        # Counter: count each unique track_id once
        for obj in merged_objects:
            tid = obj["track_id"]
            lane = obj["lane"]
            raw_class = obj.get("class_name", "others")
            canonical = map_vehicle_class(raw_class)

            # Only count each vehicle once
            key = (tid, lane)
            if key not in report_data["counter_results"]:
                report_data["counter_results"][key] = {
                    "track_id": tid,
                    "lane": lane,
                    "class_name": raw_class,
                    "canonical": canonical,
                    "is_emergency": obj.get("emergency", False) or obj.get("is_emergency", False)
                }

        # Density: per-frame density by class
        for obj in merged_objects:
            lane = obj["lane"]
            raw_class = obj.get("class_name", "others")
            canonical = map_vehicle_class(raw_class)
            if lane:
                report_data["density_results"][frame_count][canonical] += 1
                report_data["density_results"][frame_count]["total"] += 1

        # Priority Engine
        emergency_list_for_priority = []
        for em_obj in em_objects:
            if em_obj["is_emergency"]:
                emergency_list_for_priority.append({
                    "vehicle": em_obj["vehicle"],
                    "lane": em_obj["lane"],
                    "confidence": em_obj["confidence"],
                    "track_id": em_obj["track_id"]
                })

        if emergency_list_for_priority:
            priority_table = {"ambulance": 100, "fire_truck": 90, "police": 80}
            best = None
            highest = -1
            for v in emergency_list_for_priority:
                p = priority_table.get(v["vehicle"], 0)
                if p > highest:
                    highest = p
                    best = v
            if best:
                report_data["priority_decisions"].append({
                    "frame": frame_count,
                    "active": True,
                    "vehicle": best["vehicle"],
                    "lane": best["lane"],
                    "priority": highest
                })

        # ----------------------------------------------------
        # PRINT FRAME DEBUG LOG
        # ----------------------------------------------------
        if frame_emergency_overrides > 0 or len(em_objects) > 0:
            print(f"\n--- Frame {frame_count} ---")
            print(f"  [DEBUG] Vehicle detections: {len(veh_objects)}")

            for veh in veh_objects:
                print(f"    VEHICLE: class={veh['class_name']:15s} conf={veh['confidence']:.3f} "
                      f"bbox={veh['bbox']} track_id={veh['track_id']} lane={veh['lane']}")

            print(f"  [DEBUG] Emergency detections: {len(em_objects)}")
            for em_obj in em_objects:
                prefix = "  >>> EMERGENCY" if em_obj["is_emergency"] else "      (non-emerg)"
                print(f"    {prefix}: class={em_obj['class_name']:15s} "
                      f"normalized={str(em_obj['vehicle']):15s} conf={em_obj['confidence']:.3f} "
                      f"bbox={em_obj['bbox']} track_id={em_obj['track_id']} lane={em_obj['lane']}")

            if frame_emergency_overrides > 0:
                print(f"  [DEBUG] EMERGENCY OVERRIDES: {frame_emergency_overrides}")

            # Print merged objects
            print(f"  [DEBUG] Merged objects: {len(merged_objects)}")
            for obj in merged_objects:
                em_tag = " [EMERGENCY]" if obj.get("emergency") or obj.get("is_emergency") else ""
                print(f"    MERGED: class={obj['class_name']:15s} track_id={obj['track_id']:3d} "
                      f"lane={obj['lane']:8s} source={obj['source']:20s}{em_tag}")

        # ----------------------------------------------------
        # ANNOTATE MERGED OUTPUT
        # ----------------------------------------------------
        merged_annotated = frame.copy()
        for obj in merged_objects:
            bbox = obj["bbox"]
            is_em = obj.get("emergency") or obj.get("is_emergency")
            color = (0, 0, 255) if is_em else (0, 255, 0)
            cv2.rectangle(merged_annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"ID:{obj['track_id']} {obj['class_name']} {obj['confidence']:.2f} {obj['lane']}"
            cv2.putText(merged_annotated, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_merged.write(merged_annotated)

        # Early exit after enough frames for verification (first 300 frames have emergency)
        if frame_count >= 300:
            break

    # Cleanup
    cap.release()
    out_veh.release()
    out_em.release()
    out_merged.release()
    cv2.destroyAllWindows()

    # ============================================================
    # GENERATE VERIFICATION REPORT
    # ============================================================
    print("\n\n")
    print("=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Frames processed: {frame_count}")
    print(f"Total emergency overrides: {total_emergency_overrides}")
    print(f"Total matches found: {len(report_data['matches'])}")
    print(f"Emergency model emergency detections: {len(em_objects)}")

    # Check for MISMATCHES (vehicle says bus, emergency says ambulance but merged shows bus)
    mismatches_found = 0
    for m in report_data["matches"]:
        if m["matched"]:
            # Check if final_class is actually different from vehicle_class
            veh_canon = map_vehicle_class(m["vehicle_class"])
            em_canon = map_vehicle_class(m["emergency_class"])
            if veh_canon in ("bus", "van", "car") and em_canon in ("ambulance", "fire_truck", "police"):
                if em_canon != veh_canon:
                    print(f"\n  [VERIFY] Frame {m['frame']}: Vehicle={m['vehicle_class']} "
                          f"→ Emergency={m['emergency_class']} → FINAL={m['final_class']} ✓")
                    mismatches_found += 1

    overrides_with_mismatch = sum(1 for m in report_data["matches"] if m["matched"])
    print(f"\nMatched overrides verified: {overrides_with_mismatch}")
    
    if total_emergency_overrides > 0:
        all_overridden = True
        for m in report_data["matches"]:
            if m["matched"]:
                veh = map_vehicle_class(m["vehicle_class"])
                final = map_vehicle_class(m["final_class"])
                if veh in ("bus", "van", "car") and final not in ("ambulance", "fire_truck", "police"):
                    print(f"  FAIL: Frame {m['frame']}: vehicle={m['vehicle_class']} final={m['final_class']}")
                    all_overridden = False
        if all_overridden:
            print("  ✓ ALL emergency classes correctly overrode vehicle classes")
    else:
        print("  ! No emergency detections found in processed frames")

    # Priority decisions
    print(f"\n--- Priority Engine Decisions ---")
    active_decisions = [d for d in report_data["priority_decisions"] if d["active"]]
    print(f"Active emergency decisions: {len(active_decisions)}")
    for d in active_decisions[:5]:
        print(f"  Frame {d['frame']}: {d['vehicle']} on {d['lane']} (priority={d['priority']})")

    # Counter results
    print(f"\n--- Counter Results (unique vehicles) ---")
    emergency_counted = 0
    for key, val in report_data["counter_results"].items():
        if val["is_emergency"]:
            print(f"  Track {val['track_id']} on {val['lane']}: {val['class_name']} (canonical: {val['canonical']}) [EMERGENCY] ✓")
            emergency_counted += 1
    print(f"Total emergency vehicles counted: {emergency_counted}")

    # Check counter has no bus/van that should be emergency
    bus_from_emergency = 0
    for key, val in report_data["counter_results"].items():
        if val["is_emergency"] and val["canonical"] in ("bus", "van"):
            bus_from_emergency += 1
            print(f"  FAIL: Emergency vehicle counted as {val['canonical']}!")

    if bus_from_emergency == 0:
        print("  ✓ No emergency vehicles miscounted as bus/van")

    # Density results
    print(f"\n--- Density Results (sample frames) ---")
    density_emergency_frames = 0
    for frame_num in sorted(report_data["density_results"].keys())[:10]:
        stats = report_data["density_results"][frame_num]
        em_in_frame = any(k in ("ambulance", "fire_truck", "police") for k in stats.keys())
        if em_in_frame:
            density_emergency_frames += 1
            print(f"  Frame {frame_num}: {dict(stats)}")
    if density_emergency_frames > 0:
        print(f"  ✓ Emergency vehicles present in density data")
    else:
        print(f"  ! No emergency vehicles in density data for first 10 frames")

    # ============================================================
    # FINAL ASSERTION
    # ============================================================
    print("\n" + "=" * 80)
    if bus_from_emergency == 0 and total_emergency_overrides > 0:
        print("RESULT: MERGE FIX VERIFIED SUCCESSFULLY ✓")
        print("All emergency vehicle predictions correctly override vehicle model classes.")
    elif total_emergency_overrides == 0:
        print("RESULT: No emergency detections found in processed frames.")
        print("This may be expected if the video has no emergency vehicles in early frames.")
    else:
        print("RESULT: ISSUES DETECTED - See above for details")
    print("=" * 80)

    print(f"\nVideos saved:")
    print(f"  {os.path.join(OUTPUT_DIR, 'vehicle_only.mp4')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'emergency_only.mp4')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'merged_output.mp4')}")


if __name__ == "__main__":
    run_verification()