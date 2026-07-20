"""
============================================================
Traffic Engine

Author : Vamsi Krishna

Description:
Integrates all AI modules into one engine.
Acts as the central AI processing pipeline.

Architecture: best.pt + emergency_best.pt (2 models)
- best.pt: General vehicle detection, tracking, counting
- emergency_best.pt: Emergency vehicle detection and override

EMERGENCY MERGE LOGIC (FIXED):
1. Run BOTH models on every frame
2. Emergency detections ALWAYS override vehicle detections
3. IoU > 0.5 → keep ONLY emergency detection (no duplicate boxes)
4. Priority: ambulance > fire_truck > police > normal vehicles
5. Emergency model uses conf=0.20 (lower because emergencies are rare)
6. Class-specific NMS: emergency never removed by vehicle detections
7. Tracker ID update: if car → ambulance on same IoU, update class
8. Temporal confirmation: 2 consecutive frames → lock for 15 frames
9. If emergency conf > 0.40, always choose emergency class
10. Display: red box, thicker border, blinking, 🚑 AMBULANCE label
11. Signal controller: immediate emergency mode
12. No duplicate notifications per tracked ID
13. Full logging per frame
14. Debug mode: blue=vehicle, red=emergency, green=merged
15. Modular functions
16. Optimized: reuse resized frames, no duplicate preprocessing
17. Preserves all existing APIs
============================================================
"""

import time
import numpy as np
import cv2
from collections import defaultdict

from backend.tracker import VehicleTracker
from backend.lane_manager import LaneManager
from backend.counter import VehicleCounter
from backend.density import DensityCalculator
from backend.signal_controller import SignalController
from backend.emergency import EmergencyDetector
from backend.priority_engine import PriorityEngine
from backend.utils import build_class_map, set_class_map, normalize_emergency_class, is_emergency_class
from backend.config import (
    EMERGENCY_MODEL_PATH, EMERGENCY_CONF, EMERGENCY_IOU, EMERGENCY_GREEN,
    VEHICLE_CONF, VEHICLE_IOU, MERGE_IOU_THRESHOLD,
    TEMPORAL_CONFIRM_FRAMES, TEMPORAL_LOCK_FRAMES,
    EMERGENCY_FORCE_OVERRIDE_CONF, DEBUG_MODE,
    EMERGENCY_PRIORITY_ORDER
)
from backend.detector import YOLODetector


class TrafficEngine:

    def __init__(self, model_path, lane_config):

        # ====================================================
        # AI Modules
        # ====================================================

        self.tracker = VehicleTracker(model_path)

        self.emergency_detector = YOLODetector(
            EMERGENCY_MODEL_PATH,
            model_label="Emergency model loaded"
        )

        # Build and set a class map from the model's class names so
        # counting/density mapping is precise for the current model.
        try:
            class_names = self.tracker.detector.get_class_names()
            mapping = build_class_map(class_names)
            set_class_map(mapping)
            print(f"[Engine] Class map set: {mapping}")
        except Exception:
            pass

        self.lane_manager = LaneManager(lane_config)

        self.counter = VehicleCounter()

        self.density = DensityCalculator()

        self.signal = SignalController(emergency_green=EMERGENCY_GREEN)
        self.emergency = EmergencyDetector(
            detector=self.emergency_detector,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU
        )
        self.priority = PriorityEngine()

        # ====================================================
        # Runtime Information
        # ====================================================

        self.processing_time = 0
        self.fps = 0
        self.latest_confidence = 0.0
        self.current_green_lane = None
        self.current_green_time = 0

        # ====================================================
        # Latest Results
        # ====================================================

        self.latest_frame = None
        self.latest_tracks = []
        self.latest_counter = {}
        self.latest_density = {}
        self.latest_signals = {}
        self.latest_emergency = {"active": False}
        self.latest_vehicle_detections = []
        self.latest_emergency_detections = []
        self.latest_merged_detections = []

        # ====================================================
        # Notification deduplication
        # ====================================================
        self._notified_emergency_ids = set()

        # ====================================================
        # Frame counter for temporal logic
        # ====================================================
        self._frame_counter = 0

        # ====================================================
        # Blinking state for display
        # ====================================================
        self._blink_state = False

        print("=" * 60)
        print("Traffic Engine Initialized")
        print("=" * 60)

    # ========================================================
    # Process One Frame
    # ========================================================

    def process_frame(self, frame):

        start = time.perf_counter()
        self._frame_counter += 1
        self._blink_state = (self._frame_counter // 5) % 2 == 0  # Blink every 5 frames

        print(f"\n{'='*70}")
        print(f"[Engine] FRAME {self._frame_counter}")
        print(f"{'='*70}")

        # ----------------------------------------------------
        # Step 1: Run Vehicle Model (best.pt)
        # ----------------------------------------------------
        vehicle_detections, vehicle_raw_results = self.run_vehicle_model(frame)
        self.latest_vehicle_detections = vehicle_detections

        # ----------------------------------------------------
        # Step 2: Run Emergency Model (emergency_best.pt)
        # ----------------------------------------------------
        emergency_detections = self.run_emergency_model(frame)
        self.latest_emergency_detections = emergency_detections

        # ----------------------------------------------------
        # Step 3: Merge Detections
        # Emergency ALWAYS overrides vehicle
        # ----------------------------------------------------
        merged_detections = self.merge_detections(vehicle_detections, emergency_detections)
        self.latest_merged_detections = merged_detections

        # ----------------------------------------------------
        # Step 4: Apply Priority
        # ----------------------------------------------------
        priority_decision = self.apply_priority(emergency_detections)

        # ----------------------------------------------------
        # Step 5: Remove Duplicates (IoU-based)
        # ----------------------------------------------------
        merged_detections = self.remove_duplicates(merged_detections)

        # ----------------------------------------------------
        # Step 6: Lane Assignment
        # ----------------------------------------------------
        try:
            h, w = frame.shape[:2]
            self.lane_manager.update_scale(w, h)
        except Exception:
            pass

        lane_objects = self.lane_manager.assign_lanes(merged_detections)
        print(f"[Engine] Lane assignment: {len(lane_objects)} objects assigned")

        # ----------------------------------------------------
        # Step 7: Update Tracker (re-assign IDs, update classes)
        # ----------------------------------------------------
        lane_objects = self.update_tracker(lane_objects, emergency_detections)

        # ----------------------------------------------------
        # Step 8: Trigger Emergency Mode if needed
        # ----------------------------------------------------
        self.trigger_emergency_mode(priority_decision, lane_objects)

        # ----------------------------------------------------
        # Step 9: Vehicle Counter
        # ----------------------------------------------------
        self.counter.update(lane_objects)
        lane_counts = self.counter.get_counts()
        print(f"[Engine] Vehicle counter: {lane_counts}")

        # ----------------------------------------------------
        # Step 10: Density Calculation
        # ----------------------------------------------------
        lane_density, class_density = self.density.calculate_density(lane_objects)
        print(f"[Engine] Density: lane={dict(lane_density)}, class={dict(class_density)}")

        # ----------------------------------------------------
        # Step 11: Signal Controller
        # ----------------------------------------------------
        signal_plan = self.signal.generate_signal_plan(class_density, priority_decision)
        print(f"[Engine] Signal plan: {signal_plan}")
        print(f"[Engine] Emergency active: {priority_decision.get('active', False)}")

        # ----------------------------------------------------
        # Step 12: Draw Results
        # ----------------------------------------------------
        annotated = self.draw_results(frame, lane_objects, vehicle_detections, emergency_detections)

        # ----------------------------------------------------
        # Save Latest Results
        # ----------------------------------------------------
        self.latest_frame = annotated
        self.latest_tracks = lane_objects
        self.latest_counter = lane_counts

        self.latest_density = {
            "lane_density": dict(lane_density),
            "class_density": dict(class_density)
        }

        self.latest_signals = signal_plan
        self.latest_emergency = priority_decision

        # Average confidence across all tracked objects
        self.latest_confidence = 0.0
        if len(lane_objects) > 0:
            self.latest_confidence = round(
                sum(obj.get("confidence", 0) for obj in lane_objects)
                / len(lane_objects),
                3
            )

        # Current green lane (first lane in signal plan)
        self.current_green_lane = None
        self.current_green_time = 0
        if len(signal_plan) > 0:
            first_lane = next(iter(signal_plan))
            self.current_green_lane = first_lane
            self.current_green_time = signal_plan[first_lane].get("green_time", 0)

        # ----------------------------------------------------
        # Processing Time & FPS
        # ----------------------------------------------------
        end = time.perf_counter()
        self.processing_time = round((end - start) * 1000, 2)
        self.fps = round(1000 / self.processing_time, 2) if self.processing_time > 0 else 0

        print(f"[Engine] Processing time: {self.processing_time}ms | FPS: {self.fps}")

        # ----------------------------------------------------
        # Return Complete Pipeline Result
        # ----------------------------------------------------
        return {
            "frame": annotated,
            "tracks": lane_objects,
            "counter": lane_counts,
            "density": self.latest_density,
            "signals": signal_plan,
            "processing_time": self.processing_time,
            "fps": self.fps,
            "emergency": priority_decision,
            "emergency_summary": self.emergency.get_summary(),
            "vehicle_detections": self.latest_vehicle_detections,
            "emergency_detections": self.latest_emergency_detections,
            "merged_detections": self.latest_merged_detections,
        }

    # ========================================================
    # run_vehicle_model()
    # Runs best.pt on the frame
    # Returns: (detections_list, raw_results)
    # ========================================================

    def run_vehicle_model(self, frame):
        """Run vehicle detection model (best.pt) on the frame."""
        print(f"[Engine] Step 1 - Running vehicle model (best.pt)...")

        tracks, annotated, vehicle_track_result = self.tracker.track_frame(
            frame,
            conf=VEHICLE_CONF,
            return_results=True
        )

        detections = self.tracker.detector.get_detections(vehicle_track_result)
        print(f"[Engine] Vehicle model output: {len(detections)} detections")
        for det in detections:
            print(f"  [Vehicle] {det['class_name']} | Conf: {det['confidence']} | BBox: {det['bbox']}")

        return detections, vehicle_track_result

    # ========================================================
    # run_emergency_model()
    # Runs emergency_best.pt on the frame
    # Returns: list of emergency detections
    # ========================================================

    def run_emergency_model(self, frame):
        """Run emergency detection model (emergency_best.pt) on the frame."""
        print(f"[Engine] Step 2 - Running emergency model (emergency_best.pt)...")

        emergency_raw_results = self.emergency_detector.model.predict(
            source=frame,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU,
            verbose=False
        )

        emergency_list = self.emergency.detect(
            frame,
            self.lane_manager,
            raw_results=emergency_raw_results
        )

        print(f"[Engine] Emergency model output: {len(emergency_list)} detections")
        for emerg in emergency_list:
            locked = emerg.get("_temporal_locked", False)
            print(f"  [Emergency] {emerg['vehicle']} | Conf: {emerg['confidence']} | "
                  f"Lane: {emerg['lane']} | BBox: {emerg['bbox']} | "
                  f"TrackID: {emerg['track_id']} | Locked: {locked}")

        return list(emergency_list)

    # ========================================================
    # merge_detections()
    # Merges vehicle and emergency detections.
    # Emergency ALWAYS overrides vehicle when IoU > threshold.
    # ========================================================

    def merge_detections(self, vehicle_detections, emergency_detections):
        """
        Merge vehicle and emergency detections.
        
        Rules:
        1. Emergency detections ALWAYS override vehicle detections
        2. If IoU > MERGE_IOU_THRESHOLD (0.5), keep ONLY emergency
        3. If emergency confidence > EMERGENCY_FORCE_OVERRIDE_CONF (0.40),
           always choose emergency class even without high IoU
        4. Priority: ambulance > fire_truck > police > normal vehicles
        """
        print(f"[Engine] Step 3 - Merging detections...")
        print(f"  [Merge] Vehicle detections: {len(vehicle_detections)}")
        print(f"  [Merge] Emergency detections: {len(emergency_detections)}")

        # Start with all emergency detections (they have highest priority)
        merged = []
        matched_vehicle_indices = set()

        for emerg in emergency_detections:
            emerg_bbox = emerg.get("bbox")
            if not emerg_bbox:
                continue

            emerg_priority = EMERGENCY_PRIORITY_ORDER.get(emerg["vehicle"], 0)
            best_match_idx = None
            best_iou = 0.0

            for v_idx, veh in enumerate(vehicle_detections):
                if v_idx in matched_vehicle_indices:
                    continue

                veh_bbox = veh.get("bbox")
                if not veh_bbox:
                    continue

                iou = self._compute_iou(emerg_bbox, veh_bbox)

                # Rule: If IoU > threshold, this is the same object
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = v_idx

            # Rule: If IoU > 0.5, emergency overrides vehicle completely
            if best_match_idx is not None and best_iou > MERGE_IOU_THRESHOLD:
                matched_vehicle_indices.add(best_match_idx)
                veh = vehicle_detections[best_match_idx]

                # Create merged object with emergency class
                merged_obj = self._create_emergency_object(emerg, veh)
                print(f"  [Merge] OVERRIDE: Vehicle {veh['class_name']} -> "
                      f"{emerg['vehicle']} (IoU: {best_iou:.2f})")
                merged.append(merged_obj)

            # Rule: If emergency confidence > 0.40, always add as emergency
            # even without matching vehicle (it's a new detection)
            elif emerg["confidence"] >= EMERGENCY_FORCE_OVERRIDE_CONF:
                merged_obj = self._create_emergency_object(emerg, None)
                print(f"  [Merge] FORCE: {emerg['vehicle']} (conf={emerg['confidence']}) "
                      f"added as emergency (no vehicle match)")
                merged.append(merged_obj)

            # Otherwise, add as emergency detection
            else:
                merged_obj = self._create_emergency_object(emerg, None)
                print(f"  [Merge] ADD: {emerg['vehicle']} (conf={emerg['confidence']}) "
                      f"added as emergency detection")
                merged.append(merged_obj)

        # Add remaining vehicle detections that weren't matched
        for v_idx, veh in enumerate(vehicle_detections):
            if v_idx not in matched_vehicle_indices:
                merged_obj = {
                    "track_id": veh.get("track_id", hash(str(veh["bbox"])) % 100000),
                    "class_name": veh["class_name"],
                    "vehicle_type": veh["class_name"],
                    "display_name": veh["class_name"],
                    "label": veh["class_name"],
                    "vehicle": veh["class_name"],
                    "confidence": veh["confidence"],
                    "bbox": veh["bbox"],
                    "center": veh.get("center", (
                        (veh["bbox"][0] + veh["bbox"][2]) // 2,
                        (veh["bbox"][1] + veh["bbox"][3]) // 2
                    )),
                    "source": "vehicle_model",
                    "emergency": False,
                    "is_emergency": False,
                    "priority": "NORMAL",
                }
                merged.append(merged_obj)

        print(f"  [Merge] Total merged objects: {len(merged)}")
        for obj in merged:
            emerg_tag = " [EMERGENCY]" if obj.get("emergency") else ""
            print(f"  [Merged] {obj['class_name']}{emerg_tag} | Conf: {obj['confidence']} | "
                  f"BBox: {obj['bbox']}")

        return merged

    # ========================================================
    # _create_emergency_object()
    # Creates a standardized emergency object from detection data
    # ========================================================

    def _create_emergency_object(self, emerg, vehicle_match=None):
        """Create a standardized emergency vehicle object."""
        bbox = emerg.get("bbox", [0, 0, 0, 0])
        center = emerg.get("center", (
            (bbox[0] + bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2
        ))

        obj = {
            "track_id": emerg.get("track_id", 900000),
            "class_name": emerg["vehicle"],
            "vehicle_type": emerg["vehicle"],
            "display_name": emerg["vehicle"],
            "label": emerg["vehicle"],
            "vehicle": emerg["vehicle"],
            "confidence": emerg["confidence"],
            "bbox": bbox,
            "center": center,
            "lane": emerg.get("lane", "Unknown"),
            "source": "emergency_model",
            "emergency": True,
            "is_emergency": True,
            "emergency_vehicle": emerg["vehicle"],
            "priority": "HIGH",
            "dashboard_class": emerg["vehicle"],
            "dashboard_label": f"🚑 {emerg['vehicle'].upper()}",
            "override_active": True,
            "override_reason": "Emergency Model Override",
            "emergency_confidence": emerg["confidence"],
            "emergency_track_id": emerg.get("track_id", 900000),
            "_temporal_locked": emerg.get("_temporal_locked", False),
            "_bbox_key": emerg.get("_bbox_key", ""),
        }

        # If there's a vehicle match, preserve its track_id for continuity
        if vehicle_match is not None:
            obj["track_id"] = vehicle_match.get("track_id", obj["track_id"])
            obj["_original_vehicle_class"] = vehicle_match.get("class_name", "unknown")

        return obj

    # ========================================================
    # apply_priority()
    # Determines highest priority emergency vehicle
    # ========================================================

    def apply_priority(self, emergency_detections):
        """Apply priority ordering to emergency detections."""
        print(f"[Engine] Step 4 - Applying priority...")

        priority_decision = self.priority.evaluate(emergency_detections)
        print(f"[Engine] Priority decision: {priority_decision}")

        return priority_decision

    # ========================================================
    # remove_duplicates()
    # Removes duplicate detections using class-specific NMS
    # Emergency detections are NEVER removed by vehicle detections
    # ========================================================

    def remove_duplicates(self, detections):
        """
        Remove duplicate detections using class-specific NMS.
        Emergency detections are NEVER removed by vehicle detections.
        """
        print(f"[Engine] Step 5 - Removing duplicates...")

        if len(detections) <= 1:
            return detections

        # Separate emergency and non-emergency
        emergency_objs = [d for d in detections if d.get("emergency")]
        normal_objs = [d for d in detections if not d.get("emergency")]

        # Run NMS on normal objects only (emergency are protected)
        keep_normal = []
        suppressed = set()

        for i, obj_a in enumerate(normal_objs):
            if i in suppressed:
                continue
            keep_normal.append(obj_a)
            bbox_a = obj_a.get("bbox")
            if not bbox_a:
                continue
            for j, obj_b in enumerate(normal_objs):
                if j <= i or j in suppressed:
                    continue
                bbox_b = obj_b.get("bbox")
                if not bbox_b:
                    continue
                iou = self._compute_iou(bbox_a, bbox_b)
                if iou > MERGE_IOU_THRESHOLD:
                    # Keep the one with higher confidence
                    if obj_b["confidence"] > obj_a["confidence"]:
                        # Replace the kept one
                        keep_normal[-1] = obj_b
                    suppressed.add(j)

        # Also check if any normal object overlaps with emergency
        # If so, remove the normal object (emergency wins)
        final_normal = []
        for norm in keep_normal:
            norm_bbox = norm.get("bbox")
            if not norm_bbox:
                final_normal.append(norm)
                continue
            should_remove = False
            for emerg in emergency_objs:
                emerg_bbox = emerg.get("bbox")
                if not emerg_bbox:
                    continue
                iou = self._compute_iou(norm_bbox, emerg_bbox)
                if iou > MERGE_IOU_THRESHOLD:
                    should_remove = True
                    print(f"  [Dedup] Removed {norm['class_name']} (overlaps emergency {emerg['class_name']}, IoU: {iou:.2f})")
                    break
            if not should_remove:
                final_normal.append(norm)

        result = emergency_objs + final_normal
        print(f"  [Dedup] Before: {len(detections)}, After: {len(result)}")

        return result

    # ========================================================
    # update_tracker()
    # Updates tracker state: if a tracked "car" becomes "ambulance"
    # on the same IoU, update the class
    # ========================================================

    def update_tracker(self, lane_objects, emergency_detections):
        """
        Update tracker state.
        If tracker assigned "car" and now emergency model says "ambulance"
        on the same object (IoU > threshold), update the class.
        """
        print(f"[Engine] Step 7 - Updating tracker...")

        # Build a map of emergency bboxes for quick lookup
        for obj in lane_objects:
            if obj.get("emergency"):
                print(f"  [Tracker] Track {obj['track_id']}: {obj['class_name']} [EMERGENCY]")

        # Check for any vehicle-tracked objects that should be emergency
        for obj in lane_objects:
            if obj.get("emergency"):
                continue  # Already emergency

            obj_bbox = obj.get("bbox")
            if not obj_bbox:
                continue

            for emerg in emergency_detections:
                emerg_bbox = emerg.get("bbox")
                if not emerg_bbox:
                    continue

                iou = self._compute_iou(obj_bbox, emerg_bbox)
                if iou > MERGE_IOU_THRESHOLD:
                    # This vehicle should be reclassified as emergency
                    old_class = obj["class_name"]
                    new_class = emerg["vehicle"]
                    print(f"  [Tracker] RE-CLASSIFY: Track {obj['track_id']}: {old_class} -> {new_class} (IoU: {iou:.2f})")

                    # Update all class fields
                    obj["class_name"] = new_class
                    obj["vehicle_type"] = new_class
                    obj["display_name"] = new_class
                    obj["label"] = new_class
                    obj["vehicle"] = new_class
                    obj["emergency"] = True
                    obj["is_emergency"] = True
                    obj["emergency_vehicle"] = new_class
                    obj["priority"] = "HIGH"
                    obj["dashboard_class"] = new_class
                    obj["dashboard_label"] = f"🚑 {new_class.upper()}"
                    obj["override_active"] = True
                    obj["override_reason"] = "Tracker Class Update (Emergency)"
                    obj["emergency_confidence"] = emerg["confidence"]
                    break  # Only one emergency per vehicle

        return lane_objects

    # ========================================================
    # trigger_emergency_mode()
    # If emergency detected, immediately trigger emergency mode
    # ========================================================

    def trigger_emergency_mode(self, priority_decision, lane_objects):
        """Trigger emergency mode if emergency vehicle is detected."""
        print(f"[Engine] Step 8 - Checking emergency mode...")

        if priority_decision.get("active", False):
            vehicle = priority_decision.get("vehicle", "Unknown")
            lane = priority_decision.get("lane", "Unknown")
            print(f"  [EmergencyMode] ACTIVE: {vehicle} in {lane}")

            # Send notification if not already sent for this track_id
            track_id = priority_decision.get("track_id")
            if track_id and track_id not in self._notified_emergency_ids:
                self._notified_emergency_ids.add(track_id)
                print(f"  [Notification] 🚨 EMERGENCY VEHICLE DETECTED: {vehicle} "
                      f"(Track ID: {track_id}, Lane: {lane})")
                print(f"  [Notification] Signal override: GREEN for {lane}, "
                      f"RED for all other lanes")
        else:
            print(f"  [EmergencyMode] INACTIVE (no emergency vehicles)")

    # ========================================================
    # draw_results()
    # Draws all results on the frame
    # Debug mode: blue=vehicle, red=emergency, green=merged
    # ========================================================

    def draw_results(self, frame, lane_objects, vehicle_detections, emergency_detections):
        """Draw detection results on the frame."""
        annotated = frame.copy()

        if DEBUG_MODE:
            # Draw vehicle model detections in BLUE
            for det in vehicle_detections:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    label = f"V:{det['class_name']} {det['confidence']:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Draw emergency model detections in RED
            for det in emergency_detections:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    label = f"E:{det['vehicle']} {det['confidence']:.2f}"
                    cv2.putText(annotated, label, (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Draw final merged output
        for obj in lane_objects:
            bbox = obj.get("bbox")
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            is_emergency = obj.get("emergency", False)

            if is_emergency:
                # Emergency vehicle: RED box, thicker border, blinking
                color = (0, 0, 255)  # Red in BGR
                thickness = 3

                # Blinking effect
                if self._blink_state:
                    # Draw double border for emphasis
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 1)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # Priority label with emoji
                vehicle = obj.get("class_name", "emergency")
                emoji_map = {
                    "ambulance": "🚑",
                    "fire_truck": "🚒",
                    "police": "🚔",
                }
                emoji = emoji_map.get(vehicle, "🚨")
                label = f"{emoji} {vehicle.upper()}"
                conf = obj.get("confidence", 0)

                # Background for label
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Confidence below
                conf_label = f"Conf: {conf:.2f}"
                cv2.putText(annotated, conf_label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                if DEBUG_MODE:
                    # Green outline for merged emergency
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

            else:
                # Normal vehicle: standard box
                color = (0, 255, 0)  # Green in BGR
                thickness = 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                label = f"{obj.get('class_name', 'vehicle')}"
                conf = obj.get("confidence", 0)
                display = f"{label} {conf:.2f}"

                (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 5), (x1 + tw + 5, y1), (0, 255, 0), -1)
                cv2.putText(annotated, display, (x1 + 3, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add frame info
        info_text = f"Frame: {self._frame_counter} | Objects: {len(lane_objects)}"
        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Emergency status overlay
        if self.latest_emergency.get("active", False):
            emerg_text = f"🚨 EMERGENCY: {self.latest_emergency.get('vehicle', '').upper()}"
            cv2.putText(annotated, emerg_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated

    # ========================================================
    # IoU Computation
    # ========================================================

    def _compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bounding boxes."""
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection coordinates
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)

        # Areas of both boxes
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        if area1 + area2 - inter_area <= 0:
            return 0.0

        return inter_area / (area1 + area2 - inter_area)

    # ========================================================
    # Get Latest Counter
    # ========================================================

    def get_counter(self):
        return self.latest_counter

    # ========================================================
    # Get Density
    # ========================================================

    def get_density(self):
        return self.latest_density

    # ========================================================
    # Get Signals
    # ========================================================

    def get_signals(self):
        return self.latest_signals

    # ========================================================
    # Get Tracks
    # ========================================================

    def get_tracks(self):
        return self.latest_tracks

    # ========================================================
    # Get Latest Frame
    # ========================================================

    def get_latest_frame(self):
        return self.latest_frame

    # ========================================================
    # Analytics
    # ========================================================

    def get_statistics(self):

        total = 0
        cars = 0
        buses = 0
        vans = 0
        others = 0
        emergencies = 0

        for lane, lane_data in self.latest_counter.items():

            total += lane_data.get("total", 0)
            cars += lane_data.get("car", 0)
            buses += lane_data.get("bus", 0)
            vans += lane_data.get("van", 0)
            others += lane_data.get("others", 0)

            # Count emergency vehicles separately
            emergencies += lane_data.get("ambulance", 0)
            emergencies += lane_data.get("fire_truck", 0)
            emergencies += lane_data.get("police", 0)

        stats = {
            "total_vehicles": total,
            "cars": cars,
            "bus": buses,
            "van": vans,
            "others": others,
            "emergency_vehicles": emergencies,
        }

        if self.latest_confidence is not None:
            stats["confidence"] = f"{round(self.latest_confidence * 100)}%"

        return stats

    # ========================================================
    # Dashboard Data
    # ========================================================

    def get_dashboard_data(self):

        lane_names = set()
        lane_names.update(self.latest_counter.keys())
        lane_names.update(self.latest_density.get("class_density", {}).keys())
        lane_names.update(self.latest_signals.keys())

        lane_data = {}
        for lane in lane_names:
            lane_data[lane] = {
                "vehicles": self.latest_counter.get(lane, {}).get("total", 0),
                "density": self.latest_density.get("class_density", {}).get(lane, {}).get("total", 0),
                "score": self.latest_signals.get(lane, {}).get("score", 0),
                "green_time": self.latest_signals.get(lane, {}).get("green_time", 0),
                "mode": self.latest_signals.get(lane, {}).get("mode", "NORMAL"),
                "reason": self.latest_signals.get(lane, {}).get("reason", "Adaptive AI Density Control"),
                "priority": self.latest_signals.get(lane, {}).get("priority", "NORMAL"),
            }

        return {
            "counter": self.latest_counter,
            "density": self.latest_density,
            "signals": self.latest_signals,
            "statistics": self.get_statistics(),
            "processing_time": self.processing_time,
            "fps": self.fps,
            "confidence": self.latest_confidence,
            "current_green": {
                "lane": self.current_green_lane,
                "green_time": self.current_green_time,
            },
            "lane_data": lane_data,
            "tracked_vehicles": len(self.latest_tracks),
            "emergency": self.latest_emergency,
            "emergency_summary": self.emergency.get_summary(),
            "vehicle_detections": self.latest_vehicle_detections,
            "emergency_detections": self.latest_emergency_detections,
            "merged_detections": self.latest_merged_detections,
        }

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        self.counter = VehicleCounter()
        self.density = DensityCalculator()
        self.signal = SignalController(emergency_green=EMERGENCY_GREEN)

        self.emergency = EmergencyDetector(
            detector=self.emergency_detector,
            conf=EMERGENCY_CONF,
            iou=EMERGENCY_IOU
        )

        self.processing_time = 0
        self.fps = 0
        self.latest_confidence = 0.0
        self.current_green_lane = None
        self.current_green_time = 0
        self.latest_frame = None
        self.latest_tracks = []
        self.latest_counter = {}
        self.latest_density = {}
        self.latest_signals = {}
        self.latest_emergency = {"active": False}
        self.latest_vehicle_detections = []
        self.latest_emergency_detections = []
        self.latest_merged_detections = []
        self._notified_emergency_ids = set()
        self._frame_counter = 0
        self._blink_state = False

        print("=" * 60)
        print("Traffic Engine Reset")
        print("=" * 60)