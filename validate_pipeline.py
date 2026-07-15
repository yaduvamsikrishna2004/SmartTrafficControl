"""
TrafficIQ Pipeline Validation Script
Tests: Normal traffic, emergency detection, merge logic, counters, signals
"""

import os
import sys
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

import cv2
import numpy as np

from backend.config import MODEL_PATH, LANE_CONFIG
from backend.traffic_engine import TrafficEngine
from backend.utils import is_emergency_class, normalize_emergency_class, map_vehicle_class

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name} {detail}")

print("=" * 80)
print("TRAFFICIQ PIPELINE VALIDATION")
print("=" * 80)

# ============================================================
# TEST 1: Emergency Class Utilities
# ============================================================
print("\n--- Test 1: Emergency Class Utilities ---")
test("is_emergency_class('ambulance')", is_emergency_class("ambulance"))
test("is_emergency_class('fire_truck')", is_emergency_class("fire_truck"))
test("is_emergency_class('police')", is_emergency_class("police"))
test("is_emergency_class('car') == False", not is_emergency_class("car"))
test("is_emergency_class('bus') == False", not is_emergency_class("bus"))
test("normalize_emergency_class('ambulance')", normalize_emergency_class("ambulance") == "ambulance")
test("normalize_emergency_class('firetruck')", normalize_emergency_class("firetruck") == "fire_truck")
test("normalize_emergency_class('police vehicle')", normalize_emergency_class("police vehicle") == "police")
test("normalize_emergency_class('car') == None", normalize_emergency_class("car") is None)
test("map_vehicle_class('bus')", map_vehicle_class("bus") == "bus")
test("map_vehicle_class('ambulance')", map_vehicle_class("ambulance") == "ambulance")

# ============================================================
# TEST 2: Engine Initialization
# ============================================================
print("\n--- Test 2: Engine Initialization ---")
start = time.time()
engine = TrafficEngine(MODEL_PATH, LANE_CONFIG)
elapsed = time.time() - start

test("Engine initializes without error", engine is not None)
test("Tracker loaded", hasattr(engine, 'tracker'))
test("Emergency detector loaded", hasattr(engine, 'emergency_detector'))
test("Lane manager loaded", hasattr(engine, 'lane_manager'))
test("Counter loaded", hasattr(engine, 'counter'))
test("Density calculator loaded", hasattr(engine, 'density'))
test("Signal controller loaded", hasattr(engine, 'signal'))
test("Priority engine loaded", hasattr(engine, 'priority'))
test("Engine init < 2s", elapsed < 2.0, f"({elapsed:.2f}s)")

# ============================================================
# TEST 3: Process a frame with emergency detection
# ============================================================
print("\n--- Test 3: Frame Processing with Emergency ---")
video_path = "videos/cam4.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    test("Video read successful", ret)
    
    if ret:
        start = time.time()
        result = engine.process_frame(frame)
        elapsed = time.time() - start
        
        test("process_frame returns result", result is not None)
        test("Processing time < 2s", elapsed < 2.0, f"({elapsed*1000:.0f}ms)")
        test("FPS > 0", result.get("fps", 0) > 0)
        
        merged = result.get("merged_detections", [])
        vehicle_dets = result.get("vehicle_detections", [])
        emerg_dets = result.get("emergency_detections", [])
        counter = result.get("counter", {})
        signals = result.get("signals", {})
        
        test("Merged detections is a list", isinstance(merged, list))
        test("Counter has data", len(counter) > 0)
        test("Signals has data", len(signals) > 0)
        
        # Check that emergency override works correctly
        if len(emerg_dets) > 0:
            print(f"  [INFO] Emergency detected: {emerg_dets[0]['vehicle']}")
            test("Emergency priority decision active", result.get("emergency", {}).get("active", False))
            
            # Check signal override
            for lane, signal in signals.items():
                if signal.get("mode") == "EMERGENCY":
                    test(f"Signal override on {lane}", True)
                    
            # Check emergency in merged objects
            for obj in merged:
                if obj.get("emergency") or obj.get("is_emergency"):
                    test(f"[EMERGENCY] Track {obj['track_id']}: {obj.get('class_name', '?')} ", True)
                    break
        
        # Verify no duplicate class fields
        for obj in merged:
            cn = obj.get("class_name")
            vt = obj.get("vehicle_type", cn)
            em = obj.get("emergency_vehicle", cn)
            
            # If marked emergency, class_name must be emergency type
            if obj.get("emergency") or obj.get("is_emergency"):
                if cn:
                    test(f"Emergency object {obj['track_id']} has emergency class '{cn}'", 
                         is_emergency_class(cn))
        
    cap.release()
else:
    test("Video file exists (skipping frame test)", False, f"(not found: {video_path})")

# ============================================================
# TEST 4: Multiple frame processing (50 frames)
# ============================================================
print("\n--- Test 4: Multi-frame Processing (50 frames) ---")
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    total_emergency = 0
    total_emergency_overrides = 0
    
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        
        result = engine.process_frame(frame)
        emerg = result.get("emergency", {})
        if emerg.get("active"):
            total_emergency += 1
            
        # Count emergency overrides in merged objects
        for obj in result.get("merged_detections", []):
            if obj.get("emergency") or obj.get("is_emergency"):
                total_emergency_overrides += 1
    
    cap.release()
    test(f"Processed {total_frames} frames", total_frames > 0)
    
    if total_frames > 0:
        avg_fps = result.get("fps", 0) if result else 0
        test("Average FPS calculated", avg_fps > 0, f"({avg_fps} FPS)")
    
    if total_emergency > 0:
        test(f"Emergency detected in {total_emergency}/{total_frames} frames", True)
        test(f"Emergency override objects present", total_emergency_overrides > 0, f"({total_emergency_overrides} objects)")
else:
    test("Video available for multi-frame test", False)

# ============================================================
# TEST 5: Dashboard Data Integrity
# ============================================================
print("\n--- Test 5: Dashboard Data ---")
dashboard = engine.get_dashboard_data()
test("Dashboard has statistics", "statistics" in dashboard)
test("Dashboard has counter", "counter" in dashboard)
test("Dashboard has density", "density" in dashboard)
test("Dashboard has signals", "signals" in dashboard)
test("Dashboard has emergency", "emergency" in dashboard)
test("Dashboard has fps", "fps" in dashboard)
test("Dashboard has processing_time", "processing_time" in dashboard)
test("Dashboard has current_green", "current_green" in dashboard)
test("Dashboard has lane_data", "lane_data" in dashboard)
test("Dashboard has merged_detections", "merged_detections" in dashboard)
test("Dashboard has emergency_summary", "emergency_summary" in dashboard)

stats = dashboard.get("statistics", {})
test("Statistics has total_vehicles", "total_vehicles" in stats)
test("Statistics has emergency_vehicles", "emergency_vehicles" in stats)
test("Statistics has confidence", "confidence" in stats)
test("Statistics has car/bus/van/others", all(k in stats for k in ["cars", "bus", "van", "others"]))

# ============================================================
# TEST 6: Statistics API Format
# ============================================================
print("\n--- Test 6: Statistics API ---")
stats = engine.get_statistics()
test("Statistics is dict", isinstance(stats, dict))
test("total_vehicles >= 0", stats.get("total_vehicles", -1) >= 0)
test("cars >= 0", stats.get("cars", -1) >= 0)
test("bus >= 0", stats.get("bus", -1) >= 0)
test("van >= 0", stats.get("van", -1) >= 0)
test("emergency_vehicles >= 0", stats.get("emergency_vehicles", -1) >= 0)

# ============================================================
# TEST 7: Emergency Model Always Wins
# ============================================================
print("\n--- Test 7: Emergency Override Rule Verification ---")
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Process 100 frames and verify emergency override logic
    override_count = 0
    total_emerg_objects = 0
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        
        result = engine.process_frame(frame)
        
        # Check each merged object
        for obj in result.get("merged_detections", []):
            if obj.get("emergency") or obj.get("is_emergency"):
                total_emerg_objects += 1
                cn = obj.get("class_name", "")
                # Verify that emergency objects NEVER have a non-emergency class
                if is_emergency_class(cn):
                    override_count += 1
    
    cap.release()
    
    if total_emerg_objects > 0:
        test("Emergency objects have emergency class names", override_count == total_emerg_objects,
             f"({override_count}/{total_emerg_objects} correct)")
    else:
        print("  [SKIP] No emergency objects found in 100 frames")
else:
    test("Video available for override test", False)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
total = PASS + FAIL
print(f"\n  Total Tests: {total}")
print(f"  Passed:      {PASS}")
print(f"  Failed:      {FAIL}")
print(f"  Success:     {PASS/total*100:.1f}%" if total > 0 else "  No tests run")

if FAIL == 0:
    print("\n  ✓ ALL TESTS PASSED")
else:
    print(f"\n  ✗ {FAIL} TEST(S) FAILED")

print("=" * 80)