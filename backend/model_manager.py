"""
============================================================
ModelManager
Loads ALL AI models ONCE at FastAPI startup.
Keeps models in GPU memory permanently.
Never reloads models on Start/Stop/Restart.
============================================================
"""

import os
import time
import torch
from ultralytics import YOLO
from backend.config import MODEL_PATH, EMERGENCY_MODEL_PATH


class ModelManager:
    """
    Singleton that loads both YOLO models at startup and keeps them in GPU memory.
    - best.pt: General vehicle detection + tracking
    - emergency_best.pt: Emergency vehicle detection
    """

    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.vehicle_model = None
        self.emergency_model = None
        self.device = None
        self.vehicle_class_names = None
        self.emergency_class_names = None
        self.load_time_ms = 0

    def load_models(self):
        """Load both models ONCE. Called during FastAPI startup."""
        if ModelManager._models_loaded:
            print("[ModelManager] Models already loaded, reusing existing instances.")
            return

        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print(f"[ModelManager] GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("[ModelManager] GPU not available, using CPU")

        start = time.perf_counter()

        # ========================================================
        # Load Vehicle Model (best.pt)
        # ========================================================
        print("=" * 60)
        print("[ModelManager] Loading vehicle model (best.pt)...")
        print("=" * 60)
        v_start = time.perf_counter()
        self.vehicle_model = YOLO(MODEL_PATH)
        self.vehicle_class_names = self.vehicle_model.names
        v_elapsed = (time.perf_counter() - v_start) * 1000
        print(f"[ModelManager] Vehicle model loaded in {v_elapsed:.1f}ms")

        # Move to GPU immediately and keep it there
        if self.device != "cpu":
            self.vehicle_model.to(self.device)
            print(f"[ModelManager] Vehicle model moved to {self.device}")

        # Warm up the model with a dummy inference
        print("[ModelManager] Warming up vehicle model...")
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.vehicle_model.predict(source=dummy, conf=0.5, verbose=False)
        print("[ModelManager] Vehicle model warm-up complete")

        # ========================================================
        # Load Emergency Model (emergency_best.pt)
        # ========================================================
        print("=" * 60)
        print("[ModelManager] Loading emergency model (emergency_best.pt)...")
        print("=" * 60)
        e_start = time.perf_counter()
        self.emergency_model = YOLO(EMERGENCY_MODEL_PATH)
        self.emergency_class_names = self.emergency_model.names
        e_elapsed = (time.perf_counter() - e_start) * 1000
        print(f"[ModelManager] Emergency model loaded in {e_elapsed:.1f}ms")

        if self.device != "cpu":
            self.emergency_model.to(self.device)
            print(f"[ModelManager] Emergency model moved to {self.device}")

        # Warm up emergency model
        print("[ModelManager] Warming up emergency model...")
        _ = self.emergency_model.predict(source=dummy, conf=0.5, verbose=False)
        print("[ModelManager] Emergency model warm-up complete")

        self.load_time_ms = (time.perf_counter() - start) * 1000
        ModelManager._models_loaded = True

        print("=" * 60)
        print(f"[ModelManager] ALL MODELS LOADED in {self.load_time_ms:.1f}ms")
        print(f"[ModelManager] Device: {self.device}")
        print(f"[ModelManager] Vehicle classes: {len(self.vehicle_class_names)}")
        print(f"[ModelManager] Emergency classes: {len(self.emergency_class_names)}")
        print("=" * 60)

    def get_vehicle_model(self):
        """Return the vehicle model instance. Never reloads."""
        if not ModelManager._models_loaded:
            self.load_models()
        return self.vehicle_model

    def get_emergency_model(self):
        """Return the emergency model instance. Never reloads."""
        if not ModelManager._models_loaded:
            self.load_models()
        return self.emergency_model

    def get_device(self):
        return self.device

    @classmethod
    def is_loaded(cls):
        return cls._models_loaded