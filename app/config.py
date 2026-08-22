"""
Configuration file for the Bridge Structural Health Monitoring application.
This file stores all project constants in one place.
"""

from pathlib import Path

# -------------------------------
# Project Directories
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

DATA_DIR = BASE_DIR / "data"

# -------------------------------
# Model Files
# -------------------------------

MODEL_PATH = MODEL_DIR / "bridge_damage_classifier.keras"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

# -------------------------------
# Dataset Specifications
# -------------------------------

EXPECTED_SENSORS = 27

EXPECTED_TIME_STEPS = 6000

FEATURES_PER_SENSOR = 7

TOTAL_FEATURES = EXPECTED_SENSORS * FEATURES_PER_SENSOR