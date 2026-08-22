"""
Prediction utilities for the Bridge Structural Health Monitoring project.

This module loads the trained neural network and scaler,
preprocesses the input sample, and returns the predicted
damage class with its confidence score.
"""

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from config import MODEL_PATH, SCALER_PATH
from preprocessing import extract_features


# ---------------------------------
# Load model and scaler only once
# ---------------------------------

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)


def predict_damage(sample):
    """
    Predict the bridge damage class.

    Parameters
    ----------
    sample : numpy.ndarray

        Raw vibration signal

        Shape:
        (27,6000)

    Returns
    -------
    predicted_class : int

    confidence : float

    probabilities : numpy.ndarray
    """

    # ------------------------------
    # Feature Extraction
    # ------------------------------

    features = extract_features(sample)

    # reshape from (189,) → (1,189)

    features = features.reshape(1, -1)

    # ------------------------------
    # Feature Scaling
    # ------------------------------

    scaled_features = scaler.transform(features)

    # ------------------------------
    # Prediction
    # ------------------------------

    probabilities = model.predict(
        scaled_features,
        verbose=0
    )

    predicted_class = np.argmax(probabilities)

    confidence = np.max(probabilities)

    return (
        int(predicted_class),
        float(confidence),
        probabilities[0]
    )