"""
Preprocessing utilities for the Bridge Structural Health Monitoring project.

This module converts raw vibration signals into the same
189-dimensional feature vector that was used during model training.
"""

import numpy as np

from config import (
    EXPECTED_SENSORS,
    EXPECTED_TIME_STEPS,
)


def validate_input_shape(data):
    """
    Validate the uploaded vibration data.

    Parameters
    ----------
    data : numpy.ndarray
        Raw vibration signal.

    Expected Shape
    --------------
    (27, 6000)

    Returns
    -------
    True

    Raises
    ------
    ValueError
        If the input shape is incorrect.
    """

    expected_shape = (
        EXPECTED_SENSORS,
        EXPECTED_TIME_STEPS,
    )

    if data.shape != expected_shape:
        raise ValueError(
            f"Expected input shape {expected_shape}, "
            f"but received {data.shape}."
        )

    return True


def extract_features(sample):
    """
    Extract seven statistical features from each sensor.

    Features
    --------
    1. Mean
    2. Standard Deviation
    3. RMS
    4. Maximum
    5. Minimum
    6. Peak-to-Peak
    7. Energy

    Parameters
    ----------
    sample : numpy.ndarray

    Shape:
        (27,6000)

    Returns
    -------
    numpy.ndarray

    Shape:
        (189,)
    """

    validate_input_shape(sample)

    features = []

    for sensor in sample:

        mean = np.mean(sensor)

        std = np.std(sensor)

        rms = np.sqrt(np.mean(sensor ** 2))

        maximum = np.max(sensor)

        minimum = np.min(sensor)

        peak_to_peak = np.ptp(sensor)

        energy = np.sum(sensor ** 2)

        features.extend([
            mean,
            std,
            rms,
            maximum,
            minimum,
            peak_to_peak,
            energy
        ])

    return np.array(features, dtype=np.float32)