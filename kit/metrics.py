# Authors: Gan Huang, Shasha Guan
# Date: 2025-01-16
# Version: 1.0

from math import log2

import numpy as np


def calculate_itr(n_classes: int, accuracy: float,
                  time_seconds: float) -> float:
    """
    Calculate Information Transfer Rate (ITR) in bits per minute.

    Args:
        n_classes: Number of target classes
        accuracy: Classification accuracy in range [0.0, 1.0]
        time_seconds: Total time per trial including stimulus and gaze shift

    Returns:
        float: Information Transfer Rate (bits per minute)
    """
    # Boundary condition check
    if accuracy <= 0 or accuracy < 1 / n_classes or time_seconds <= 0:
        return 0.0

    # Avoid log(0) or log(1) edge cases
    if accuracy >= 1.0:
        accuracy = 0.9999

    # ITR formula
    itr = (log2(n_classes) + accuracy * log2(accuracy) +
           (1 - accuracy) * log2((1 - accuracy) / (n_classes - 1))) * 60 / time_seconds
    
    return max(0.0, itr)