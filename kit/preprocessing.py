# Authors: Gan Huang, Shasha Guan
# Date: 2025-01-16
# Version: 1.0

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(eeg: np.ndarray, fs: float, lowcut: float = 5.0,
                    highcut: float = 95.0, order: int = 4) -> np.ndarray:
    """
    Apply band-pass filtering to EEG data.

    Args:
        eeg: Input data with shape (n_frequency, n_channels, n_timepoints, n_blocks)
        fs: Sampling frequency (Hz)
        lowcut: Low-frequency cutoff (Hz), default 5.0
        highcut: High-frequency cutoff (Hz), default 95.0
        order: Filter order, default 4

    Returns:
        np.ndarray: Band-pass filtered EEG data with same shape as input
    """
    nyquist = fs / 2
    b, a = butter(N=order, Wn=[lowcut / nyquist, highcut / nyquist],
                  btype='band')
    filtered_eeg = filtfilt(b, a, eeg, axis=2)
    return filtered_eeg


def filter_bank(eeg: np.ndarray, fs: float,
                bands: list = None) -> np.ndarray:
    """
    Decompose EEG into multiple frequency sub-bands and concatenate them.

    Args:
        eeg: Input EEG data
        fs: Sampling frequency (Hz)
        bands: List of (low, high) frequency pairs for each band.
               Default: 10 bands from 5-95 Hz

    Returns:
        np.ndarray: Concatenated multi-band EEG with shape
                    (n_frequency, n_channels * n_bands, n_timepoints, n_blocks)
    """
    if bands is None:
        bands = [(5.0, 95.0), (12.0, 95.0), (19.0, 95.0), (27.0, 95.0),
                 (35.0, 95.0), (43.0, 95.0), (51.0, 95.0),
                 (59.0, 95.0), (67.0, 95.0), (75.0, 95.0)]

    filtered_list = []
    for (low, high) in bands:
        band_eeg = bandpass_filter(eeg, fs, low, high)
        filtered_list.append(band_eeg)

    return np.concatenate(filtered_list, axis=1)