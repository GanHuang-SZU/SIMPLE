# Authors: Gan Huang, Shasha Guan
# Date: 2025-01-16
# Version: 1.0

import numpy as np
from scipy.signal import find_peaks


class DataSplitter:
    """
    Prepare train-test data splits for cross-validation evaluation.
    """

    @staticmethod
    def split_standard(eeg: np.ndarray, segment_indices: np.ndarray,
                       test_block_idx: int) -> dict:
        """
        Standard Leave-One-Block-Out cross-validation split.

        Args:
            eeg: EEG data with shape (n_target, ch_num, win_len, n_blocks)
            segment_indices: Time indices for segment extraction
            test_block_idx: Block index to use as test set

        Returns:
            dict: Dictionary containing 'train_data', 'train_y', 'test_data', 'test_y'
        """
        n_target, ch_num, _, n_blocks = eeg.shape
        n_train_blocks = n_blocks - 1
        win_len = len(segment_indices)

        temp = eeg[:, :, segment_indices, :]

        # Training set construction
        train_blocks = np.setdiff1d(np.arange(n_blocks), test_block_idx)
        train_data = temp[:, :, :, train_blocks]
        # Reshape: (n_target, n_blocks, ch, time) -> (n_samples, ch, time)
        train_data = np.transpose(train_data, (0, 3, 1, 2))
        train_data = train_data.reshape(n_target * n_train_blocks, ch_num,
                                        win_len)
        train_y = np.repeat(np.arange(n_target), n_train_blocks)

        # Test set construction
        test_data = temp[:, :, :, test_block_idx]  # (n_target, ch, time)
        test_y = np.arange(n_target)

        return {
            'train_data': train_data,
            'train_y': train_y,
            'test_data': test_data,
            'test_y': test_y
        }

    @staticmethod
    def split_augmented(eeg: np.ndarray, fs: int,
                        segment_indices: np.ndarray,
                        test_block_idx: int, freqs: np.ndarray,
                        phases: np.ndarray) -> dict:
        """
        Data augmentation using PLTS (Phase-Locked Time-Shifted) method.

        Args:
            eeg: EEG data
            fs: Sampling frequency (Hz)
            segment_indices: Time indices for segment extraction
            test_block_idx: Block index for test set
            freqs: Target frequencies
            phases: Target phases

        Returns:
            dict: Dictionary with augmented training data and test data
        """
        n_target, ch_num, total_len, n_blocks = eeg.shape
        n_train_blocks = n_blocks - 1
        t = np.arange(total_len) / fs
        win_len = len(segment_indices)

        train_blocks = np.setdiff1d(np.arange(n_blocks), test_block_idx)
        source_eeg = eeg[:, :, :, train_blocks]

        X_aug_list = []
        Y_aug_list = []

        seg_min = np.min(segment_indices)
        seg_max = np.max(segment_indices)

        for freq_idx in range(n_target):
            # Generate theoretical sine wave for phase alignment
            wave = np.sin(2 * np.pi * freqs[freq_idx] * t +
                          phases[freq_idx])
            peaks, _ = find_peaks(wave)

            # Select valid peak points
            valid_peaks = peaks[peaks < seg_max]
            if len(valid_peaks) == 0:
                continue

            mask = valid_peaks[-1]
            tau1 = mask - seg_min
            tau2 = mask - seg_max

            # Find peaks meeting specified conditions
            peak_candidates = peaks
            peak_times = t[peak_candidates]
            valid_idx = np.where(
                (peak_times - tau1 / fs > 0.18) &
                (peak_times - tau2 / fs < total_len / fs)
            )[0]

            final_peaks = peak_candidates[valid_idx]

            for peak_idx in final_peaks:
                start_idx = peak_idx - tau1
                indices = start_idx + np.arange(win_len)
                # Extract data for specific frequency and time period
                seg_temp = source_eeg[freq_idx, :, indices, :]
                # Transpose to (n_train, time, ch)
                seg_temp = np.transpose(seg_temp, (2, 1, 0))

                X_aug_list.append(seg_temp)
                Y_aug_list.append(np.full(n_train_blocks, freq_idx))

        if X_aug_list:
            train_data = np.concatenate(X_aug_list, axis=0)
            train_y = np.concatenate(Y_aug_list, axis=0)
        else:
            # Fallback to standard split if augmentation fails
            print("Warning: Augmentation failed, using standard split.")
            return DataSplitter.split_standard(eeg, segment_indices,
                                               test_block_idx)

        # Test set (without augmentation)
        temp = eeg[:, :, segment_indices, :]
        test_data = temp[:, :, :, test_block_idx]
        test_y = np.arange(n_target)

        return {
            'train_data': train_data,
            'train_y': train_y,
            'test_data': test_data,
            'test_y': test_y,
            'freqs': freqs,
            'phases': phases
        }