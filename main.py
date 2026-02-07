# Authors: Gan Huang, Shasha Guan
# Date: 2025-01-16
# Version: 1.0

import os
import argparse

import numpy as np
import scipy.io as sio

from kit.preprocessing import bandpass_filter, filter_bank
from kit.data_splitter import DataSplitter
from kit.metrics import calculate_itr
from kit.algorithms.tdca import TDCA, FilterBankTDCA
from kit.algorithms.lda import LDA, FilterBankLDA
from kit.algorithms.cca import CCA, FilterBankCCA


def load_mat_data(data_dir):
    freq_phase = sio.loadmat(os.path.join(data_dir, 'Freq_Phase.mat'))
    freqs = np.squeeze(freq_phase['freqs'])
    phases = np.squeeze(freq_phase['phases'])

    eeg_data = sio.loadmat(os.path.join(data_dir, 'sample.mat'))['eeg']
    return eeg_data, freqs, phases


def run_benchmark(method_name, data_dir='./data', win_len_s=0.5):
    """
    Run benchmark evaluation for specified classification method.

    Args:
        method_name: Name of classification algorithm to use
        data_dir: Path to data directory (default: './data')
        win_len_s: Window length in seconds (default: 0.5)
    """
    # 1. Configuration parameters
    FS = 250
    T_START = 0.5
    WIN_LEN = int(win_len_s * FS)
    SAMPLE_START = int(T_START * FS)
    SEGMENT_INDICES = SAMPLE_START + np.arange(0, WIN_LEN)

    # 2. Load data
    print(f"Loading data from {data_dir}...")
    eeg, freqs, phases = load_mat_data(data_dir)
    n_freq, n_ch, n_sample, n_blocks = eeg.shape

    # 3. Preprocessing
    print("Preprocessing (Filtering)...")
    eeg_bandpass = bandpass_filter(eeg, FS)
    eeg_fb = filter_bank(eeg, FS)
    # Calculate the number of filter banks (for model initialization)
    n_bands = eeg_fb.shape[1] // n_ch

    # 4. Select the model
    # Format: (ModelClass, use_filterbank_data, use_augmentation)
    methods_map = {
        'LDA': (LDA(), False, False),
        'PLTS+LDA': (LDA(), False, True),
        'PLTS+FB+LDA': (FilterBankLDA(n_bands=n_bands), True, True),

        'TDCA': (TDCA(sfreq=FS), False, False),
        'PLTS+TDCA': (TDCA(sfreq=FS), False, True),
        'PLTS+FB+TDCA': (FilterBankTDCA(n_bands=n_bands, sfreq=FS), True, True),

        'CCA': (CCA(), False, False),
        'FB+CCA': (FilterBankCCA(n_bands=n_bands), True, False)
    }

    if method_name not in methods_map:
        raise ValueError(f"Method {method_name} not found. "
                         f"Options: {list(methods_map.keys())}")

    model, use_fb, use_aug = methods_map[method_name]
    input_eeg = eeg_fb if use_fb else eeg_bandpass

    print(f"Running {method_name} (Win: {win_len_s}s)...")

    accuracies = []
    itrs = []

    # 5. Cross-validation loop (Leave-One-Block-Out)
    for block_idx in range(n_blocks):
        # Data preparation
        if use_aug:
            data = DataSplitter.split_augmented(
                input_eeg, FS, SEGMENT_INDICES, block_idx, freqs, phases
            )
        else:
            data = DataSplitter.split_standard(
                input_eeg, SEGMENT_INDICES, block_idx
            )

        X_train, y_train = data['train_data'], data['train_y']
        X_test, y_test = data['test_data'], data['test_y']

        # Train
        model.fit(X_train, y_train, freqs=freqs, phases=phases)

        # Test
        y_pred = model.predict(X_test)

        # Evaluation
        acc = np.mean(y_pred == y_test)
        itr = calculate_itr(n_freq, acc, win_len_s + 0.5)  # 0.5s gaze shift

        accuracies.append(acc)
        itrs.append(itr)
        print(f"  Block {block_idx + 1}/{n_blocks}: "
              f"Acc={acc * 100:.2f}%, ITR={itr:.2f}")

    # 6. Summary of results
    mean_acc = np.mean(accuracies) * 100
    mean_itr = np.mean(itrs)
    print("=" * 40)
    print(f"Result for {method_name}:")
    print(f"Average Accuracy: {mean_acc:.2f}%")
    print(f"Average ITR:      {mean_itr:.2f}")
    print("=" * 40)

    # 7. Save the result
    sio.savemat(f'result_{method_name}_{win_len_s}s.mat', {
        'acc': accuracies,
        'itr': itrs
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SSVEP classification benchmark"
    )
    parser.add_argument(
        '--method',
        type=str,
        default='PLTS+FB+LDA',
        help='Algorithm name'
    )
    parser.add_argument(
        '--win',
        type=float,
        default=0.1,
        help='Window length in seconds'
    )
    args = parser.parse_args()

    run_benchmark(args.method, win_len_s=args.win)