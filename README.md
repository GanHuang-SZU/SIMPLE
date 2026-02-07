<!-- 
    Authors: Gan Huang, Shasha Guan
    Date: 2025-01-16
    Version: 1.0
-->

# SIMPLE: Swift, Intuitive, Minimalist Processing for Low-latency, Efficient SSVEP Decoder

A lightweight, high-performance Python framework for ultra-short-window SSVEP (Steady-State Visual Evoked Potential) decoding tailored for low-latency brain-computer interfaces (BCIs).

## Overview

SIMPLE addresses a critical challenge in SSVEP-based BCIs: achieving accurate classification with analysis windows under 300 milliseconds. Unlike traditional methods that degrade sharply when windows are shortened below ~300 ms, SIMPLE leverages phase-locked spatiotemporal dynamics to consistently improve performance across the 50–300 ms regime, advancing toward genuinely responsive and natural BCI interaction.

## Key Features

- **Phase-Locked Time-Shift (PLTS) Augmentation**: Generates phase-matched training segments from brief calibration recordings to stabilize supervised learning with limited data

- **Multi-Algorithm Support**:
  - Linear Discriminant Analysis (LDA)
  - Task-Discriminant Component Analysis (TDCA)
  - Canonical Correlation Analysis (CCA)
  - All with optional filter-bank variants

- **Filter Bank Front-End**: Preserves discriminative multi-harmonic information across 10 frequency sub-bands (5–95 Hz), essential under poor frequency resolution

- **Spatiotemporal Feature Learning**: Single-stage LDA classifier trained on concatenated filter-bank features that captures time-resolved spatial weighting patterns

- **Comprehensive Evaluation**:
  - Leave-One-Block-Out cross-validation
  - Accuracy and Information Transfer Rate (ITR) metrics
  - Benchmarked on Benchmark, BETA, and MP datasets

## Performance

On the Testing sample from the Benchmark dataset with a 100 ms window:
- **91.25%** classification accuracy
- **446.25** bits/min ITR


## Architecture

```
SIMPLE/
├── main.py                      # Main evaluation script
├── kit/
│   ├── preprocessing.py         # Bandpass filtering and filter bank decomposition
│   ├── data_splitter.py         # Data split and augmentation utilities
│   ├── metrics.py               # ITR and performance metric calculations
│   └── algorithms/
│       ├── lda.py               # LDA and Filter-Bank LDA
│       ├── tdca.py              # TDCA and Filter-Bank TDCA
│       └── cca.py               # CCA and Filter-Bank CCA
└── data/
    ├── Freq_Phase.mat           # Stimulus frequency and phase information
    └── sample.mat               # Sample EEG data
```

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy
- scikit-learn

### Setup

```bash
git clone https://github.com/huanggan/SIMPLE.git
cd SIMPLE
pip install -r requirements.txt
```

## Usage

### Running Benchmark Evaluation

```python
python main.py --method PLTS+FB+LDA --window 0.1
```

### Available Methods

- `LDA`: Baseline linear discriminant analysis
- `PLTS+LDA`: LDA with phase-locked time-shift augmentation
- `PLTS+FB+LDA`: Full SIMPLE framework (recommended)
- `TDCA`: Baseline task-discriminant component analysis
- `PLTS+TDCA`: TDCA with augmentation
- `PLTS+FB+TDCA`: TDCA with filter bank
- `CCA`: Canonical correlation analysis (training-free)
- `FB+CCA`: Filter-bank CCA (training-free)

### Option: Window Length

Specify the analysis window in seconds:
```python
python main.py --method PLTS+FB+LDA --window 0.05  # 50 ms
python main.py --method PLTS+FB+LDA --window 0.1   # 100 ms
python main.py --method PLTS+FB+LDA --window 0.3   # 300 ms
```

## Core Modules

### `preprocessing.py`
- **bandpass_filter()**: Applies 2nd-order Butterworth filter (5–95 Hz default)
- **filter_bank()**: Decomposes EEG into 10 frequency sub-bands and concatenates them

### `data_splitter.py`
- **split_standard()**: Leave-One-Block-Out cross-validation split
- **split_augmented()**: PLTS augmentation generating phase-matched segments across trial phases

### `algorithms/`
- **LDA/FilterBankLDA**: Flattens spatiotemporal features for linear classification
- **TDCA/FilterBankTDCA**: Task-discriminant spatial filtering with correlation-based decision
- **CCA/FilterBankCCA**: Training-free canonical correlation with reference signals

### `metrics.py`
- **calculate_itr()**: Computes information transfer rate in bits per minute

## Scientific Background

### Challenges in Short-Window SSVEP Decoding

Traditional SSVEP decoders are optimized for windows ≥ 500 ms, where:
- Frequency resolution is sufficient for reliable spectral analysis
- Phase relationships are stable and well-estimated
- Static spatial filters adequately capture response patterns

Below 300 ms, these assumptions break down:
- **Poor frequency resolution**: Spectral leakage increases and closely-spaced frequency discrimination becomes unreliable
- **Evolving spatial patterns**: Scalp topography and relative source contributions change rapidly
- **Phase sensitivity**: Trial-to-trial phase and timing variability amplifies noise
- **Sample scarcity**: Brief segments limit supervised learning stability

### SIMPLE's Solution

Instead of treating short windows as poorly-conditioned "steady-state" responses, SIMPLE views them as **phase-locked transient impulse responses** to periodic stimulation. This regime shift enables:

1. **PLTS Augmentation**: Extracts many phase-shifted segments from the same short calibration block by leveraging the periodic stimulus structure, effectively multiplying training data
2. **Multi-Harmonic Preservation**: Filter bank captures discriminative information across frequency, mitigating spectral leakage
3. **Spatiotemporal Learning**: Time-concatenated features capture the evolution of spatial patterns during the transient response window

## Experimental Validation

SIMPLE has been validated on:
- **Benchmark Dataset**: 40 frequencies, 6 subjects, diverse analysis windows
- **BETA Dataset**: 40 frequencies, 70 subjects across multiple recording settings
- **MP Dataset**: 12 targets, 20 subjects with multiple presentation paradigms
- **In-House Online Feasibility Study**: Real-time BCI interaction assessment



## Authors

Gan Huang and Shasha Guan


## Affiliation

School of Biomedical Engineering, Medical School, Shenzhen University, Guangdong, China  
Guangdong Provincial Key Laboratory of Biomedical Measurements and Ultrasound Imaging, Guangdong, China



