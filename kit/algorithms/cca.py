# Authors: Gan Huang, Shasha Guan
# Date: 2025-01-16
# Version: 1.0

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CCA(BaseEstimator, ClassifierMixin):
    """
    Canonical Correlation Analysis (CCA) for SSVEP classification.
    """

    def __init__(self, n_harmonics: int = 3):
        """
        Initialize CCA classifier.

        Args:
            n_harmonics: Number of harmonics in reference signal, default 3
        """
        self.n_harmonics = n_harmonics
        self.ref_signals_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, freqs: np.ndarray = None,
            phases: np.ndarray = None, fs: int = 250) -> 'CCA':
        """
        Generate reference signals (CCA is training-free).

        Args:
            X: Training data with shape (n_samples, n_channels, n_points)
            y: Class labels
            freqs: Target frequencies
            phases: Target phases, default None (not used in CCA)
            fs: Sampling frequency (Hz), default 250

        Returns:
            self: Reference signal generator ready for prediction
        """
        self.classes_ = np.unique(y)
        n_points = X.shape[2]

        # Generate reference signals
        t = np.arange(n_points) / fs
        self.ref_signals_ = []
        for freq in freqs:
            ref = []
            for h in range(1, self.n_harmonics + 1):
                ref.append(np.sin(2 * np.pi * h * freq * t))
                ref.append(np.cos(2 * np.pi * h * freq * t))
            self.ref_signals_.append(np.array(ref))
        return self

    def _cca_score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate correlation between single trial and reference signal.

        Args:
            X: Single trial with shape (n_channels, n_points)
            Y: Reference signal with shape (n_refs, n_points)

        Returns:
            float: Maximum canonical correlation value
        """
        N = X.shape[1]

        X_c = X - X.mean(axis=1, keepdims=True)
        Y_c = Y - Y.mean(axis=1, keepdims=True)

        Sxx = X_c @ X_c.T
        Syy = Y_c @ Y_c.T
        Sxy = X_c @ Y_c.T

        # Regularization
        reg = 1e-6
        Sxx += reg * np.eye(Sxx.shape[0])
        Syy += reg * np.eye(Syy.shape[0])

        def inv_sqrt(mat):
            val, vec = np.linalg.eigh(mat)
            val = np.clip(val, 1e-12, None)
            return vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T

        M = inv_sqrt(Sxx) @ Sxy @ inv_sqrt(Syy)
        s = np.linalg.svd(M, compute_uv=False)
        return s[0] if s.size > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Test data with shape (n_samples, n_channels, n_points)

        Returns:
            np.ndarray: Predicted class labels
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for j in range(n_classes):
                scores[i, j] = self._cca_score(X[i], self.ref_signals_[j])

        pred_indices = np.argmax(scores, axis=1)
        return self.classes_[pred_indices]


class FilterBankCCA(CCA):
    """
    Filter Bank CCA: applies CCA independently to each frequency band.
    """

    def __init__(self, n_bands: int = 5, n_harmonics: int = 3,
                 fs: int = 250):
        """
        Initialize Filter Bank CCA.

        Args:
            n_bands: Number of frequency bands, default 5
            n_harmonics: Number of harmonics, default 3
            fs: Sampling frequency (Hz), default 250
        """
        super().__init__(n_harmonics)
        self.n_bands = n_bands
        self.fs = fs
        # Standard filter bank weights: n^(-1.25) + 0.25
        self.weights_ = np.array(
            [(n + 1) ** (-1.25) + 0.25 for n in range(n_bands)]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using weighted fusion from all bands.

        Args:
            X: Test data with shape
               (n_samples, n_channels_total, n_points)

        Returns:
            np.ndarray: Predicted class labels
        """
        n_samples = X.shape[0]
        n_total_ch = X.shape[1]
        n_ch = n_total_ch // self.n_bands
        n_classes = len(self.classes_)

        weighted_scores = np.zeros((n_samples, n_classes))

        for band_i in range(self.n_bands):
            X_band = X[:, band_i * n_ch: (band_i + 1) * n_ch, :]

            band_scores = np.zeros((n_samples, n_classes))
            for i in range(n_samples):
                for j in range(n_classes):
                    band_scores[i, j] = self._cca_score(
                        X_band[i], self.ref_signals_[j]
                    )
            # Weighted sum of correlations
            weighted_scores += self.weights_[band_i] * band_scores

        pred_indices = np.argmax(weighted_scores, axis=1)
        return self.classes_[pred_indices]