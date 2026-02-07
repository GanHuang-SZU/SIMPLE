# Authors: Gan Huang, Shasha Guan
# Date: 2025-01-16
# Version: 1.0

import numpy as np
from scipy.linalg import eigh, qr
from sklearn.base import BaseEstimator, ClassifierMixin


class TDCA(BaseEstimator, ClassifierMixin):
    """
    Task-Discriminant Component Analysis (TDCA) for SSVEP classification.
    """

    def __init__(self, n_components: int = 8, n_harmonics: int = 3,
                 delay_len: int = 5, sfreq: int = 250):
        """
        Initialize TDCA classifier.

        Args:
            n_components: Number of spatial filters to extract, default 8
            n_harmonics: Number of harmonics in reference signal, default 3
            delay_len: Time-delay length for augmentation, default 5
            sfreq: Sampling frequency (Hz), default 250
        """
        self.n_components = n_components
        self.n_harmonics = n_harmonics
        self.delay_len = delay_len
        self.sfreq = sfreq

        self.w_ = None
        self.templates_ = None
        self.classes_ = None
        self.Q_matrices_ = {}

    def _construct_reference(self, n_points: int, freq: float,
                             phase: float) -> np.ndarray:
        """
        Construct Sine-Cosine reference signals.

        Args:
            n_points: Number of time points
            freq: Target frequency (Hz)
            phase: Target phase (rad)

        Returns:
            np.ndarray: Reference signal with shape (2*n_harmonics, n_points)
        """
        t = np.arange(n_points) / self.sfreq
        Y = []
        for h in range(1, self.n_harmonics + 1):
            Y.append(np.sin(2 * np.pi * h * freq * t + h * phase))
            Y.append(np.cos(2 * np.pi * h * freq * t + h * phase))
        return np.array(Y)

    def _calculate_Q_matrix(self, Y: np.ndarray) -> np.ndarray:
        """
        Calculate Q matrix from QR decomposition.

        Args:
            Y: Reference signal matrix

        Returns:
            np.ndarray: Orthogonal matrix from QR decomposition
        """
        Q, _ = qr(Y.T, mode='economic')
        return Q

    def _augment_data(self, X: np.ndarray) -> np.ndarray:
        """
        Construct time-delayed augmented EEG data.

        Args:
            X: Input EEG with shape (n_trials, n_channels, n_points)

        Returns:
            np.ndarray: Augmented EEG with shape
                        (n_trials, n_channels * (delay_len + 1), n_points)
        """
        n_trials, n_channels, n_points = X.shape
        X_aug = np.zeros(
            (n_trials, n_channels * (self.delay_len + 1), n_points),
            dtype=X.dtype
        )

        for i in range(self.delay_len + 1):
            c_start = i * n_channels
            c_end = (i + 1) * n_channels
            if i == 0:
                X_aug[:, c_start:c_end, :] = X
            else:
                X_aug[:, c_start:c_end, :-i] = X[:, :, i:]
        return X_aug

    def fit(self, X: np.ndarray, y: np.ndarray, freqs: np.ndarray = None,
            phases: np.ndarray = None, **kwargs) -> 'TDCA':
        """
        Train the TDCA spatial filter and templates.

        Args:
            X: Training data with shape (n_samples, n_channels, n_points)
            y: Class labels
            freqs: Target frequencies, default None
            phases: Target phases, default None

        Returns:
            self: Trained TDCA instance
        """
        self.classes_ = np.unique(y)
        n_trials, n_channels, n_points = X.shape

        X_tilde = self._augment_data(X)
        n_aug_ch = X_tilde.shape[1]

        self.Q_matrices_ = {}
        for k_idx, k in enumerate(self.classes_):
            freq = freqs[k_idx] if freqs is not None else 0
            phase = phases[k_idx] if phases is not None else 0
            Y = self._construct_reference(n_points, freq, phase)
            self.Q_matrices_[k] = self._calculate_Q_matrix(Y)

        class_means = {}
        class_counts = {}
        sum_all = np.zeros((n_aug_ch, 2 * n_points))
        n_total = 0

        Sb = np.zeros((n_aug_ch, n_aug_ch))
        Sw = np.zeros((n_aug_ch, n_aug_ch))

        for k in self.classes_:
            idxs = np.where(y == k)[0]
            Nk = len(idxs)
            class_counts[k] = Nk

            X_k_tilde = X_tilde[idxs]
            Q_k = self.Q_matrices_[k]

            X_k_proj = (X_k_tilde @ Q_k) @ Q_k.T
            X_k_a = np.concatenate([X_k_tilde, X_k_proj], axis=2)

            mean_k = np.mean(X_k_a, axis=0)
            class_means[k] = mean_k

            sum_all += np.sum(X_k_a, axis=0)
            n_total += Nk

            X_k_centered = X_k_a - mean_k
            Sw += np.einsum('nij,nkj->ik', X_k_centered, X_k_centered)

        global_mean = sum_all / n_total

        for k in self.classes_:
            Nk = class_counts[k]
            diff = class_means[k] - global_mean
            Sb += Nk * (diff @ diff.T)

        Sw += 1e-6 * np.eye(n_aug_ch) * np.trace(Sw) / n_aug_ch

        evals, evecs = eigh(Sb, Sw)
        self.w_ = evecs[:, -self.n_components:]

        # Extract and normalize templates
        self.templates_ = []
        for k in self.classes_:
            temp = self.w_.T @ class_means[k]
            temp_flat = temp.flatten()
            temp_flat = temp_flat - np.mean(temp_flat)
            norm = np.linalg.norm(temp_flat)
            if norm > 0:
                temp_flat /= norm
            self.templates_.append(temp_flat)

        return self

    def get_correlations(self, X: np.ndarray) -> np.ndarray:
        """
        Compute correlation with templates for each class.

        Args:
            X: Test data with shape (n_trials, n_channels, n_points)

        Returns:
            np.ndarray: Correlation scores with shape (n_trials, n_classes)
        """
        n_trials = X.shape[0]
        n_classes = len(self.classes_)

        X_tilde = self._augment_data(X)
        corrs = np.zeros((n_trials, n_classes))

        for k_idx, k in enumerate(self.classes_):
            Q_k = self.Q_matrices_[k]

            X_proj = (X_tilde @ Q_k) @ Q_k.T
            X_a = np.concatenate([X_tilde, X_proj], axis=2)

            feat = np.tensordot(X_a, self.w_, axes=([1], [0]))
            feat = np.transpose(feat, (0, 2, 1))

            feat_flat = feat.reshape(n_trials, -1)

            feat_mean = np.mean(feat_flat, axis=1, keepdims=True)
            feat_centered = feat_flat - feat_mean

            feat_norm = np.linalg.norm(feat_centered, axis=1)
            feat_norm[feat_norm == 0] = 1.0
            feat_centered /= feat_norm[:, None]

            template_flat = self.templates_[k_idx]
            corrs[:, k_idx] = feat_centered @ template_flat

        return corrs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Test data

        Returns:
            np.ndarray: Predicted class labels
        """
        corrs = self.get_correlations(X)
        return self.classes_[np.argmax(corrs, axis=1)]


class FilterBankTDCA(BaseEstimator, ClassifierMixin):
    """
    Filter Bank TDCA: applies TDCA independently to each frequency band.
    """

    def __init__(self, n_bands: int = 5, n_components: int = 8,
                 n_harmonics: int = 3, delay_len: int = 5,
                 sfreq: int = 250):
        """
        Initialize Filter Bank TDCA.

        Args:
            n_bands: Number of frequency bands, default 5
            n_components: Number of spatial filters per band, default 8
            n_harmonics: Number of harmonics, default 3
            delay_len: Time-delay length, default 5
            sfreq: Sampling frequency (Hz), default 250
        """
        self.n_bands = n_bands
        self.n_components = n_components
        self.n_harmonics = n_harmonics
        self.delay_len = delay_len
        self.sfreq = sfreq
        self.models_ = []
        # Standard filter bank weights: n^(-1.25) + 0.25
        self.weights_ = [(n + 1) ** (-1.25) + 0.25 for n in range(n_bands)]

    def fit(self, X: np.ndarray, y: np.ndarray, freqs: np.ndarray = None,
            phases: np.ndarray = None, **kwargs) -> 'FilterBankTDCA':
        """
        Train TDCA for each frequency band.

        Args:
            X: Training data
            y: Class labels
            freqs: Target frequencies
            phases: Target phases

        Returns:
            self: Trained FilterBankTDCA instance
        """
        n_ch = X.shape[1] // self.n_bands
        self.models_ = []
        for i in range(self.n_bands):
            X_band = X[:, i * n_ch:(i + 1) * n_ch, :]
            tdca = TDCA(
                n_components=self.n_components,
                n_harmonics=self.n_harmonics,
                delay_len=self.delay_len,
                sfreq=self.sfreq
            )
            tdca.fit(X_band, y, freqs, phases)
            self.models_.append(tdca)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using weighted fusion from all bands.

        Args:
            X: Test data

        Returns:
            np.ndarray: Predicted class labels
        """
        n_trials = X.shape[0]
        n_classes = len(self.models_[0].classes_)
        n_ch = X.shape[1] // self.n_bands
        rho_total = np.zeros((n_trials, n_classes))

        for i in range(self.n_bands):
            X_band = X[:, i * n_ch:(i + 1) * n_ch, :]
            rho_i = self.models_[i].get_correlations(X_band)
            # Weighted sum of correlations
            rho_total += self.weights_[i] * (rho_i ** 2) * np.sign(rho_i)

        predictions = np.argmax(rho_total, axis=1)
        return self.models_[0].classes_[predictions]