# Authors: Gan Huang, Shasha Guan
# Date: 2025-01-16
# Version: 1.0

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
from sklearn.base import BaseEstimator, ClassifierMixin


class LDA(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis for EEG classification.
    """

    def __init__(self):
        """Initialize LDA classifier."""
        self.clf = SkLDA(solver='svd')

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LDA':
        """
        Train LDA classifier.

        Args:
            X: Training data with shape (n_samples, n_channels, n_timepoints)
            y: Class labels
            **kwargs: Additional arguments (unused)

        Returns:
            self: Trained LDA instance
        """
        # Reshape: (n_samples, n_ch, n_time) -> (n_samples, n_features)
        X_flat = X.reshape(X.shape[0], -1)
        self.clf.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Test data with same shape as training data

        Returns:
            np.ndarray: Predicted class labels
        """
        X_flat = X.reshape(X.shape[0], -1)
        return self.clf.predict(X_flat)


class FilterBankLDA(BaseEstimator, ClassifierMixin):
    """
    Filter Bank LDA: applies LDA to each frequency band, then combines features.
    """

    def __init__(self, n_bands: int):
        """
        Initialize Filter Bank LDA.

        Args:
            n_bands: Number of frequency bands
        """
        self.n_bands = n_bands
        self.sub_models_ = []
        self.final_model_ = SkLDA(solver='svd')

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'FilterBankLDA':
        """
        Train LDA for each band, then meta-LDA on combined features.

        Args:
            X: Training data with shape (n_samples, n_channels_total, n_time)
            y: Class labels
            **kwargs: Additional arguments (unused)

        Returns:
            self: Trained FilterBankLDA instance
        """
        n_samples = X.shape[0]
        n_total_ch = X.shape[1]
        n_ch = n_total_ch // self.n_bands
        n_classes = len(np.unique(y))
        feat_dim = n_classes - 1  # LDA projection dimension

        self.sub_models_ = []
        features = np.zeros((n_samples, feat_dim * self.n_bands))

        # Train LDA for each band and extract features
        for i in range(self.n_bands):
            X_band = X[:, i * n_ch: (i + 1) * n_ch, :].reshape(
                n_samples, -1
            )
            clf = SkLDA(n_components=feat_dim, solver='svd')
            clf.fit(X_band, y)
            self.sub_models_.append(clf)
            # Extract discriminant features from this band
            features[:, i * feat_dim: (i + 1) * feat_dim] = (
                clf.transform(X_band)
            )

        # Train final classifier on concatenated features
        self.final_model_.fit(features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using combined band features.

        Args:
            X: Test data with same shape as training data

        Returns:
            np.ndarray: Predicted class labels
        """
        n_samples = X.shape[0]
        n_total_ch = X.shape[1]
        n_ch = n_total_ch // self.n_bands
        feat_dim = len(self.sub_models_[0].classes_) - 1

        features = np.zeros((n_samples, feat_dim * self.n_bands))

        for i in range(self.n_bands):
            X_band = X[:, i * n_ch: (i + 1) * n_ch, :].reshape(
                n_samples, -1
            )
            features[:, i * feat_dim: (i + 1) * feat_dim] = (
                self.sub_models_[i].transform(X_band)
            )

        return self.final_model_.predict(features)