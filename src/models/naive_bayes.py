"""
Gaussian Naive Bayes classifier implemented from scratch using NumPy.
Author: Person B (@aakritishakya9)
Role: Generative / Probabilistic Model
"""

import numpy as np


class ScratchGaussianNB:
    """
    Gaussian Naive Bayes (GNB) from scratch using NumPy.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability to prevent division by zero.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None

    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.class_priors_ = np.zeros(n_classes)

        # Global variance epsilon for numerical stability
        global_max_var = np.var(X, axis=0).max()
        epsilon = self.var_smoothing * global_max_var

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[idx] = X_c.shape[0] / n_samples
            self.means_[idx, :] = np.mean(X_c, axis=0)
            self.variances_[idx, :] = np.var(X_c, axis=0) + epsilon

        return self

    def _calculate_log_likelihood(self, X):
        """
        Compute log-posterior for each class:
        log P(y=c | x) ~ log P(y=c) + sum_j log P(x_j | y=c)
        where P(x_j | y=c) is N(mu_{c,j}, sigma^2_{c,j}).
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            mean = self.means_[idx]
            var = self.variances_[idx]
            prior = self.class_priors_[idx]

            # Gaussian log-density: -0.5 * ln(2*pi*var) - 0.5 * ((x - mean)^2 / var)
            log_prior = np.log(prior)
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(
                ((X - mean) ** 2) / var, axis=1
            )
            log_posteriors[:, idx] = log_prior + log_likelihood

        return log_posteriors

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        P : array-like of shape (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=np.float64)
        log_posteriors = self._calculate_log_likelihood(X)

        # Log-sum-exp trick for numerical stability
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors - max_log)
        probabilities = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
        return probabilities

    def predict(self, X, threshold=0.5):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        threshold : float or None, default=0.5
            Decision threshold for binary classification (class 1).
            If None or multi-class, argmax over class posteriors is used.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values.
        """
        X = np.asarray(X, dtype=np.float64)
        if len(self.classes_) == 2 and threshold is not None:
            probs = self.predict_proba(X)
            # Default to class 1 as positive if classes are [0, 1]
            return np.where(probs[:, 1] >= threshold, self.classes_[1], self.classes_[0])

        log_posteriors = self._calculate_log_likelihood(X)
        class_indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[class_indices]

    def explain_feature_contributions(self, X):
        """
        Compute per-feature log-likelihood ratio (LLR) for binary classification:
            LLR_j = log( P(x_j | y=1) / P(x_j | y=0) )

        Positive value -> pushes prediction towards High Risk (Late).
        Negative value -> pushes prediction towards Low Risk (On-Time).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        llr : ndarray of shape (n_samples, n_features)
        """
        if len(self.classes_) != 2:
            raise ValueError("Explainability via LLR is supported for binary classification.")

        X = np.asarray(X, dtype=np.float64)
        mu0, mu1 = self.means_[0], self.means_[1]
        var0, var1 = self.variances_[0], self.variances_[1]

        # log P(x_j | 1) - log P(x_j | 0)
        log_density_1 = -0.5 * np.log(2.0 * np.pi * var1) - 0.5 * ((X - mu1) ** 2) / var1
        log_density_0 = -0.5 * np.log(2.0 * np.pi * var0) - 0.5 * ((X - mu0) ** 2) / var0

        return log_density_1 - log_density_0

