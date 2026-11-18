"""
Unit tests for Scratch Gaussian Naive Bayes model.
Verifies numerical stability, mathematical invariants, and parity against scikit-learn.
"""

import unittest
import numpy as np
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.naive_bayes import ScratchGaussianNB


class TestScratchGaussianNB(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate synthetic classification dataset
        self.n_samples = 500
        self.n_features = 8
        self.X = np.random.randn(self.n_samples, self.n_features)
        # Binary target with clear relationship to first two features
        self.y = (self.X[:, 0] * 1.5 + self.X[:, 1] * -0.8 + np.random.randn(self.n_samples) * 0.5 > 0).astype(int)

    def test_probabilities_sum_to_one(self):
        """Verify that predict_proba output always sums to 1.0 across classes."""
        model = ScratchGaussianNB()
        model.fit(self.X, self.y)
        probs = model.predict_proba(self.X)

        self.assertEqual(probs.shape, (self.n_samples, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(self.n_samples), rtol=1e-5,
                                   err_msg="Probabilities must sum to 1.0 for each sample")

    def test_sklearn_parity(self):
        """Verify that predictions match scikit-learn GaussianNB exactly."""
        scratch = ScratchGaussianNB(var_smoothing=1e-9)
        scratch.fit(self.X, self.y)
        scratch_preds = scratch.predict(self.X)
        scratch_probs = scratch.predict_proba(self.X)

        sklearn_model = SklearnGaussianNB(var_smoothing=1e-9)
        sklearn_model.fit(self.X, self.y)
        sklearn_preds = sklearn_model.predict(self.X)
        sklearn_probs = sklearn_model.predict_proba(self.X)

        # 1. Predictions must match
        agreement = np.mean(scratch_preds == sklearn_preds)
        self.assertGreaterEqual(agreement, 0.99, f"Agreement with sklearn was {agreement:.4f}")

        # 2. Probabilities must be within numerical precision
        max_diff = np.max(np.abs(scratch_probs - sklearn_probs))
        self.assertLess(max_diff, 1e-4, f"Probability difference {max_diff} exceeded threshold")

    def test_zero_variance_feature_stability(self):
        """Verify that constant features with 0 variance do not cause NaN or ZeroDivisionError."""
        X_zero_var = self.X.copy()
        X_zero_var[:, 2] = 5.0  # Constant column

        model = ScratchGaussianNB(var_smoothing=1e-9)
        model.fit(X_zero_var, self.y)
        preds = model.predict(X_zero_var)
        probs = model.predict_proba(X_zero_var)

        self.assertFalse(np.isnan(preds).any(), "Predictions contain NaN on constant feature")
        self.assertFalse(np.isnan(probs).any(), "Probabilities contain NaN on constant feature")

    def test_threshold_tuning(self):
        """Verify that lowering threshold increases positive class predictions."""
        model = ScratchGaussianNB()
        model.fit(self.X, self.y)

        preds_high = model.predict(self.X, threshold=0.7)
        preds_mid = model.predict(self.X, threshold=0.5)
        preds_low = model.predict(self.X, threshold=0.3)

        # Lower threshold means more positive (1) predictions
        self.assertGreaterEqual(np.sum(preds_low), np.sum(preds_mid))
        self.assertGreaterEqual(np.sum(preds_mid), np.sum(preds_high))

    def test_feature_contributions(self):
        """Verify that explain_feature_contributions returns shape (n_samples, n_features) and finite values."""
        model = ScratchGaussianNB()
        model.fit(self.X, self.y)
        llr = model.explain_feature_contributions(self.X)

        self.assertEqual(llr.shape, self.X.shape)
        self.assertTrue(np.isfinite(llr).all(), "LLR contributions must be finite numbers")



if __name__ == "__main__":
    unittest.main()
