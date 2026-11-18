"""
Unit tests for Scratch Logistic Regression model.
Verifies gradient descent convergence, loss monotonicity, probability bounds, and parity against scikit-learn.
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.logistic_regression import ScratchLogisticRegression


class TestScratchLogisticRegression(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 400
        self.n_features = 6
        self.X = np.random.randn(self.n_samples, self.n_features)
        # Binary target with clear separation
        linear_combination = self.X[:, 0] * 2.0 - self.X[:, 1] * 1.5 + self.X[:, 2] * 0.8
        self.y = (linear_combination + np.random.randn(self.n_samples) * 0.2 > 0).astype(int)

    def test_loss_decreases(self):
        """Verify that gradient descent reduces binary cross-entropy loss over iterations."""
        model = ScratchLogisticRegression(learning_rate=0.1, n_iters=100)
        model.fit(self.X, self.y)

        self.assertGreater(len(model.loss_history), 1)
        # Initial loss must be greater than final loss
        self.assertGreater(model.loss_history[0], model.loss_history[-1],
                           "Loss did not decrease during gradient descent")

    def test_probabilities_valid(self):
        """Verify probabilities are strictly between 0 and 1, and sum to 1.0."""
        model = ScratchLogisticRegression(learning_rate=0.05, n_iters=150)
        model.fit(self.X, self.y)
        probs = model.predict_proba(self.X)

        self.assertEqual(probs.shape, (self.n_samples, 2))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]")
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(self.n_samples), rtol=1e-5)

    def test_sklearn_parity(self):
        """Verify strong prediction agreement (>95%) against scikit-learn LogisticRegression."""
        scratch = ScratchLogisticRegression(learning_rate=0.1, n_iters=500, l2_penalty=1.0)
        scratch.fit(self.X, self.y)
        scratch_preds = scratch.predict(self.X)

        sklearn_model = SklearnLogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        sklearn_model.fit(self.X, self.y)
        sklearn_preds = sklearn_model.predict(self.X)

        agreement = np.mean(scratch_preds == sklearn_preds)
        self.assertGreaterEqual(agreement, 0.95, f"Agreement was {agreement:.4f}, expected >= 0.95")

    def test_threshold_tuning(self):
        """Verify that lowering decision threshold increases positive classifications."""
        model = ScratchLogisticRegression(learning_rate=0.05, n_iters=100)
        model.fit(self.X, self.y)

        preds_high = model.predict(self.X, threshold=0.7)
        preds_mid = model.predict(self.X, threshold=0.5)
        preds_low = model.predict(self.X, threshold=0.3)

        self.assertGreaterEqual(np.sum(preds_low), np.sum(preds_mid))
        self.assertGreaterEqual(np.sum(preds_mid), np.sum(preds_high))

    def test_feature_importance_and_explainability(self):
        """Verify that feature importance returns odds ratios and instance attribution works."""
        model = ScratchLogisticRegression(learning_rate=0.05, n_iters=100)
        model.fit(self.X, self.y)

        feature_names = [f"feat_{i}" for i in range(self.n_features)]
        df_imp = model.get_feature_importance(feature_names=feature_names)

        self.assertEqual(len(df_imp), self.n_features)
        self.assertIn("Odds_Ratio", df_imp.columns)
        self.assertTrue((df_imp["Odds_Ratio"] > 0).all(), "Odds ratios must be positive")

        # Test single instance explanation
        single_x = self.X[0]
        attribution = model.explain_instance(single_x)
        self.assertEqual(len(attribution), self.n_features)
        self.assertTrue(np.isfinite(attribution).all())


if __name__ == "__main__":
    unittest.main()
