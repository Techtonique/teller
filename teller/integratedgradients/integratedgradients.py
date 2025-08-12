import numpy as np
from typing import Callable, Optional, Union, Dict
from sklearn.base import BaseEstimator

class IntegratedGradientsExplainer(BaseEstimator):
    """
    Integrated Gradients Explainer using scale-dependent finite differences.

    Parameters:
        prediction_function (Callable): Model's prediction function
        baseline_method (str): "mean", "median", or "zero" (default="mean")
        n_steps (int): Number of steps along the path (default=50)
        zero (float): Threshold for small values (default=1e-4)
    """

    def __init__(self,
                 prediction_function: Callable,
                 baseline_method: str = "mean",
                 n_steps: int = 50,
                 zero: float = 1e-4):
        self.prediction_function = prediction_function
        self.baseline_method = baseline_method
        self.n_steps = n_steps
        self.zero = zero

    def _compute_baseline(self, X_train: np.ndarray) -> np.ndarray:
        if self.baseline_method == "mean":
            return np.mean(X_train, axis=0)
        elif self.baseline_method == "median":
            return np.median(X_train, axis=0)
        elif self.baseline_method == "zero":
            return np.zeros(X_train.shape[1])
        else:
            raise ValueError("Unknown baseline_method")

    def _scale_dependent_gradients(self, X: np.ndarray) -> np.ndarray:
        # Scale-dependent finite difference, as in numerical_gradient.py
        n, p = X.shape
        grad = np.zeros_like(X)
        zero = self.zero
        eps_factor = zero ** (1 / 3)

        for ix in range(p):
            value_x = X[:, ix].copy()
            cond = np.abs(value_x) > zero
            h = eps_factor * value_x * cond + zero * (~cond)
            X[:, ix] = value_x + h
            fx_plus = self.prediction_function(X)
            X[:, ix] = value_x - h
            fx_minus = self.prediction_function(X)
            X[:, ix] = value_x  # restore
            grad[:, ix] = (np.asarray(fx_plus) - np.asarray(fx_minus)) / (2 * h)
        return grad

    def explain(self,
                X_train: np.ndarray,
                X_new: np.ndarray,
                feature_names: Optional[list] = None) -> Dict:
        """
        Compute integrated gradients attributions.

        Args:
            X_train: Training data (for baseline computation)
            X_new: New data to explain
            feature_names: Optional feature names

        Returns:
            Dictionary with attributions, integrated gradients, baseline, etc.
        """
        X_train = np.array(X_train)
        X_new = np.array(X_new)
        baseline = self._compute_baseline(X_train)
        n_obs, n_features = X_new.shape
        integrated_grads = np.zeros((n_obs, n_features))

        for alpha in np.linspace(0, 1, self.n_steps):
            X_interp = baseline + (X_new - baseline) * alpha
            grads = self._scale_dependent_gradients(X_interp)
            integrated_grads += grads / self.n_steps

        attributions = (X_new - baseline) * integrated_grads

        baseline_pred = np.array(self.prediction_function(baseline.reshape(1, -1))).reshape(-1)
        predictions = np.array(self.prediction_function(X_new)).reshape(-1)
        reconstructed = baseline_pred + np.sum(attributions, axis=1)
        residuals = predictions - reconstructed

        # Additivity check: prediction ≈ baseline + sum(attributions)
        additivity_rmse = np.sqrt(np.mean(residuals ** 2))
        additivity_max = np.max(np.abs(residuals))

        return {
            'attributions': attributions,
            'integrated_gradients': integrated_grads,
            'baseline': baseline,
            'baseline_prediction': baseline_pred,
            'predictions': predictions,
            'reconstructed': reconstructed,
            'residuals': residuals,
            'additivity_rmse': additivity_rmse,
            'additivity_max': additivity_max,
            'feature_names': feature_names
        }