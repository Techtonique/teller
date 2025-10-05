import numpy as np
from typing import Callable, Optional, Union, Dict
from sklearn.base import BaseEstimator


class IntegratedGradientsExplainer(BaseEstimator):
    """
    Integrated Gradients Explainer using scale-dependent finite differences.

    This class estimates feature attributions for a model's predictions by integrating
    gradients along the straight-line path from a baseline to each input, using
    scale-dependent finite differences for gradient estimation.

    Parameters
    ----------
    prediction_function : Callable
        The model's prediction function. Should accept a 2D numpy array and return a 1D array of predictions.
    baseline_method : str, default="mean"
        Method for computing the baseline/reference input. One of {"mean", "median", "zero"}.
    n_steps : int, default=50
        Number of steps for the integration path from baseline to input.
    zero : float, default=1e-4
        Threshold for small values in scale-dependent finite difference calculation.

    Examples
    --------
    >>> explainer = IntegratedGradientsExplainer(model.predict, baseline_method="mean", n_steps=50)
    >>> results = explainer.explain(X_train, X_test, feature_names=list(X.columns))
    >>> print(results["attributions"])
    """

    def __init__(
        self,
        prediction_function: Callable,
        baseline_method: str = "mean",
        n_steps: int = 50,
        zero: float = 1e-4,
    ):
        self.prediction_function = prediction_function
        self.baseline_method = baseline_method
        self.n_steps = n_steps
        self.zero = zero

    def _compute_baseline(self, X_train: np.ndarray) -> np.ndarray:
        """
        Compute the baseline/reference input for integrated gradients.

        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (n_samples, n_features).

        Returns
        -------
        baseline : np.ndarray
            Baseline input of shape (n_features,).
        """
        if self.baseline_method == "mean":
            return np.mean(X_train, axis=0)
        elif self.baseline_method == "median":
            return np.median(X_train, axis=0)
        elif self.baseline_method == "zero":
            return np.zeros(X_train.shape[1])
        else:
            raise ValueError("Unknown baseline_method")

    def _scale_dependent_gradients(self, X: np.ndarray) -> np.ndarray:
        """
        Compute local gradients using scale-dependent finite differences.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        grad : np.ndarray
            Estimated gradients of shape (n_samples, n_features).
        """
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

    def explain(
        self,
        X_train: np.ndarray,
        X_new: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict:
        """
        Compute integrated gradients feature attributions for new data.

        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (n_samples, n_features), used to compute the baseline.
        X_new : np.ndarray
            New data of shape (n_samples, n_features) for which to compute attributions.
        feature_names : list, optional
            List of feature names for output and interpretation.

        Returns
        -------
        results : dict
            Dictionary containing:
                attributions : np.ndarray
                    Integrated gradients attributions for each sample, shape (n_samples, n_features).
                integrated_gradients : np.ndarray
                    Average gradients along the integration path, shape (n_samples, n_features).
                baseline : np.ndarray
                    Baseline/reference input used, shape (n_features,).
                baseline_prediction : np.ndarray
                    Model prediction at the baseline input, shape (1,).
                predictions : np.ndarray
                    Model predictions for X_new, shape (n_samples,).
                reconstructed : np.ndarray
                    Baseline prediction plus sum of attributions, shape (n_samples,).
                residuals : np.ndarray
                    Difference between model prediction and reconstructed value, shape (n_samples,).
                additivity_rmse : float
                    Root mean squared error of residuals (additivity check).
                additivity_max : float
                    Maximum absolute residual (additivity check).
                feature_names : list or None
                    Feature names if provided.

        Examples
        --------
        >>> explainer = IntegratedGradientsExplainer(model.predict)
        >>> results = explainer.explain(X_train, X_test)
        >>> print(results["attributions"])
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

        baseline_pred = np.array(
            self.prediction_function(baseline.reshape(1, -1))
        ).reshape(-1)
        predictions = np.array(self.prediction_function(X_new)).reshape(-1)
        reconstructed = baseline_pred + np.sum(attributions, axis=1)
        residuals = predictions - reconstructed

        # Additivity check: prediction ≈ baseline + sum(attributions)
        additivity_rmse = np.sqrt(np.mean(residuals**2))
        additivity_max = np.max(np.abs(residuals))

        return {
            "attributions": attributions,
            "integrated_gradients": integrated_grads,
            "baseline": baseline,
            "baseline_prediction": baseline_pred,
            "predictions": predictions,
            "reconstructed": reconstructed,
            "residuals": residuals,
            "additivity_rmse": additivity_rmse,
            "additivity_max": additivity_max,
            "feature_names": feature_names,
        }
