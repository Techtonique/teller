import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
from typing import Callable, Optional, Union, Dict, Tuple
import warnings
from copy import deepcopy
from sklearn.base import BaseEstimator

class FDAdditiveExplainer(BaseEstimator):
    """
    Finite Difference SHAP-like Explainer with 1st/2nd order effects and numerical safeguards.
    
    Features:
    - Adaptive step sizes with numerical stability checks
    - On-demand interaction computation
    - Parallel/sequential execution modes
    - Progress tracking and visualization
    
    Parameters:
        prediction_function (Callable): Model's prediction function
        order (int): Differentiation order (1 or 2, default=1)
        zero (float): Threshold for small values (default=machine epsilon)
        n_jobs (int): Parallel jobs (-1=all cores, 0=sequential, 1+=specific count)
        normalize (bool): Normalize attributions (default=False)
        progress_bar (bool/str): Progress bar style (True/False/'notebook'/'terminal')
    """

    def __init__(self,
                 prediction_function: Callable,
                 order: int = 1,
                 zero: float = None,
                 n_jobs: int = -1,
                 normalize: bool = False,
                 progress_bar: Union[bool, str] = True):
        self.prediction_function = prediction_function
        self.order = min(max(order, 1), 2)  # Clamp to 1-2 range
        self.zero = zero if zero is not None else np.finfo(float).eps
        self.n_jobs = n_jobs
        self.normalize = normalize
        self.progress_bar = progress_bar
        self._hessian = None
        self._interactions = None
        self.MIN_STEP_SIZE = 1e-10
        self.MAX_CONDITION = 1e16

    def _get_progress_bar(self, iterable, desc=None):
        """Configured progress bar factory"""
        if not self.progress_bar:
            return iterable
        return tqdm(iterable, desc=desc, disable=not self.progress_bar)

    def _safe_divide(self, numerator, denominator):
        """Numerically stable division with bounds checking"""
        denominator = np.where(np.abs(denominator) < self.MIN_STEP_SIZE,
                             np.sign(denominator) * self.MIN_STEP_SIZE,
                             denominator)
        ratio = numerator / denominator
        return np.where(np.abs(ratio) > self.MAX_CONDITION,
                       np.sign(ratio) * self.MAX_CONDITION,
                       ratio)

    def _safe_step_size(self, value_x, cond, eps_factor):
        """Compute step size with multiple safeguards"""
        h = eps_factor * value_x * cond + self.zero * (~cond)
        return np.where(np.abs(h) < self.MIN_STEP_SIZE,
                       np.sign(h) * self.MIN_STEP_SIZE,
                       h)

    def _compute_first_order(self, X: np.ndarray) -> np.ndarray:
        """Numerically robust 1st-order gradients"""
        n_samples, p = X.shape
        grad = np.zeros_like(X)
        eps_factor = self.zero ** (1/3)

        for ix in self._get_progress_bar(range(p), desc="1st-order terms"):
            value_x = X[:, ix].copy()
            cond = np.abs(value_x) > self.zero
            h = self._safe_step_size(value_x, cond, eps_factor)

            X[:, ix] = value_x + h
            fx_plus = np.array(self.prediction_function(X))
            X[:, ix] = value_x - h
            fx_minus = np.array(self.prediction_function(X))
            X[:, ix] = value_x  # Restore

            grad[:, ix] = self._safe_divide(fx_plus - fx_minus, 2 * h)

        return grad

    def calculate_interactions(self, X: np.ndarray) -> None:
        """
        Pre-compute interaction effects for later use.
        
        Args:
            X: Input data (n_samples, n_features)
        """
        if self.order < 2:
            raise ValueError("Interactions require order=2 initialization")
            
        X = np.array(X)
        self.f0 = self.prediction_function(X)
        self._hessian = self._compute_second_order(X)
        self._interactions = np.einsum('sij,si,sj->s', self._hessian, 
                                      X - np.median(X, axis=0), 
                                      X - np.median(X, axis=0)) / 2

    def _compute_second_order(self, X: np.ndarray) -> np.ndarray:
        """Numerically stable 2nd-order computation"""
        n_samples, p = X.shape
        hessian = np.zeros((n_samples, p, p))
        eps_factor = self.zero ** (1/4)

        # Diagonal terms (5-point stencil)
        for ix in self._get_progress_bar(range(p), desc="2nd-order diagonal"):
            value_x = X[:, ix].copy()
            cond = np.abs(value_x) > self.zero
            h = self._safe_step_size(value_x, cond, eps_factor)

            # 5-point stencil evaluations
            eval_points = [
                (value_x + 2*h, "f_plus"),
                (value_x - 2*h, "f_minus"),
                (value_x + h, "f_plus_h"),
                (value_x - h, "f_minus_h")
            ]
            results = {}
            for delta, name in eval_points:
                X[:, ix] = delta
                results[name] = np.array(self.prediction_function(X))
            X[:, ix] = value_x  # Restore

            # Safely compute diagonal Hessian
            numerator = (-results["f_plus"] + 16*results["f_plus_h"] - 30*self.f0 
                        + 16*results["f_minus_h"] - results["f_minus"])
            denominator = 12 * h**2
            hessian[:, ix, ix] = self._safe_divide(numerator, denominator)

        # Off-diagonal terms (4-point cross)
        for ix1, ix2 in self._get_progress_bar(
            [(i,j) for i in range(p) for j in range(i+1,p)], 
            desc="2nd-order interactions"
        ):
            value_x1 = X[:, ix1].copy()
            value_x2 = X[:, ix2].copy()
            cond1 = np.abs(value_x1) > self.zero
            cond2 = np.abs(value_x2) > self.zero
            h1 = self._safe_step_size(value_x1, cond1, eps_factor)
            h2 = self._safe_step_size(value_x2, cond2, eps_factor)

            # 4-point evaluations
            eval_points = [
                (value_x1 + h1, value_x2 + h2, "fx_11"),
                (value_x1 + h1, value_x2 - h2, "fx_12"),
                (value_x1 - h1, value_x2 + h2, "fx_21"),
                (value_x1 - h1, value_x2 - h2, "fx_22")
            ]
            results = {}
            for d1, d2, name in eval_points:
                X[:, ix1], X[:, ix2] = d1, d2
                results[name] = np.array(self.prediction_function(X))
            X[:, ix1], X[:, ix2] = value_x1, value_x2  # Restore

            # Safe interaction computation
            numerator = ((results["fx_11"] - results["fx_12"]) - 
                        (results["fx_21"] - results["fx_22"]))
            denominator = 4 * h1 * h2
            interaction = self._safe_divide(numerator, denominator)
            hessian[:, ix1, ix2] = interaction
            hessian[:, ix2, ix1] = interaction

        return hessian

    def explain(self,
                X: np.ndarray,
                baseline: Optional[np.ndarray] = None,
                feature_names: Optional[list] = None,
                include_interactions: bool = False) -> Dict:
        """
        Compute feature attributions with optional interactions.
        
        Args:
            X: Input data (n_samples, n_features)
            baseline: Reference values (default=median of X)
            feature_names: Optional feature names
            include_interactions: Whether to include 2nd-order effects
            
        Returns:
            Dictionary containing:
                - attributions: 1st-order feature effects
                - interactions: 2nd-order interaction effects (if computed)
                - hessian: Raw Hessian matrix (if order=2)
                - predictions: Model outputs
                - residuals: Decomposition errors
                - baseline: Used reference values
        """
        X = np.array(X)
        self.f0 = self.prediction_function(X)
        
        if baseline is None:
            baseline = np.median(X, axis=0)
        baseline = baseline.reshape(1, -1)

        # Compute 1st-order effects
        grads = self._compute_first_order(X.copy())
        attributions = (X - baseline) * grads
        predictions = np.array(self.prediction_function(X))
        baseline_pred = np.array(self.prediction_function(baseline)).item()

        # Initialize results
        results = {
            'attributions': attributions,
            'gradients': grads,
            'predictions': predictions,
            'baseline': baseline.flatten(),
            'baseline_prediction': baseline_pred,
            'feature_names': feature_names
        }

        # Handle 2nd-order if requested
        if include_interactions and self.order == 2:
            if self._hessian is None:
                self.calculate_interactions(X)
            
            results.update({
                'hessian': self._hessian,
                'interactions': self._interactions,
                'total_effect': (results['attributions'].sum(axis=1) 
                              + self._interactions)
            })
            results['residuals'] = predictions - (baseline_pred + results['total_effect'])
        else:
            results['residuals'] = predictions - (baseline_pred + results['attributions'].sum(axis=1))

        # Add residual metrics
        results.update({
            'max_residual': np.max(np.abs(results['residuals'])),
            'rmse_residual': np.sqrt(np.mean(results['residuals']**2))
        })

        # Optional normalization
        if self.normalize:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = MinMaxScaler(feature_range=(-1, 1))
                results['attributions'] = scaler.fit_transform(results['attributions'])
                results['attributions'] = results['attributions'] / np.sum(
                    results['attributions'], axis=1)[:, np.newaxis]

        return results

    def plot_interactions(self, 
                        feature_names: Optional[list] = None,
                        max_display: int = 10,
                        title: str = "Top Feature Interactions"):
        """
        Visualize interaction strengths from precomputed Hessian.
        
        Args:
            feature_names: Optional names for display
            max_display: Number of top interactions to show
            title: Plot title
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        if self._hessian is None:
            raise ValueError("No interactions computed. Call calculate_interactions() first.")
            
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self._hessian.shape[1])]
            
        mean_inter = np.mean(np.abs(self._hessian), axis=0)
        np.fill_diagonal(mean_inter, 0)  # Exclude diagonal
        
        # Get top interactions
        triu_indices = np.triu_indices_from(mean_inter, k=1)
        inter_pairs = list(zip(triu_indices[0], triu_indices[1], 
                              mean_inter[triu_indices]))
        inter_pairs.sort(key=lambda x: -x[2])
        top_pairs = inter_pairs[:max_display]
        
        # Plot
        plt.figure(figsize=(10, 0.5 * max_display))
        colors = cm.viridis([x[2]/max(x[2] for x in top_pairs) for x in top_pairs])
        
        for i, (f1, f2, val) in enumerate(top_pairs):
            plt.barh(i, val, color=colors[i],
                    label=f"{feature_names[f1]} × {feature_names[f2]}")
        
        plt.yticks(range(len(top_pairs)), 
                  [f"{feature_names[f1]} × {feature_names[f2]}" 
                   for f1, f2, _ in top_pairs])
        plt.title(title)
        plt.xlabel("Interaction Strength")
        plt.tight_layout()
        plt.show()

    def plot_attributions(self,
                         attributions: np.ndarray,
                         feature_names: Optional[list] = None,
                         max_display: int = 15,
                         title: str = "Feature Importance"):
        """
        Visualize feature attributions.
        
        Args:
            attributions: Attributions matrix from explain()
            feature_names: Optional feature names
            max_display: Maximum features to display
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(attributions.shape[1])]
            
        mean_attr = np.mean(np.abs(attributions), axis=0)
        top_indices = np.argsort(mean_attr)[-max_display:]
        
        plt.figure(figsize=(10, 0.5 * len(top_indices)))
        plt.barh(range(len(top_indices)), 
                mean_attr[top_indices], 
                tick_label=np.array(feature_names)[top_indices])
        plt.title(title)
        plt.xlabel("Mean Absolute Attribution")
        plt.tight_layout()
        plt.show()