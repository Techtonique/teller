import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from tqdm import tqdm
from ..utils import (
    is_factor,
    numerical_gradient_conformal,
    numerical_interactions,
    numerical_interactions_jackknife,
    numerical_interactions_gaussian,
    Progbar,
    score_regression,
    score_classification,
    sensitivity_confidence_intervals,
)


class ConformalExplainer(BaseEstimator):
    """Class ConformalExplainer: effects of features on the response.

    Attributes:

        obj: an object;
            fitted object containing methods `fit` and `predict`

        n_jobs: an integer;
            number of jobs for parallel computing

        y_class: an integer;
            class whose probability has to be explained (for classification only, default is 0)

    """

    def __init__(self, obj, n_jobs=None, y_class=0):
        self.obj = obj
        self.n_jobs = n_jobs
        self.y_class = y_class
        self.summary_ = None 
        self.col_inters = None 

    def fit(
        self,
        X,
        X_names=None,
        level=95,
        col_inters=None,
    ):
        """Fit the explainer's attribute `obj` to training data (X, y).

        Args:
            X: array-like, shape = [n_samples, n_features];
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            X_names: {array-like}, shape = [n_features, ];
                Column names (strings) for training vectors (default
                is None, and not used if X is a data frame).

            level: confidence level for intervals (default: 95)

            col_inters: str; Name of column for computing interactions.
        """
        n, p = X.shape

        self.col_inters = col_inters

        if isinstance(X, pd.DataFrame):
            self.X_names = X.columns
        else:
            assert (
                X_names is not None
            ), "'X' is a numpy array, 'X_names' must be provided"
            self.X_names = X_names

        # Check if estimator is a classifier using hasattr
        is_classifier = hasattr(self.obj, "predict_proba")

        if is_classifier:  # classification ---
            self.type_fit = "classification" 
            self.summary_ = sensitivity_confidence_intervals(
                model=self.obj, 
                X_test=X,
                confidence_level=level/100
            )

        else:  # regression ---
            self.type_fit = "regression"
            self.summary_ = sensitivity_confidence_intervals(
                model=self.obj, 
                X_test=X,
                confidence_level=level/100
            )

        # interactions
        if self.col_inters is not None:
            raise NotImplementedError

        return self

    def summary(self):
        """Summarise results in a user-friendly format.

        Args:
            None

        Returns:
            Formatted summary of the sensitivity analysis results
        """
        if self.summary_ is None:
            return "Model has not been fitted yet. Call fit() first."

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'Mean Estimate': self.summary_.mean,
            'Median Estimate': self.summary_.mean,
            '.95 Lower Bound': self.summary_.lower,
            '.95 Upper Bound': self.summary_.upper,
            'Signif.': self.summary_.signif_codes,
            'PI length': self.summary_.pi_length,
        }, index=self.X_names)

        # Sort by absolute value of estimate
        results_df = results_df.reindex(
            results_df['Mean Estimate'].sort_values(ascending=False).index
        )

        # Format the output
        print("\nSensitivity Analysis Results:")
        print("============================")
        print(f"\nModel type: {self.type_fit}")
        print("\nFeature Effects:")
        print(results_df.round(4))

        # Don't return the DataFrame to avoid duplication
        #return

    def plot(self):
        """Plot confidence intervals for each feature.

        Args:
            None
        """
        assert self.summary_ is not None, "Call method 'fit' before plotting"

        # Create DataFrame with results
        results_df = pd.DataFrame({
            'Mean Estimate': self.summary_.mean,
            '.95 Lower Bound': self.summary_.lower,
            '.95 Upper Bound': self.summary_.upper,
        }, index=self.X_names)

        # Filter features with non-zero mean estimates
        results_df = results_df[results_df['Mean Estimate'] != 0]

        # Sort by mean estimate
        results_df = results_df.sort_values(by="Mean Estimate", ascending=False)

        # Plot confidence intervals
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x=results_df['Mean Estimate'],
            y=results_df.index,
            xerr=[
                results_df['Mean Estimate'] - results_df['.95 Lower Bound'],
                results_df['.95 Upper Bound'] - results_df['Mean Estimate']
            ],
            fmt='o',
            capsize=5,
            color='blue',
            ecolor='gray',
        )
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.title("Confidence Intervals for Feature Effects")
        plt.xlabel("Mean Estimate")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()