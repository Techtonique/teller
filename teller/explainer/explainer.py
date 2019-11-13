import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..utils import (
    deepcopy,
    is_factor,
    memoize,
    numerical_gradient,
    numerical_gradient_jackknife,
    score_regression, 
    score_classification
)


class Explainer(BaseEstimator):
    """class Explainer.
        
       Parameters
       ----------
       obj: object
           fitted object containing a method 'predict'
       n_jobs: int
           number of jobs for parallel computing
    """


    # construct the object -----
    def __init__(self, obj, n_jobs=None):

        self.obj = obj
        self.n_jobs = n_jobs
        self.y_mean_ = None
        self.effects_ = None
        self.residuals_ = None
        self.r_squared_ = None
        self.adj_r_squared_ = None
        self.effects_ = None
        self.ci_ = None


    # fit the object -----    
    def fit(
        self, X, y, X_names, y_name, 
        method="avg", scoring=None, 
        level=95
    ):

        assert method in (
            "avg",
            "ci",
        ), "must have: `method` in ('avg', 'ci')"

        n, p = X.shape

        self.X_names = X_names
        self.y_name = y_name
        self.level = level
        self.scoring = scoring
        
        if is_factor(y):
            self.score_ = score_classification(self.obj, X, y, 
                                              scoring=scoring)
            if scoring is None:
                self.scoring = "accuracy"
        else:
            self.score_ = score_regression(self.obj, X, y, 
                                          scoring=scoring)
            if scoring is None:
                self.scoring = "rmse"
                    

        y_hat = self.obj.predict(X)

        # heterogeneity of effects
        if method == "avg":

            grad = numerical_gradient(
                self.obj.predict, X, n_jobs=self.n_jobs
            )

            res_df = pd.DataFrame(
                data=grad, columns=X_names
            )

            res_df_mean = res_df.mean()
            res_df_std = res_df.std()
            res_df_min = res_df.min()
            res_df_max = res_df.max()
            data = pd.concat(
                [
                    res_df_mean,
                    res_df_std,
                    res_df_min,
                    res_df_max,
                ],
                axis=1,
            )

            df_effects = pd.DataFrame(
                data=data.values,
                columns=["mean", "std", "min", "max"],
                index=X_names,
            )

            # heterogeneity of effects
            self.effects_ = df_effects.sort_values(
                by=["mean"]
            )

        # confidence intervals
        if method == "ci":

            self.ci_ = numerical_gradient_jackknife(
                self.obj.predict,
                X,
                n_jobs=self.n_jobs,
                level=level,
            )

        # any case:
        self.y_mean_ = np.mean(y)
        ss_tot = np.sum((y - self.y_mean_) ** 2)
        ss_reg = np.sum((y_hat - self.y_mean_) ** 2)
        ss_res = np.sum((y - y_hat) ** 2)

        self.residuals_ = y - y_hat
        self.r_squared_ = 1 - ss_res / ss_tot
        self.adj_r_squared_ = 1 - (1 - self.r_squared_) * (
            n - 1
        ) / (n - p - 1)

        return self


    # summary for the object -----
    def summary(self):

        assert (self.ci_ is not None) | (
            self.effects_ is not None
        ), "object not fitted, fit the object first"

        if self.ci_ is not None:

            # (mean_est, se_est,
            # mean_est + qt*se_est, mean_est - qt*se_est,
            # p_values, signif_codes)

            df_mean = pd.Series(
                data=self.ci_[0], index=self.X_names
            )
            df_se = pd.Series(
                data=self.ci_[1], index=self.X_names
            )
            df_ubound = pd.Series(
                data=self.ci_[2], index=self.X_names
            )
            df_lbound = pd.Series(
                data=self.ci_[3], index=self.X_names
            )
            df_pvalue = pd.Series(
                data=self.ci_[4], index=self.X_names
            )
            df_signif = pd.Series(
                data=self.ci_[5], index=self.X_names
            )

            data = pd.concat(
                [
                    df_mean,
                    df_se,
                    df_lbound,
                    df_ubound,
                    df_pvalue,
                    df_signif,
                ],
                axis=1,
            )

            self.ci_summary_ = pd.DataFrame(
                data=data.values,
                columns=[
                    "Estimate",
                    "Std. Error",
                    str(self.level) + "% lbound",
                    str(self.level) + "% ubound",
                    "Pr(>|t|)",
                    "",
                ],
                index=self.X_names,
            ).sort_values(by=["Estimate"])

            
            print("\n")
            print(f"Score ({self.scoring}): \n {np.round(self.score_, 3)}")            
            
            
            print("\n")
            print("Residuals: ")
            self.residuals_dist_ = pd.DataFrame(
                pd.Series(
                    data=np.quantile(
                        self.residuals_,
                        q=[0, 0.25, 0.5, 0.75, 1],
                    ),
                    index=[
                        "Min",
                        "1Q",
                        "Median",
                        "3Q",
                        "Max",
                    ],
                )
            ).transpose()

            print(
                self.residuals_dist_.to_string(index=False)
            )

            print("\n")
            print("Tests on marginal effects (Jackknife): ")
            print(self.ci_summary_)
            print("\n")
            print(
                "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘-’ 1"
            )

            print("\n")
            print(
                f"Multiple R-squared:  {np.round(self.r_squared_, 3)},	Adjusted R-squared:  {np.round(self.adj_r_squared_, 3)}"
            )

        if self.effects_ is not None:
            print("\n")
            print("Heterogeneity of marginal effects: ")
            print(self.effects_)
