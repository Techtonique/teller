import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..utils import (
    deepcopy,
    is_factor,
    memoize,
    numerical_gradient,
    numerical_gradient_jackknife,
    numerical_interactions,
    numerical_interactions_jackknife,
    score_regression, 
    score_classification
)
from scipy.special import expit


class Explainer(BaseEstimator):
    """class Explainer.
        
       Parameters
       ----------
       obj: object
           fitted object containing a method 'predict'
       n_jobs: int
           number of jobs for parallel computing
       y_class: int
           class whose probability has to be explained (for classification only)
       normalize:  boolean
           whether the effects must be normalized or not
    """


    # construct the object -----
    
    def __init__(self, obj, n_jobs=None, 
                 y_class=0, normalize=False):

        self.obj = obj
        self.n_jobs = n_jobs
        self.y_mean_ = None
        self.effects_ = None
        self.residuals_ = None
        self.r_squared_ = None
        self.adj_r_squared_ = None
        self.effects_ = None
        self.ci_ = None
        self.type_fit = None
        self.y_class = y_class # classification only
        self.normalize = normalize


    # fit the object -----  
    
    def fit(
        self, X, y, 
        X_names, y_name, 
        method="avg", 
        scoring=None,
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
        
        
        if is_factor(y): # classification ---
            
            self.n_classes = len(np.unique(y))
            
            assert self.y_class <= self.n_classes,\
            "self.y_class must be <= number of classes"            
            
            assert hasattr(self.obj, 'predict_proba'),\
            "`self.obj` must be a classifier and have a method `predict_proba`"

            self.type_fit = "classification"
            
            self.score_ = score_classification(self.obj, X, y, 
                                              scoring=scoring)
            if scoring is None:
                self.scoring = "accuracy"                        
            
            def predict_proba(x): return self.obj.predict_proba(x)[:, self.y_class]
            
            y_hat = predict_proba(X)
            
            # heterogeneity of effects
            if method == "avg":                                
    
                self.grad = numerical_gradient(
                    predict_proba, X, 
                    normalize=self.normalize,
                    n_jobs=self.n_jobs
                )
            
            # confidence intervals
            if method == "ci":

                self.ci_ = numerical_gradient_jackknife(predict_proba,
                X, normalize=self.normalize, 
                n_jobs=self.n_jobs, level=level)
                
                
        else: # is_factor(y) == False # regression ---
            
            self.type_fit = "regression"
            
            self.score_ = score_regression(self.obj, X, y, 
                                          scoring=scoring)
            if scoring is None:
                self.scoring = "rmse"
                    
            y_hat = self.obj.predict(X)
    
            # heterogeneity of effects
            if method == "avg":
    
                self.grad = numerical_gradient(
                    self.obj.predict, X, 
                    normalize=self.normalize,
                    n_jobs=self.n_jobs
                )
            
            # confidence intervals
            if method == "ci":

                self.ci_ = numerical_gradient_jackknife(
                self.obj.predict,
                X, normalize=self.normalize,
                n_jobs=self.n_jobs,
                level=level,
                )
            
            self.y_mean_ = np.mean(y)
            ss_tot = np.sum((y - self.y_mean_) ** 2)
            ss_reg = np.sum((y_hat - self.y_mean_) ** 2)
            ss_res = np.sum((y - y_hat) ** 2)
    
            self.residuals_ = y - y_hat
            self.r_squared_ = 1 - ss_res / ss_tot
            self.adj_r_squared_ = 1 - (1 - self.r_squared_) * (
                n - 1
            ) / (n - p - 1)
        
        
        # classification and regression ---
        res_df = pd.DataFrame(
            data=self.grad, columns=X_names
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
            by=["mean"], ascending=False
        )                        

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
            ).sort_values(by=["Estimate"], ascending=False)

            
            print("\n")
            print(f"Score ({self.scoring}): \n {np.round(self.score_, 3)}")            
            
            
            if self.type_fit == "regression":
                
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
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.ci_summary_)
            print("\n")
            print(
                "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘-’ 1"
            )
            
            if self.type_fit == "regression":
                
                print("\n")
                print(
                    f"Multiple R-squared:  {np.round(self.r_squared_, 3)},	Adjusted R-squared:  {np.round(self.adj_r_squared_, 3)}"
                )

        if self.effects_ is not None:
            print("\n")
            print("Heterogeneity of marginal effects: ")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(self.effects_)
            print("\n")
