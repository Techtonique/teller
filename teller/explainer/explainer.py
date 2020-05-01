import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..utils import (
    is_factor,
    numerical_gradient,
    numerical_gradient_jackknife,
    numerical_gradient_gaussian,
    numerical_interactions,
    numerical_interactions_jackknife,
    numerical_interactions_gaussian,
    Progbar,
    score_regression,
    score_classification,
)


class Explainer(BaseEstimator):
    """Class Explainer for: effects of features on the response.
        
       Parameters
       ----------
       obj: object
           fitted object containing methods `fit` and `predict`
       n_jobs: int
           number of jobs for parallel computing
       y_class: int
           class whose probability has to be explained (for classification only)
       normalize:  boolean
           whether the features must be normalized or not (changes the effects)
    """


    def __init__(self, obj, n_jobs=None, y_class=0, normalize=False):

        self.obj = obj
        self.n_jobs = n_jobs
        self.y_mean_ = None
        self.effects_ = None
        self.residuals_ = None
        self.r_squared_ = None
        self.adj_r_squared_ = None
        self.effects_ = None
        self.ci_ = None
        self.ci_inters_ = {}
        self.type_fit = None
        self.y_class = y_class  # classification only
        self.normalize = normalize


    def fit(
        self,
        X,
        y,
        X_names,
        y_name,
        method="avg",
        type_ci="jackknife",
        scoring=None,
        level=95,
        col_inters=None,
    ):
        """Fit the explainer's attribute `obj` to training data (X, y).           
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        y: {array-like}, shape = [n_samples, ]
            Target values.

        X_names: {array-like}, shape = [n_features, ]
             Column names (strings) for training vectors.
        
        y_names: str
               Column name (string) for vector of target values. 

        method: str
                Type of summary requested for effects. Either `avg` 
                (for average effects), `inters` (for interactions) 
                or `ci` (for effects including confidence intervals
                around them). 

        type_ci: str
                Type of resampling for `method == 'ci'` (confidence 
                intervals around effects). Either `jackknife` 
                bootsrapping or `gaussian` (gaussian white noise with 
                standard deviation equal to `0.01` applied to the 
                features). 

        scoring: str
                measure of errors must be in ("explained_variance", 
                "neg_mean_absolute_error", "neg_mean_squared_error", 
                "neg_mean_squared_log_error", "neg_median_absolute_error", 
                "r2", "rmse") (default: "rmse")

        level: int
            Level of confidence required for `method == 'ci'` (in %)

        col_inters: str
            Name of column for computing interactions

                   
        Returns
        -------
        self: object
        """
        assert method in (
            "avg",
            "ci",
            "inters",
        ), "must have: `method` in ('avg', 'ci', 'inters')"

        n, p = X.shape

        self.X_names = X_names
        self.y_name = y_name
        self.level = level
        self.scoring = scoring
        self.method = method

        if is_factor(y):  # classification ---

            self.n_classes = len(np.unique(y))

            assert (
                self.y_class <= self.n_classes
            ), "self.y_class must be <= number of classes"

            assert hasattr(
                self.obj, "predict_proba"
            ), "`self.obj` must be a classifier and have a method `predict_proba`"

            self.type_fit = "classification"

            self.score_ = score_classification(self.obj, X, y, scoring=scoring)
            if scoring is None:
                self.scoring = "accuracy"

            def predict_proba(x):
                return self.obj.predict_proba(x)[:, self.y_class]

            y_hat = predict_proba(X)

            # heterogeneity of effects
            if method == "avg":

                self.grad = numerical_gradient(
                    predict_proba,
                    X,
                    normalize=self.normalize,
                    n_jobs=self.n_jobs,
                )

            # confidence intervals
            if method == "ci":
                
                if type_ci=="jackknife":

                    self.ci_ = numerical_gradient_jackknife(
                        predict_proba,
                        X,
                        normalize=self.normalize,
                        n_jobs=self.n_jobs,
                        level=level,
                    )
                
                if type_ci=="gaussian":

                    self.ci_ = numerical_gradient_gaussian(
                        predict_proba,
                        X,
                        normalize=self.normalize,
                        n_jobs=self.n_jobs,
                        level=level,
                    )

            # interactions
            if method == "inters":

                assert col_inters is not None, "`col_inters` must be provided"

                self.col_inters = col_inters

                ix1 = np.where(X_names == col_inters)[0][0]

                pbar = Progbar(p)
                
                if type_ci=="jackknife":
                    
                    for ix2 in range(p):

                        self.ci_inters_.update(
                            {
                                X_names[ix2]: numerical_interactions_jackknife(
                                    f=predict_proba,
                                    X=X,
                                    ix1=ix1,
                                    ix2=ix2,
                                    verbose=0,
                                )
                            }
                        )
    
                        pbar.update(ix2)
                
                if type_ci=="gaussian":                
                    
                    for ix2 in range(p):

                        self.ci_inters_.update(
                            {
                                X_names[ix2]: numerical_interactions_gaussian(
                                    f=predict_proba,
                                    X=X,
                                    ix1=ix1,
                                    ix2=ix2,
                                    verbose=0,
                                )
                            }
                        )

                    pbar.update(ix2)

                pbar.update(p)
                print("\n")

        else:  # is_factor(y) == False # regression ---

            self.type_fit = "regression"

            self.score_ = score_regression(self.obj, X, y, scoring=scoring)
            if scoring is None:
                self.scoring = "rmse"

            y_hat = self.obj.predict(X)

            # heterogeneity of effects
            if method == "avg":

                self.grad = numerical_gradient(
                    self.obj.predict,
                    X,
                    normalize=self.normalize,
                    n_jobs=self.n_jobs,
                )

            # confidence intervals
            if method == "ci":
                
                if type_ci=="jackknife": 
                                        
                    self.ci_ = numerical_gradient_jackknife(
                    self.obj.predict,
                    X,
                    normalize=self.normalize,
                    n_jobs=self.n_jobs,
                    level=level,
                )
                    
                
                if type_ci=="gaussian": 
                    
                    self.ci_ = numerical_gradient_gaussian(
                    self.obj.predict,
                    X,
                    normalize=self.normalize,
                    n_jobs=self.n_jobs,
                    level=level,
                )

                

            # interactions
            if method == "inters":

                assert col_inters is not None, "`col_inters` must be provided"

                self.col_inters = col_inters

                ix1 = np.where(X_names == col_inters)[0][0]

                pbar = Progbar(p)
                
                if type_ci=="jackknife": 
                    
                    for ix2 in range(p):
    
                        self.ci_inters_.update(
                            {
                                X_names[ix2]: numerical_interactions_jackknife(
                                    f=self.obj.predict,
                                    X=X,
                                    ix1=ix1,
                                    ix2=ix2,
                                    verbose=0,
                                )
                            }
                        )
                
                if type_ci=="gaussian": 
                    
                    for ix2 in range(p):
    
                        self.ci_inters_.update(
                            {
                                X_names[ix2]: numerical_interactions_gaussian(
                                    f=self.obj.predict,
                                    X=X,
                                    ix1=ix1,
                                    ix2=ix2,
                                    verbose=0,
                                )
                            }
                        )
                    
                    

                    pbar.update(ix2)

                pbar.update(p)
                print("\n")

            self.y_mean_ = np.mean(y)
            ss_tot = np.sum((y - self.y_mean_) ** 2)
            ss_reg = np.sum((y_hat - self.y_mean_) ** 2)
            ss_res = np.sum((y - y_hat) ** 2)

            self.residuals_ = y - y_hat
            self.r_squared_ = 1 - ss_res / ss_tot
            self.adj_r_squared_ = 1 - (1 - self.r_squared_) * (n - 1) / (
                n - p - 1
            )

        # classification and regression ---

        if method == "avg":

            res_df = pd.DataFrame(data=self.grad, columns=X_names)

            res_df_mean = res_df.mean()
            res_df_std = res_df.std()
            res_df_min = res_df.min()
            res_df_max = res_df.max()
            data = pd.concat(
                [res_df_mean, res_df_std, res_df_min, res_df_max], axis=1
            )

            df_effects = pd.DataFrame(
                data=data.values,
                columns=["mean", "std", "min", "max"],
                index=X_names,
            )

            # heterogeneity of effects
            self.effects_ = df_effects.sort_values(by=["mean"], ascending=False)

        return self


    def summary(self):
        """Summary of effects.                           
               
        Returns
        -------
        Prints the summary of effects.                         
        """        

        assert (
            (self.ci_ is not None)
            | (self.effects_ is not None)
            | (self.ci_inters_ is not None)
        ), "object not fitted, fit the object first"

        if (self.ci_ is not None) & (self.method == "ci"):

            # (mean_est, se_est,
            # mean_est + qt*se_est, mean_est - qt*se_est,
            # p_values, signif_codes)

            df_mean = pd.Series(data=self.ci_[0], index=self.X_names)
            df_se = pd.Series(data=self.ci_[1], index=self.X_names)
            df_ubound = pd.Series(data=self.ci_[2], index=self.X_names)
            df_lbound = pd.Series(data=self.ci_[3], index=self.X_names)
            df_pvalue = pd.Series(data=self.ci_[4], index=self.X_names)
            df_signif = pd.Series(data=self.ci_[5], index=self.X_names)

            data = pd.concat(
                [df_mean, df_se, df_lbound, df_ubound, df_pvalue, df_signif],
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
                            self.residuals_, q=[0, 0.25, 0.5, 0.75, 1]
                        ),
                        index=["Min", "1Q", "Median", "3Q", "Max"],
                    )
                ).transpose()

                print(self.residuals_dist_.to_string(index=False))

            print("\n")
            if type_ci=="jackknife": 
                print("Tests on marginal effects (Jackknife): ")
            if type_ci=="gaussian": 
                print("Tests on marginal effects (Gaussian noise): ")
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
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

        if (self.effects_ is not None) & (self.method == "avg"):
            print("\n")
            print("Heterogeneity of marginal effects: ")
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(self.effects_)
            print("\n")

        if (self.ci_inters_ is not None) & (self.method == "inters"):
            print("\n")
            print("Interactions with " + self.col_inters + ": ")
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(
                    pd.DataFrame(
                        self.ci_inters_,
                        index=[
                            "Estimate",
                            "Std. Error",
                            str(95) + "% lbound",
                            str(95) + "% ubound",
                            "Pr(>|t|)",
                            "",
                        ],
                    ).transpose()
                )
