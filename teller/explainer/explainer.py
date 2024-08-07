import numpy as np
import pandas as pd
import seaborn as sns

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from tqdm import tqdm
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
    """Class Explainer: effects of features on the response.

    Attributes:

        obj: an object;
            fitted object containing methods `fit` and `predict`

        n_jobs: an integer;
            number of jobs for parallel computing

        y_class: an integer;
            class whose probability has to be explained (for classification only, default is 0)

        normalize: a boolean;
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
        self.type_ci = None
        self.col_inters = None

    def fit(
        self,
        X,
        y,
        X_names=None,
        method="avg",
        type_ci="jackknife",
        scoring=None,
        level=95,
        col_inters=None,
    ):
        """Fit the explainer's attribute `obj` to training data (X, y).

        Args:

            X: array-like, shape = [n_samples, n_features];
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples, ]; Target values.

            X_names: {array-like}, shape = [n_features, ];
                Column names (strings) for training vectors (default
                is None, and not used if X is a data frame).

            method: str;
                Type of summary requested for effects. Either `avg`
                (for average effects), `inters` (for interactions)
                or `ci` (for effects including confidence intervals
                around them).

            type_ci: str;
                Type of resampling for `method == 'ci'` (confidence
                intervals around effects). Either `jackknife`
                bootsrapping or `gaussian` (gaussian white noise with
                standard deviation equal to `0.01` applied to the
                features).

            scoring: str;
                measure of errors must be in ("explained_variance",
                "neg_mean_absolute_error", "neg_mean_squared_error",
                "neg_mean_squared_log_error", "neg_median_absolute_error",
                "r2", "rmse") (default: "rmse").

            level: int; Level of confidence required for
                `method == 'ci'` (in %).

            col_inters: str; Name of column for computing interactions.

        """
        assert method in (
            "avg",
            "ci",
            "inters",
        ), "must have: `method` in ('avg', 'ci', 'inters')"

        n, p = X.shape

        if isinstance(X, pd.DataFrame):
            self.X_names = X.columns
        else:
            assert (
                X_names is not None
            ), "'X' is a numpy array, 'X_names' must be provided"
            self.X_names = X_names

        self.level = level
        self.method = method
        self.type_ci = type_ci

        if is_factor(y):  # classification ---

            self.n_classes = len(np.unique(y))

            assert (
                self.y_class <= self.n_classes
            ), "self.y_class must be <= number of classes"

            assert hasattr(
                self.obj, "predict_proba"
            ), "`self.obj` must be a classifier and have a method `predict_proba`"

            self.type_fit = "classification"

            if scoring is None:
                self.scoring = "accuracy"

            self.score_ = score_classification(
                self.obj, X, y, scoring=self.scoring
            )

            def predict_proba(x):
                return self.obj.predict_proba(x)[:, self.y_class]

            y_hat = predict_proba(X)
            self.residuals_ = y - y_hat

            # heterogeneity of effects
            if method == "avg":

                self.grad_ = numerical_gradient(
                    predict_proba,
                    X,
                    normalize=self.normalize,
                    n_jobs=self.n_jobs,
                )

            # confidence intervals
            if method == "ci":

                if type_ci == "jackknife":

                    self.ci_ = numerical_gradient_jackknife(
                        predict_proba,
                        X,
                        normalize=self.normalize,
                        n_jobs=self.n_jobs,
                        level=level,
                    )

                if type_ci == "gaussian":

                    self.ci_ = numerical_gradient_gaussian(
                        predict_proba,
                        X,
                        normalize=self.normalize,
                        n_jobs=self.n_jobs,
                        level=level,
                    )

            # interactions
            if method == "inters":

                raise NotImplementedError

                assert col_inters is not None, "`col_inters` must be provided"

                self.col_inters = col_inters

                ix1 = np.where(np.asarray(X_names) == col_inters)[0][0]

                pbar = Progbar(p)

                if type_ci == "jackknife":

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

                if type_ci == "gaussian":

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

            if scoring is None:
                self.scoring = "rmse"

            self.score_ = score_regression(self.obj, X, y, scoring=self.scoring)

            y_hat = self.obj.predict(X)
            self.residuals_ = y - y_hat

            # heterogeneity of effects
            if method == "avg":

                self.grad_ = numerical_gradient(
                    self.obj.predict,
                    X,
                    normalize=self.normalize,
                    n_jobs=self.n_jobs,
                )

            # confidence intervals
            if method == "ci":

                if type_ci == "jackknife":

                    self.ci_ = numerical_gradient_jackknife(
                        self.obj.predict,
                        X,
                        normalize=self.normalize,
                        n_jobs=self.n_jobs,
                        level=level,
                    )

                if type_ci == "gaussian":

                    self.ci_ = numerical_gradient_gaussian(
                        self.obj.predict,
                        X,
                        normalize=self.normalize,
                        n_jobs=self.n_jobs,
                        level=level,
                    )

            # interactions
            if method == "inters":

                raise NotImplementedError

                assert col_inters is not None, "`col_inters` must be provided"

                self.col_inters = col_inters

                ix1 = np.where(np.asarray(X_names) == col_inters)[0][0]

                if self.n_jobs is None:

                    pbar = Progbar(p)

                    if type_ci == "jackknife":

                        for ix2 in range(p):

                            self.ci_inters_.update(
                                {
                                    X_names[
                                        ix2
                                    ]: numerical_interactions_jackknife(
                                        f=self.obj.predict,
                                        X=X,
                                        ix1=ix1,
                                        ix2=ix2,
                                        verbose=0,
                                    )
                                }
                            )

                    if type_ci == "gaussian":

                        for ix2 in range(p):

                            self.ci_inters_.update(
                                {
                                    X_names[
                                        ix2
                                    ]: numerical_interactions_gaussian(
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

                # else self.n_jobs is not None
                def foo_jackknife(ix):
                    return self.ci_inters_.update(
                        {
                            X_names[ix]: numerical_interactions_jackknife(
                                f=self.obj.predict,
                                X=X,
                                ix1=ix1,
                                ix2=ix,
                                verbose=0,
                            )
                        }
                    )

                def foo_gaussian(ix):
                    return self.ci_inters_.update(
                        {
                            X_names[ix]: numerical_interactions_gaussian(
                                f=self.obj.predict,
                                X=X,
                                ix1=ix1,
                                ix2=ix,
                                verbose=0,
                            )
                        }
                    )

                if type_ci == "jackknife":
                    res = Parallel(n_jobs=self.n_jobs)(
                        delayed(foo_jackknife)(ix2) for ix2 in tqdm(range(p))
                    )

                if type_ci == "gaussian":
                    res = Parallel(n_jobs=self.n_jobs)(
                        delayed(foo_gaussian)(ix2) for ix2 in tqdm(range(p))
                    )

            self.y_mean_ = np.mean(y)
            ss_tot = np.sum((y - self.y_mean_) ** 2)
            ss_reg = np.sum((y_hat - self.y_mean_) ** 2)
            ss_res = np.sum((y - y_hat) ** 2)
            
            self.r_squared_ = 1 - ss_res / ss_tot
            self.adj_r_squared_ = 1 - (1 - self.r_squared_) * (n - 1) / (
                n - p - 1
            )

        # classification and regression ---

        if method == "avg":

            res_df = pd.DataFrame(data=self.grad_, columns=X_names)

            res_df_mean = res_df.mean()
            res_df_std = res_df.std()
            res_df_median = res_df.median()
            res_df_min = res_df.min()
            res_df_max = res_df.max()
            data = pd.concat(
                [
                    res_df_mean,
                    res_df_std,
                    res_df_median,
                    res_df_min,
                    res_df_max,
                ],
                axis=1,
            )

            df_effects = pd.DataFrame(
                data=data.values,
                columns=["mean", "std", "median", "min", "max"],
                index=X_names,
            )

            # heterogeneity of effects
            self.effects_ = df_effects.sort_values(by=["mean"], ascending=False)

        return self

    def summary(self):
        """Summarise results

            a method in class Explainer

        Args:

            None

        """

        assert (
            (self.ci_ is not None)
            | (self.effects_ is not None)
            | (self.ci_inters_ is not None)
        ), "object not fitted, fit the object first"

        if (self.ci_ is not None) & (self.method == "ci"):

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
            if self.type_ci == "jackknife":
                print("Tests on marginal effects (Jackknife): ")
            if self.type_ci == "gaussian":
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
            # raise NotImplementedError("This one's a bit tricky, I'm working on it")
            print(f"self.col_inters: {self.col_inters}")
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
                
        return 

    def plot(self, what):
        """Plot average effects, heterogeneity of effects, ...

        Args:

            what: a string;
                if .
        """
        assert self.effects_ is not None, "Call method 'fit' before plotting"
        assert self.grad_ is not None, "Call method 'fit' before plotting"

        # For method == "avg"
        if self.method == "avg":

            if what == "average_effects":
                sns.set(style="darkgrid")
                fi = pd.DataFrame()
                fi["features"] = self.effects_.index.values
                fi["effect"] = self.effects_["mean"].values
                sns.barplot(
                    x="effect",
                    y="features",
                    data=fi.sort_values(by="effect", ascending=False),
                )

            if what == "hetero_effects":
                grads_df = pd.DataFrame(data=self.grad_, columns=self.X_names)
                sorted_columns = list(self.effects_.index.values)  # by mean
                sorted_columns.reverse()
                grads_df = grads_df.reindex(sorted_columns, axis=1)
                sns.set(style="darkgrid")
                grads_df.boxplot(vert=False)

        # For method == "ci"
        if self.method == "ci":
            assert self.ci_ is not None, "Call method 'fit' before plotting"
            raise NotImplementedError("No plot for method == 'ci' yet")

    def get_individual_effects(self):
        assert (
            self.grad_ is not None
        ), "Call method 'fit' before calling this method"
        if self.method == "avg":
            return pd.DataFrame(data=self.grad_, columns=self.X_names)
