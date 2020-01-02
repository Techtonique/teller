"""Scoring functions"""

# Authors: Thierry Moudiki
#
# License: BSD 3


import numpy as np
import sklearn.metrics as skm


def score_regression(obj, X, y, scoring=None, **kwargs):
    """ Score the model on test set covariates X and response y. """

    preds = obj.predict(X)

    if type(preds) == tuple:  # if there are std. devs in the predictions
        preds = preds[0]

    if scoring is None:
        scoring = "neg_mean_squared_error"

    # check inputs
    assert scoring in (
        "explained_variance",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "rmse",
    ), "'scoring' should be in ('explained_variance', 'neg_mean_absolute_error', \
                           'neg_mean_squared_error', 'neg_mean_squared_log_error', \
                           'neg_median_absolute_error', 'r2', 'rmse')"

    def f_rmse(x):
        return np.sqrt(skm.mean_squared_error(x))

    scoring_options = {
        "explained_variance": skm.explained_variance_score,
        "neg_mean_absolute_error": skm.mean_absolute_error,
        "neg_mean_squared_error": skm.mean_squared_error,
        "neg_mean_squared_log_error": skm.mean_squared_log_error,
        "neg_median_absolute_error": skm.median_absolute_error,
        "r2": skm.r2_score,
        "rmse": f_rmse,
    }

    return scoring_options[scoring](y, preds, **kwargs)


def score_classification(obj, X, y, scoring=None, **kwargs):
    """ Score the model on test set covariates X and response y. """

    preds = obj.predict(X)

    if scoring is None:
        scoring = "accuracy"

    # check inputs
    assert scoring in (
        "accuracy",
        "average_precision",
        "brier_score_loss",
        "f1",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "f1_samples",
        "neg_log_loss",
        "precision",
        "recall",
        "roc_auc",
    ), "'scoring' should be in ('accuracy', 'average_precision', \
                           'brier_score_loss', 'f1', 'f1_micro', \
                           'f1_macro', 'f1_weighted',  'f1_samples', \
                           'neg_log_loss', 'precision', 'recall', \
                           'roc_auc')"

    scoring_options = {
        "accuracy": skm.accuracy_score,
        "average_precision": skm.average_precision_score,
        "brier_score_loss": skm.brier_score_loss,
        "f1": skm.f1_score,
        "f1_micro": skm.f1_score,
        "f1_macro": skm.f1_score,
        "f1_weighted": skm.f1_score,
        "f1_samples": skm.f1_score,
        "neg_log_loss": skm.log_loss,
        "precision": skm.precision_score,
        "recall": skm.recall_score,
        "roc_auc": skm.roc_auc_score,
    }

    return scoring_options[scoring](y, preds, **kwargs)
