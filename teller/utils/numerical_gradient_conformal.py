import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from copy import deepcopy
from tqdm import tqdm


def finite_difference_sensitivity(model, X, n_jobs=1, show_progress=False):
    """
    Compute the average sensitivities of the model's predictions with respect to each feature
    using second-order finite differences (central difference approximation) across multiple samples.

    Parameters:
    - model: The trained model with a `predict` method.
    - X: A 2D array of input data points (multiple samples) to evaluate sensitivities.
    - n_jobs: The number of jobs to run in parallel.
    - show_progress: Whether to show a progress bar.
    Returns:
    - avg_sensitivities: An array of average sensitivities for each feature across the samples.
    """
    zero = 1e-4
    eps_factor = zero ** (1 / 3)

    if isinstance(X, pd.DataFrame):
        is_dataframe = True
        X = X.values
        feature_names = X.columns
    else:
        is_dataframe = False

    num_samples, num_features = X.shape

    # Prepare arrays for forward and backward predictions
    forward_predictions = np.zeros((num_samples, num_features))
    backward_predictions = np.zeros((num_samples, num_features))
    derivatives = np.zeros((num_samples, num_features))

    if show_progress:
        iterator = tqdm(range(num_features))
    else:
        iterator = range(num_features)

    if n_jobs == 1:
    
        # Iterate through each sample to calculate forward and backward perturbations
        for i in iterator:

            inputs = X.copy()
            perturbed_inputs_forward = X.copy()
            perturbed_inputs_backward = X.copy()
            # Perturb the i-th feature
            value_x = inputs[:, i]
            cond = np.abs(value_x) > zero
            h = eps_factor * value_x * cond + zero * np.logical_not(cond)
            perturbed_inputs_forward[:, i] += h
            perturbed_inputs_backward[:, i] -= h
            # Make predictions for perturbed inputs
            double_h = 2 * h
            forward_predictions[:, i] = model.predict(perturbed_inputs_forward)
            backward_predictions[:, i] = model.predict(perturbed_inputs_backward)
            derivatives[:, i] = (forward_predictions[:, i] - backward_predictions[:, i]) / double_h

    else:

        def compute_derivative(i):
            inputs = X.copy()
            perturbed_inputs_forward = X.copy()
            perturbed_inputs_backward = X.copy()
            # Perturb the i-th feature
            value_x = inputs[:, i]
            cond = np.abs(value_x) > zero
            h = eps_factor * value_x * cond + zero * np.logical_not(cond)
            perturbed_inputs_forward[:, i] += h
            perturbed_inputs_backward[:, i] -= h
            # Make predictions for perturbed inputs
            double_h = 2 * h
            forward_predictions[:, i] = model.predict(perturbed_inputs_forward)
            backward_predictions[:, i] = model.predict(perturbed_inputs_backward)
            return (forward_predictions[:, i] - backward_predictions[:, i]) / double_h
        
        # Use joblib to parallelize the computation
        with Parallel(n_jobs=n_jobs) as parallel:
            derivatives = parallel(delayed(compute_derivative)(i) for i in iterator)

    if is_dataframe:
        derivatives = pd.DataFrame(derivatives, columns=feature_names)
    else:
        derivatives = np.array(derivatives)

    return derivatives


def finite_difference_interaction(model, X, ix, n_jobs=1, show_progress=False):
    """
    Compute the interaction between the ix1-th and ix2-th features using finite differences.
    """
    n, p = X.shape
    zero = np.finfo(float).eps
    eps_factor = zero ** (1 / 4)
    value_x = deepcopy(X[:, ix])
    cond_x = np.abs(value_x) > zero
    h = eps_factor * value_x * cond_x + 1e-4 * np.logical_not(cond_x)

    if isinstance(X, pd.DataFrame):
        is_dataframe = True
        X = X.values
        feature_names = X.columns
    else:
        is_dataframe = False

    if show_progress:
        iterator = tqdm(range(p))
    else:
        iterator = range(p)

    if n_jobs == 1:

        perturbed_inputs_forward11 = X.copy()
        perturbed_inputs_forward12 = X.copy()
        perturbed_inputs_backward21 = X.copy()
        perturbed_inputs_backward22 = X.copy()

        perturbed_inputs_forward11[:, ix] = value_x + h
        perturbed_inputs_forward12[:, ix] = value_x + h
        perturbed_inputs_backward21[:, ix] = value_x - h
        perturbed_inputs_backward22[:, ix] = value_x - h

        for i in iterator:

            if i != ix:

                value_ix = deepcopy(X[:, i])
                cond_ix = np.abs(value_ix) > zero
                k = eps_factor * value_ix * cond_ix + 1e-4 * np.logical_not(cond_ix)

                forward_forward_predictions = np.zeros((n, p))
                forward_backward_predictions = np.zeros((n, p))
                backward_forward_predictions = np.zeros((n, p))
                backward_backward_predictions = np.zeros((n, p))  

                derivatives = np.zeros((n, p))                              

                perturbed_inputs_forward11[:, i] = value_ix + k
                perturbed_inputs_forward12[:, i] = value_ix - k
                perturbed_inputs_backward21[:, i] = value_ix + k
                perturbed_inputs_backward22[:, i] = value_ix - k

                forward_forward_predictions[:, i] = model.predict(perturbed_inputs_forward11)
                forward_backward_predictions[:, i] = model.predict(perturbed_inputs_forward12)
                backward_forward_predictions[:, i] = model.predict(perturbed_inputs_backward21)
                backward_backward_predictions[:, i] = model.predict(perturbed_inputs_backward22)

                derivatives[:, i] = (forward_forward_predictions[:, i] - forward_backward_predictions[:, i] - backward_forward_predictions[:, i] + backward_backward_predictions[:, i]) / (4 * k * h)

    else:

        def compute_derivative(i):
            value_ix = deepcopy(X[:, i])
            cond_ix = np.abs(value_ix) > zero
            k = eps_factor * value_ix * cond_ix + 1e-4 * np.logical_not(cond_ix)

            forward_forward_predictions = np.zeros((n, p))
            forward_backward_predictions = np.zeros((n, p))
            backward_forward_predictions = np.zeros((n, p))
            backward_backward_predictions = np.zeros((n, p))  

            derivatives = np.zeros((n, p))                              

            perturbed_inputs_forward11[:, i] = value_ix + k
            perturbed_inputs_forward12[:, i] = value_ix - k
            perturbed_inputs_backward21[:, i] = value_ix + k
            perturbed_inputs_backward22[:, i] = value_ix - k

            forward_forward_predictions[:, i] = model.predict(perturbed_inputs_forward11)
            forward_backward_predictions[:, i] = model.predict(perturbed_inputs_forward12)
            backward_forward_predictions[:, i] = model.predict(perturbed_inputs_backward21)
            backward_backward_predictions[:, i] = model.predict(perturbed_inputs_backward22)

            derivatives[:, i] = (forward_forward_predictions[:, i] - forward_backward_predictions[:, i] - backward_forward_predictions[:, i] + backward_backward_predictions[:, i]) / (4 * k * h)
        
        with Parallel(n_jobs=n_jobs) as parallel:
            derivatives = parallel(delayed(compute_derivative)(i) for i in iterator)

    if is_dataframe:
        derivatives = pd.DataFrame(derivatives, columns=feature_names)
    else:
        derivatives = np.array(derivatives)

    return derivatives  