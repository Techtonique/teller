import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple 
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity


def simulate_distribution(data, method="kde", num_samples=1000, **kwargs):
    """
    Simulate the distribution of an input vector using various methods.

    Parameters:
        data (array-like): Input vector of data.
        method (str): Method for simulation:
                      - 'bootstrap': Bootstrap resampling.
                      - 'kde': Kernel Density Estimation.
                      - 'normal': Normal distribution.
                      - 'ecdf': Empirical CDF-based sampling.
                      - 'permutation': Permutation resampling.
                      - 'smooth-bootstrap': Smoothed bootstrap with added noise.
        num_samples (int): Number of samples to generate.
        kwargs: Additional parameters for specific methods:
                - kde_bandwidth (str or float): Bandwidth for KDE ('scott', 'silverman', or float).
                - dist (str): Parametric distribution type ('normal').
                - noise_std (float): Noise standard deviation for smoothed bootstrap.

    Returns:
        np.ndarray: Simulated distribution samples.
    """
    assert method in [
        "bootstrap",
        "kde",
        "parametric",
        "ecdf",
        "permutation",
        "smooth-bootstrap",
    ], f"Unknown method '{method}'. Choose from 'bootstrap', 'kde', 'parametric', 'ecdf', 'permutation', or 'smooth_bootstrap'."

    data = np.array(data)
    print(f"Input data shape: {data.shape}")

    if method == "bootstrap":
        simulated_data = np.random.choice(data, size=num_samples, replace=True)

    elif method == "kde":
        if len(data.shape) == 1:
          kde_bandwidth = kwargs.get("kde_bandwidth", "scott")
          kde = gaussian_kde(data, bw_method=kde_bandwidth)
          simulated_data = kde.resample(num_samples).flatten()
        else:
          kde_bandwidth = kwargs.get("kde_bandwidth", "scott")
          kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian")
          kde.fit(data)
          simulated_data = kde.sample(num_samples)

    elif method == "normal":
        mean, std = np.mean(data), np.std(data)
        simulated_data = np.random.normal(mean, std, size=num_samples)

    elif method == "ecdf":
        data = np.sort(data)
        ecdf_y = np.arange(1, len(data) + 1) / len(data)
        inverse_cdf = interp1d(
            ecdf_y, data, bounds_error=False, fill_value=(data[0], data[-1])
        )
        random_uniform = np.random.uniform(0, 1, size=num_samples)
        simulated_data = inverse_cdf(random_uniform)

    elif method == "permutation":
        simulated_data = np.random.permutation(data)
        while len(simulated_data) < num_samples:
            simulated_data = np.concatenate(
                [simulated_data, np.random.permutation(data)]
            )
        simulated_data = simulated_data[:num_samples]

    elif method == "smooth_bootstrap":
        noise_std = kwargs.get("noise_std", 0.1)
        bootstrap_samples = np.random.choice(
            data, size=num_samples, replace=True
        )
        noise = np.random.normal(0, noise_std, size=num_samples)
        simulated_data = bootstrap_samples + noise

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'bootstrap', 'kde', 'parametric', 'ecdf', 'permutation', or 'smooth_bootstrap'."
        )

    return simulated_data


def simulate_replications(
    data, method="kde",
    num_replications=10, **kwargs
):
    """
    Create multiple replications of the input's distribution using a specified simulation method.

    Parameters:
        data (array-like): Input vector of data.
        method (str): Method for simulation:
                      - 'bootstrap': Bootstrap resampling.
                      - 'kde': Kernel Density Estimation.
                      - 'normal': Parametric distribution fitting.
                      - 'ecdf': Empirical CDF-based sampling.
                      - 'permutation': Permutation resampling.
                      - 'smooth_bootstrap': Smoothed bootstrap with added noise.
        num_samples (int): Number of samples in each replication.
        num_replications (int): Number of replications to generate.
        kwargs: Additional parameters for specific methods.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a replication.
    """
    data = np.array(data)
    print(f"Input data shape: {data.shape}")

    num_samples = len(data)

    replications = []

    for _ in range(num_replications):
        simulated_data = simulate_distribution(
            data, method=method, num_samples=num_samples, **kwargs
        )
        replications.append(simulated_data)

    # Combine replications into a DataFrame
    replications_df = pd.DataFrame(replications).transpose()
    replications_df.columns = [
        f"Replication_{i+1}" for i in range(num_replications)
    ]

    return replications_df

def finite_difference_sensitivity(model, X, class_index=None):
    """
    Compute the average sensitivities of the model's predictions with respect to each feature
    using second-order finite differences (central difference approximation) across multiple samples.

    Parameters:
    - model: The trained model with a `predict` method.
    - X: A 2D array of input data points (multiple samples) to evaluate sensitivities.
    - class_index: Index of class to evaluate sensitivities for.

    Returns:
    - avg_sensitivities: An array of average sensitivities for each feature across the samples.
    """
    zero = 1e-4
    eps_factor = zero ** (1 / 3)

    num_samples, num_features = X.shape

    # Prepare arrays for forward and backward predictions
    forward_predictions = np.zeros((num_samples, num_features))
    backward_predictions = np.zeros((num_samples, num_features))
    derivatives = np.zeros((num_samples, num_features))

    # Iterate through each sample to calculate forward and backward perturbations
    if class_index is None:
        for i in range(num_features):
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
        assert class_index <= model.n_classes, "class_index must be less than or equal to the number of classes"
        for i in range(num_features):
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
            forward_predictions[:, i] = model.predict_proba(perturbed_inputs_forward)[:, class_index]
            backward_predictions[:, i] = model.predict_proba(perturbed_inputs_backward)[:, class_index]
            derivatives[:, i] = (forward_predictions[:, i] - backward_predictions[:, i]) / double_h

    return derivatives

def sensitivity_confidence_intervals(model, X_test,
                                     confidence_level=0.95,
                                     seed=123):
    """
    Compute confidence intervals for the average sensitivities of the model's predictions with respect to each feature.

    Parameters:
    - model: The trained model with a `predict` method.
    - X_test: A 2D array of input data points (multiple samples) to evaluate sensitivities.
    - confidence_level: The desired confidence level for the confidence intervals.
    - seed: The seed for the random number generator.
    """

    derivatives = finite_difference_sensitivity(model,
                                                X_test)
    np.random.seed(seed)
    np.random.shuffle(derivatives)

    n_samples = derivatives.shape[0]
    half_n_samples = n_samples // 2
    derivatives_train = derivatives[:half_n_samples]
    derivatives_cal = derivatives[half_n_samples:]
    mean_derivatives_train = np.mean(derivatives_train, axis=0)
    mean_derivatives_cal = np.mean(derivatives_cal, axis=0)
    abs_residuals = np.abs(derivatives_cal - mean_derivatives_train[np.newaxis, :])
    quantiles_abs_residuals = np.quantile(abs_residuals, confidence_level, axis=0)
    mean_estimate = mean_derivatives_cal
    median_estimate = np.median(derivatives_cal, axis=0)
    lower_bounds = mean_derivatives_cal - quantiles_abs_residuals[np.newaxis, :]
    upper_bounds = mean_derivatives_cal + quantiles_abs_residuals[np.newaxis, :]

    # Create a namedtuple to store the results
    DescribeResult = namedtuple('DescribeResult', ['mean', 'median', 
                                                   'lower', 'upper',
                                                  'derivatives', 
                                                  'signif_codes', 
                                                  'pi_length'])
    DescribeResult.mean = mean_estimate.ravel()
    DescribeResult.median = median_estimate.ravel()
    DescribeResult.lower = lower_bounds.ravel()
    DescribeResult.upper = upper_bounds.ravel()
    DescribeResult.derivatives = derivatives
    DescribeResult.signif_codes = (DescribeResult.lower*DescribeResult.upper > 0)
    DescribeResult.pi_length = DescribeResult.upper - DescribeResult.lower

    return DescribeResult