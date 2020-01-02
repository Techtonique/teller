import numpy as np
from .deepcopy import deepcopy
from .memoize import memoize
from .progress_bar import Progbar
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler


@memoize
def numerical_gradient(f, X, normalize=False, h=None, n_jobs=None, verbose=1):

    n, p = X.shape
    grad = np.zeros_like(X)
    zero = 1e-4

    if n_jobs is None:

        # naive version -----

        if h is not None:

            double_h = 2 * h

            if verbose == 1:
                print("\n")
                print("Calculating the effects...")
                pbar = Progbar(p)

            for ix in range(p):

                value_x = deepcopy(X[:, ix])

                X[:, ix] = value_x + h
                fx_plus = f(X)
                X[:, ix] = value_x - h
                fx_minus = f(X)

                X[:, ix] = value_x  # restore (!)

                grad[:, ix] = (fx_plus - fx_minus) / double_h

                pbar.update(ix)

            if verbose == 1:
                pbar.update(p)
                print("\n")

            if normalize == True:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaled_grad = scaler.fit_transform(grad)
                return scaled_grad / scaled_grad.sum(axis=1)[:, None]
            return grad

        # if h is None: -----

        eps_factor = zero ** (1 / 3)

        if verbose == 1:
            print("\n")
            print("Calculating the effects...")
            pbar = Progbar(p)

        for ix in range(p):

            value_x = deepcopy(X[:, ix])

            cond = np.abs(value_x) > zero
            h = eps_factor * value_x * cond + zero * np.logical_not(cond)

            X[:, ix] = value_x + h
            fx_plus = f(X)
            X[:, ix] = value_x - h
            fx_minus = f(X)

            X[:, ix] = value_x  # restore (!)

            grad[:, ix] = (fx_plus - fx_minus) / (2 * h)

            if verbose == 1:
                pbar.update(ix)

        if verbose == 1:
            pbar.update(p)
            print("\n")

        if normalize == True:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_grad = scaler.fit_transform(grad)
            return scaled_grad / scaled_grad.sum(axis=1)[:, None]
        return grad

    # if n_jobs is not None:
    eps_factor = zero ** (1 / 3)

    def gradient_column(ix):

        value_x = deepcopy(X[:, ix])

        cond = np.abs(value_x) > zero
        h = eps_factor * value_x * cond + zero * np.logical_not(cond)

        X[:, ix] = value_x + h
        fx_plus = f(X)
        X[:, ix] = value_x - h
        fx_minus = f(X)

        X[:, ix] = value_x  # restore (!)

        grad[:, ix] = (fx_plus - fx_minus) / (2 * h)

    if verbose == 1:
        print("\n")
        print("Calculating the effects...")
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(gradient_column)(m) for m in tqdm(range(p))
        )
        print("\n")

        if normalize == True:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_grad = scaler.fit_transform(grad)
            return scaled_grad / scaled_grad.sum(axis=1)[:, None]
        return grad

    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(gradient_column)(m) for m in range(p)
    )

    if normalize == True:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_grad = scaler.fit_transform(grad)
        return scaled_grad / scaled_grad.sum(axis=1)[:, None]
    return grad


@memoize
def numerical_interactions(f, X, ix1, ix2, h=None, k=None):

    n, p = X.shape

    # naive version -----

    if h is not None:

        assert k is not None, "`k` must be provided along with `h`"

        value_x1 = deepcopy(X[:, ix1])
        value_x2 = deepcopy(X[:, ix2])

        X[:, ix1] = value_x1 + h
        X[:, ix2] = value_x2 + k
        fx_11 = f(X)
        X[:, ix1] = value_x1 + h
        X[:, ix2] = value_x2 - k
        fx_12 = f(X)
        X[:, ix1] = value_x1 - h
        X[:, ix2] = value_x2 + k
        fx_21 = f(X)
        X[:, ix1] = value_x1 - h
        X[:, ix2] = value_x2 - k
        fx_22 = f(X)

        X[:, ix1] = value_x1  # restore (!)
        X[:, ix2] = value_x2  # restore (!)

        inters = ((fx_11 - fx_12) - (fx_21 - fx_22)) / (4 * (h * k))

        return inters

    # if h is None: -----

    zero = np.finfo(float).eps
    eps_factor = zero ** (1 / 4)

    value_x1 = deepcopy(X[:, ix1])
    value_x2 = deepcopy(X[:, ix2])

    cond1 = np.abs(value_x1) > zero
    cond2 = np.abs(value_x2) > zero

    h1 = eps_factor * value_x1 * cond1 + 1e-4 * np.logical_not(cond1)

    h2 = eps_factor * value_x2 * cond2 + 1e-4 * np.logical_not(cond2)

    X[:, ix1] = value_x1 + h1
    X[:, ix2] = value_x2 + h2
    fx_11 = f(X)
    X[:, ix1] = value_x1 + h1
    X[:, ix2] = value_x2 - h2
    fx_12 = f(X)
    X[:, ix1] = value_x1 - h1
    X[:, ix2] = value_x2 + h2
    fx_21 = f(X)
    X[:, ix1] = value_x1 - h1
    X[:, ix2] = value_x2 - h2
    fx_22 = f(X)

    X[:, ix1] = value_x1  # restore (!)
    X[:, ix2] = value_x2  # restore (!)

    return ((fx_11 - fx_12) - (fx_21 - fx_22)) / (4 * h1 * h2)
