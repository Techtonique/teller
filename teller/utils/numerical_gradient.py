import numpy as np
from .deepcopy import deepcopy
from .memoize import memoize
from .progress_bar import Progbar
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler


@memoize
def numerical_gradient(
    f, X, normalize=False, 
    h=None, n_jobs=None, verbose=1,
):

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

                grad[:, ix] = (
                    fx_plus - fx_minus
                ) / double_h

                pbar.update(ix)

            if verbose == 1:
                pbar.update(p)
                print("\n")
            
            if normalize == True:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                return scaler.fit_transform(grad)
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
            h = (
                eps_factor * value_x * cond
                + zero * np.logical_not(cond)
            )

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
            return scaler.fit_transform(grad)
        return grad

    # if n_jobs is not None:
    eps_factor = zero ** (1 / 3)

    def gradient_column(ix):

        value_x = deepcopy(X[:, ix])

        cond = np.abs(value_x) > zero
        h = (
            eps_factor * value_x * cond
            + zero * np.logical_not(cond)
        )

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
            delayed(gradient_column)(m)
            for m in tqdm(range(p))
        )
        print("\n")
        
        if normalize == True:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            return scaler.fit_transform(grad)
        return grad

    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(gradient_column)(m) for m in range(p)
    )
    
    if normalize == True:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(grad)
    return grad
