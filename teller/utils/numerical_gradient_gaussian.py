import numpy as np
from .memoize import memoize
from .progress_bar import Progbar
from joblib import Parallel, delayed
from tqdm import tqdm
from .numerical_gradient import numerical_gradient
from scipy.stats import t
from scipy.special import expit
from scipy.stats import norm


def get_code_pval(pval):

    assert (pval >= 0) & (pval <= 1), "must have pval >= 0 & pval <= 1"

    if (pval >= 0) & (pval < 0.001):
        return "***"

    if (pval >= 0.001) & (pval < 0.01):
        return "**"

    if (pval >= 0.01) & (pval < 0.05):
        return "*"

    if (pval >= 0.05) & (pval < 0.1):
        return "."

    if pval >= 0.1:
        return "-"


@memoize
def numerical_gradient_gaussian(
    f, X, normalize=False, level=95, h=None, n_jobs=None
):

    n, p = X.shape
    mean_grads = []
    rv = norm()

    if n_jobs is None:

        print("\n")
        print("Calculating the effects...")
        pbar = Progbar(n)

        for i in range(n):
            
            np.random.seed(i)
            
            X_i = X + rv.rvs()*0.01

            grad_i = numerical_gradient(f, X_i, normalize, verbose=0)

            mean_grads.append(np.mean(grad_i, axis=0))

            pbar.update(i)

        pbar.update(n)
        print("\n")

        mean_grads = np.asarray(mean_grads)

        mean_est = np.mean(mean_grads, axis=0)

        se_est = np.clip(
            ((n - 1) * np.var(mean_grads, axis=0)) ** 0.5,
            a_min=np.finfo(float).eps,
            a_max=None,
        )

        t_est = mean_est / se_est

        qt = t.ppf(1 - (1 - level / 100) * 0.5, n - 1)

        p_values = 2 * t.sf(x=np.abs(t_est), df=n - 1)

        # cat("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1", "\n")
        signif_codes = [get_code_pval(elt) for elt in p_values]

        return (
            mean_est,
            se_est,
            mean_est + qt * se_est,
            mean_est - qt * se_est,
            p_values,
            signif_codes,
        )

    # if n_jobs is not None:
    def gradient_column(i):
        np.random.seed(i)            
        X_i = X + rv.rvs()*0.01
        grad_i = numerical_gradient(f, X_i, normalize, verbose=0)
        mean_grads.append(np.mean(grad_i, axis=0))

    print("\n")
    print("Calculating the effects...")
    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(gradient_column)(m) for m in tqdm(range(n))
    )
    print("\n")

    mean_grads = np.asarray(mean_grads)

    mean_est = np.mean(mean_grads, axis=0)

    se_est = np.clip(
        ((n - 1) * np.var(mean_grads, axis=0)) ** 0.5,
        a_min=np.finfo(float).eps,
        a_max=None,
    )

    t_est = mean_est / se_est

    qt = t.ppf(1 - (1 - level / 100) * 0.5, n - 1)

    p_values = 2 * t.sf(x=np.abs(t_est), df=n - 1)

    # cat("Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1", "\n")
    signif_codes = [get_code_pval(elt) for elt in p_values]

    return (
        mean_est,
        se_est,
        mean_est + qt * se_est,
        mean_est - qt * se_est,
        p_values,
        signif_codes,
    )
