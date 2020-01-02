"""Variance ratio test"""

# Authors: Thierry Moudiki
#
# License: BSD 3


import numpy as np
from scipy.stats import f


# inspired from R's var.test
def var_test(x, y, ratio=1, alternative="less", level=95):

    level = level / 100

    assert alternative in (
        "twosided",
        "less",
        "greater",
    ), "must have `alternative` in ('twosided', 'less', 'greater')"

    dfx = len(x) - 1
    dfy = len(y) - 1

    assert dfx >= 1, "not enough observations for `x`"

    assert dfy >= 1, "not enough observations for `y`"

    varx = np.var(x)
    vary = np.var(y)

    estimate = varx / vary
    statistic = estimate / ratio
    parameter = [dfx, dfy]

    pval = f.cdf(statistic, dfx, dfy)

    if alternative == "twosided":
        pval = 2 * min(pval, 1 - pval)
        beta = (1 - level) / 2
        cint = [
            estimate / f.ppf(1 - beta, dfx, dfy),
            estimate / f.ppf(beta, dfx, dfy),
        ]

    if alternative == "greater":
        pval = 1 - pval
        cint = [estimate / f.ppf(level, dfx, dfy), np.infty]

    if alternative == "less":
        cint = [0, estimate / f.ppf(1 - level, dfx, dfy)]

    return {
        "statistic": statistic,
        "parameter": parameter,
        "p_value": pval,
        "conf_int": cint,
        "estimate": estimate,
        "null_value": ratio,
        "alternative": alternative,
    }
