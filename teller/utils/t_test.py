"""Paired t-test"""

import numpy as np
import math
from scipy.stats import t


def t_test(x, y, alternative="less", var_equal=False, mu=0, conf_level=0.95):

    assert alternative in (
        "two.sided",
        "less",
        "greater",
    ), "must have: `alternative` in ('two.sided', 'less', 'greater')"

    nx = len(x)
    ny = len(y)
    mx = np.mean(x)
    vx = nx * np.var(x) / (nx - 1)
    my = np.mean(y)
    vy = ny * np.var(y) / (ny - 1)

    estimate = {"mean of x": mx, "mean of y": my}

    if var_equal:
        df = nx + ny - 2
        v = 0
        if nx > 1:
            v = v + (nx - 1) * vx
        if ny > 1:
            v = v + (ny - 1) * vy
        v = v / df
        stderr = math.sqrt(v * (1 / nx + 1 / ny))

    else:
        stderrx = math.sqrt(vx / nx)
        stderry = math.sqrt(vy / ny)
        stderr = math.sqrt(stderrx ** 2 + stderry ** 2)
        df = stderr ** 4 / (stderrx ** 4 / (nx - 1) + stderry ** 4 / (ny - 1))

    if stderr < 10 * np.finfo(float).eps * max(math.fabs(mx), math.fabs(my)):
        raise ValueError("data are essentially constant")

    tstat = (mx - my - mu) / stderr

    if alternative == "less":
        pval = t.cdf(tstat, df)
        cint = np.array([-np.inf, tstat + t.ppf(conf_level, df)])

    elif alternative == "greater":
        pval = 1 - t.cdf(tstat, df)
        cint = np.array([tstat - t.ppf(conf_level, df), np.inf])

    else:
        pval = 2 * t.cdf(-math.fabs(tstat), df)
        alpha = 1 - conf_level
        cint = t.ppf(1 - alpha / 2, df)
        cint = tstat + np.array([-cint, cint])

    cint = mu + cint * stderr

    return {
        "statistic": tstat,
        "parameter": df,
        "p.value": pval,
        "f.int": cint,
        "estimate": estimate,
        "null.value": mu,
        "alternative": alternative,
    }
