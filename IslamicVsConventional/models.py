"""CAPM models (OLS with Newey-West standard errors)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import HAC_LAGS


def _fit(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def capm(y, x) -> dict:
    """Full sample: y = α + β·x + ε"""
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    df.columns = ["y", "x"]
    m = _fit(df["y"], df[["x"]])
    return {"alpha": m.params["const"], "alpha_p": m.pvalues["const"],
            "beta": m.params["x"], "n_obs": int(m.nobs)}


def crisis_capm(y, x, d) -> dict:
    """
    y = α + Δα·D + β·x + Δβ·D·x + ε   (D = 1 on crisis days)
    Calm alpha = α; crisis alpha = α + Δα (tested against 0); Δα tested against 0 = crisis vs calm alpha.
    Calm beta = β; crisis beta = β + Δβ; Δβ tested against 0 = crisis vs calm beta.
    Days where D is missing (before the stress series starts) are dropped.
    """
    df = pd.concat([y, x, d], axis=1, join="inner").dropna()
    df.columns = ["y", "x", "d"]
    n_c = int(df["d"].sum())
    if n_c < 5 or len(df) - n_c < 5:
        return {"error": "too few crisis days", "n_crisis": n_c}
    df["dx"] = df["d"] * df["x"]
    m = _fit(df["y"], df[["x", "d", "dx"]])
    p, cov = m.params, m.cov_params()
    a = p["const"] + p["d"]
    var = cov.loc["const", "const"] + cov.loc["d", "d"] + 2 * cov.loc["const", "d"]
    a_p = 2 * (1 - stats.norm.cdf(abs(a / np.sqrt(var)))) if var > 0 else np.nan
    return {"alpha_calm": p["const"], "alpha_calm_p": m.pvalues["const"],
            "alpha_crisis": a, "alpha_crisis_p": a_p,
            "alpha_change": p["d"], "alpha_change_p": m.pvalues["d"],
            "beta_calm": p["x"], "beta_crisis": p["x"] + p["dx"],
            "beta_change": p["dx"], "beta_change_p": m.pvalues["dx"],
            "n_crisis": n_c, "n_obs": int(m.nobs), "error": None}
