"""Significance and formatting helpers."""

import pandas as pd

from config import TRADING_DAYS


def stars(p) -> str:
    if p is None or pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def sig_phrase(p) -> str:
    if p is None or pd.isna(p):
        return "significance not available"
    if p < 0.01:
        return f"significant at 1%, p = {p:.3f}"
    if p < 0.05:
        return f"significant at 5%, p = {p:.3f}"
    if p < 0.10:
        return f"weakly significant at 10%, p = {p:.3f}"
    return f"not significant, p = {p:.3f}"


def is_sig(p, level: float = 0.05) -> bool:
    return p is not None and not pd.isna(p) and p < level


def annualise(v):
    """Daily decimal -> annualised %."""
    return v * TRADING_DAYS * 100 if pd.notna(v) else float("nan")


def risk_table(isl, conv, rf, isl_name, conv_name) -> pd.DataFrame:
    """Annualised return, volatility, Sharpe and Sortino for both funds."""
    import numpy as np
    rows = {}
    for name, r in ((isl_name, isl), (conv_name, conv)):
        ex = r - rf.reindex(r.index).fillna(0.0)
        down = ex[ex < 0].std(ddof=1)
        rows[name] = {
            "Ann. return (%)": r.mean() * TRADING_DAYS * 100,
            "Ann. volatility (%)": r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100,
            "Sharpe ratio": ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS),
            "Sortino ratio": ex.mean() / down * np.sqrt(TRADING_DAYS) if down > 0 else float("nan"),
        }
    return pd.DataFrame(rows).T
