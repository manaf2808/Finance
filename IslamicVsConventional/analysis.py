"""Crisis detection from a stress measure, and the episode list."""

import numpy as np
import pandas as pd

from config import MERGE_GAP, MIN_DAYS


def detect_crises(stress: pd.Series, threshold: float):
    """
    Crisis day = previous day's stress above a fixed level (like VIX > 30).
    Runs of crisis days separated by <= MERGE_GAP calm days are merged; runs shorter than MIN_DAYS dropped.
    Returns (dummy 0/1 Series, [[start_idx, end_idx], ...]).
    """
    flag = (stress.shift(1) > threshold).fillna(False).values
    runs, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= MERGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    episodes = [r for r in merged if r[1] - r[0] + 1 >= MIN_DAYS]
    dummy = np.zeros(n)
    for s, e in episodes:
        dummy[s:e + 1] = 1
    return pd.Series(dummy, index=stress.index, name="crisis"), episodes


def episode_list(combined: pd.DataFrame, stress: pd.Series, episodes: list) -> pd.DataFrame:
    rows = []
    for k, (s, e) in enumerate(episodes, 1):
        w = combined.iloc[s:e + 1]
        ri, rc = (np.exp(w["islamic"].sum()) - 1) * 100, (np.exp(w["conventional"].sum()) - 1) * 100
        rows.append({"Episode": k, "Start": w.index[0].date(), "End": w.index[-1].date(), "Days": len(w),
                     "Peak stress": stress.iloc[s:e + 1].max(),
                     "Islamic return (%)": ri, "Conventional return (%)": rc, "Difference (pp)": ri - rc})
    return pd.DataFrame(rows)
