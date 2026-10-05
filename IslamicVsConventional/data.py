"""Data download and preparation."""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import (CREDIT_ID, EM_TICKER, ENERGY_TICKER, GPR_COLUMN, RATES_TICKER, STLFSI_ID, TECH_TICKER,
                    TRADING_DAYS, VIX_TICKER)


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_series(ticker: str, start: str, end: str) -> pd.Series:
    """Adjusted daily close prices (or index levels) for one Yahoo ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
    close.name = ticker
    return close.dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def build_dataset(isl, conv, rf_ticker, start, end):
    """Returns (combined log returns ['islamic','conventional'], daily risk-free rate, error)."""
    start, end = str(start), str(end)
    isl_px = download_price_series(isl, start, end)
    conv_px = download_price_series(conv, start, end)
    if isl_px.empty or conv_px.empty:
        return None, None, f"Could not download '{isl}' or '{conv}'. Check the symbols."
    combined = pd.concat([log_returns(isl_px), log_returns(conv_px)], axis=1, join="inner").dropna()
    combined.columns = ["islamic", "conventional"]

    rf = pd.Series(0.0, index=combined.index)
    if rf_ticker:
        raw = download_price_series(rf_ticker, start, end)
        if not raw.empty:                                 # ^IRX is an annual % -> daily decimal
            rf = ((1 + raw / 100.0) ** (1 / TRADING_DAYS) - 1).reindex(combined.index).ffill().fillna(0.0)
    return combined, rf, None


def _read_csv_any(source) -> pd.DataFrame:
    """Read a CSV saved by FRED, Excel (French or English) or by hand: ',' or ';' separator."""
    if hasattr(source, "seek"):
        source.seek(0)
    df = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _num(col: pd.Series) -> pd.Series:
    """'123,45' -> 123.45 ; '.' or '' (missing) -> NaN."""
    return pd.to_numeric(col.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def _dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Find the date column (date / DAY / observation_date / first column) and parse it."""
    lower = {c.lower(): c for c in df.columns}
    col = next((lower[k] for k in ("date", "day", "observation_date") if k in lower), df.columns[0])
    txt = df[col].astype(str).str.strip()
    if txt.str.fullmatch(r"\d{8}(\.0)?").all():                                     # 20000101
        return pd.DatetimeIndex(pd.to_datetime(txt.str[:8], format="%Y%m%d", errors="coerce"))
    dayfirst = txt.str.contains("/").any()                                          # Excel FR: 01/02/2000
    return pd.DatetimeIndex(pd.to_datetime(txt, dayfirst=dayfirst, errors="coerce"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_gpr(source) -> pd.DataFrame:
    """Daily GPR: needs a date column (date or DAY) and GPRD; GPRD_MA7 is computed if absent."""
    df = _read_csv_any(source)
    lower = {c.lower(): c for c in df.columns}
    if "gprd" not in lower:
        raise ValueError(f"no GPRD column; columns found: {list(df.columns)}")
    out = pd.DataFrame({"GPRD": _num(df[lower["gprd"]]).values}, index=_dates(df))
    if "gprd_ma7" in lower:
        out["GPRD_MA7"] = _num(df[lower["gprd_ma7"]]).values
    out = out[out.index.notna() & out["GPRD"].notna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if "GPRD_MA7" not in out or out["GPRD_MA7"].isna().all():
        out["GPRD_MA7"] = out["GPRD"].rolling(7, min_periods=1).mean()
    if out.empty:
        raise ValueError(f"no valid rows; columns found: {list(df.columns)}")
    return out[["GPRD", "GPRD_MA7"]]


@st.cache_data(ttl=86400, show_spinner=False)
def load_fred(source, series_id: str) -> pd.Series:
    """Any FRED series from its downloaded CSV (columns: date, value). Missing values ('.') are dropped."""
    df = _read_csv_any(source)
    val_col = series_id if series_id in df.columns else df.columns[1]
    s = pd.Series(_num(df[val_col]).values, index=_dates(df), name=series_id)
    s = s[s.index.notna()].dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def _on_dates(s: pd.Series, index) -> pd.Series:
    """Align a weekly/daily series to the return dates (last known value)."""
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def _yahoo_level(ticker, start, end, index):
    s = download_price_series(ticker, start, end)
    return None if s.empty else _on_dates(s, index)


def build_stress(index, start, end, gpr_source, fred_sources: dict) -> tuple:
    """
    Daily stress levels on the return dates.
      Market:       CBOE VIX
      Banking:      St. Louis Fed Financial Stress Index (weekly, carried forward)
      Energy:       CBOE Crude Oil Volatility Index (OVX)
      Technology:   CBOE Nasdaq-100 Volatility Index (VXN)
      Geopolitical: daily GPR, 7-day moving average
      Credit:       Moody's Baa spread over 10-year Treasury (FRED BAA10Y)
      Rates:        ICE BofA MOVE Index
      EM:           CBOE Emerging Markets ETF Volatility Index (VXEEM)
    fred_sources = {"Banking": path, "Credit": path}  (local CSV files)
    Returns (stress DataFrame, components dict, list of warnings).
    """
    start, end = str(start), str(end)
    out, parts, notes = pd.DataFrame(index=index), {}, []

    for name, ticker in (("Market", VIX_TICKER), ("Energy", ENERGY_TICKER), ("Technology", TECH_TICKER),
                         ("Rates", RATES_TICKER), ("EM", EM_TICKER)):
        s = _yahoo_level(ticker, start, end, index)
        if s is None:
            notes.append(f"{ticker} could not be downloaded: {name.lower()} stress skipped.")
        else:
            out[name] = s

    for name, sid in (("Banking", STLFSI_ID), ("Credit", CREDIT_ID)):
        try:
            out[name] = _on_dates(load_fred(fred_sources[name], sid), index)
        except Exception as e:  # noqa: BLE001
            notes.append(f"FRED series {sid} could not be loaded ({e}). Download it from FRED (CSV, date range "
                         f"Max) and save it as {sid}.csv in the app folder.")

    if gpr_source is None:
        notes.append("GPR file not found: geopolitical stress skipped.")
    else:
        try:
            g = load_gpr(gpr_source)
            g = g.reindex(g.index.union(index)).ffill().reindex(index)
            out["Geopolitical"], parts["Geopolitical"] = g[GPR_COLUMN], g
        except Exception as e:  # noqa: BLE001
            notes.append(f"GPR file could not be read ({e}): geopolitical stress skipped.")
    return out, parts, notes
