#!/usr/bin/env python3
"""Shared utilities for CSI Consumer Index analysis project."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CSI_CONSUMER_SYMBOL = "sz399932"
HS300_SYMBOL = "sh000300"
TRADING_DAYS_PER_YEAR = 252


def fetch_index_daily(symbol: str, start_date: str) -> pd.DataFrame:
    """Fetch index daily OHLCV from AkShare, return DataFrame with [date, close].

    Tries `stock_zh_index_daily` (primary) with retry; falls back to `index_zh_a_hist`.
    """
    import akshare as ak

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalise column names from both AKShare endpoints
        date_candidates = [c for c in df.columns if "date" in str(c).lower() or "日" in str(c)]
        close_candidates = [c for c in df.columns if "close" in str(c).lower() or "收盘" in str(c)]
        if not date_candidates or not close_candidates:
            raise RuntimeError(f"Unexpected columns from AKShare for {symbol}: {list(df.columns)}")
        df = df.rename(columns={date_candidates[0]: "date", close_candidates[0]: "close"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        df = df[df["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"{symbol} has no valid data after {start_date}.")
        return df[["date", "close"]]

    def _try_fetch(fn, max_retries=3):
        last_err = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (2**attempt))
        raise last_err

    # Primary: stock_zh_index_daily
    try:
        df = _try_fetch(lambda: ak.stock_zh_index_daily(symbol=symbol))
        return _clean(df)
    except Exception:
        pass

    # Fallback: index_zh_a_hist
    symbol_num = symbol[2:] if symbol.startswith(("sz", "sh")) else symbol
    try:
        df = _try_fetch(lambda: ak.index_zh_a_hist(
            symbol=symbol_num, period="daily",
            start_date=start_date.replace("-", ""),
            end_date=pd.Timestamp.today().strftime("%Y%m%d"),
        ))
        return _clean(df)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {symbol} via both AKShare endpoints.") from e


def fetch_index_daily_cached(
    symbol: str, start_date: str, end_date: str | None = None, cache_path: str | None = None
) -> pd.DataFrame:
    """Like fetch_index_daily but with CSV cache + extended retry logic for GARCH workflows."""
    import akshare as ak

    if cache_path is None:
        cache_path = str(Path(__file__).resolve().parent.parent / "sz399932_akshare_cache.csv")

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    def _load_cache():
        if not os.path.exists(cache_path):
            return None
        try:
            cached = pd.read_csv(cache_path)
            for col_pair in [("date", "close"), ("日期", "收盘")]:
                if col_pair[0] in cached.columns and col_pair[1] in cached.columns:
                    cached = cached.rename(columns={col_pair[0]: "date", col_pair[1]: "close"})
                    break
            else:
                return None
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached["close"] = pd.to_numeric(cached["close"], errors="coerce")
            cached = cached.dropna(subset=["date", "close"]).sort_values("date")
            s = pd.to_datetime(start_date, errors="coerce")
            e = pd.to_datetime(end_date, errors="coerce")
            cached = cached[(cached["date"] >= s) & (cached["date"] <= e)]
            return cached.loc[:, ["date", "close"]].copy() if not cached.empty else None
        except Exception:
            return None

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        cols = set(df.columns)
        if {"日期", "收盘"}.issubset(cols):
            df = df.rename(columns={"日期": "date", "收盘": "close"})
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        return df[["date", "close"]].reset_index(drop=True)

    def _fetch_with_retries(fn, max_retries=5, base_sleep=2.0):
        last_err = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_err = e
                time.sleep(base_sleep * (2**attempt))
        raise last_err

    cached = _load_cache()

    symbol_num = symbol[2:] if symbol.startswith(("sz", "sh")) else symbol
    start_fmt = start_date.replace("-", "") if "-" in start_date else start_date
    end_fmt = end_date.replace("-", "") if "-" in end_date else end_date

    try:
        df = _fetch_with_retries(lambda: ak.index_zh_a_hist(
            symbol=symbol_num, period="daily",
            start_date=start_fmt, end_date=end_fmt,
        ))
        out = _clean(df)
    except Exception:
        try:
            df = _fetch_with_retries(lambda: ak.stock_zh_index_daily(symbol=symbol), max_retries=3, base_sleep=1.5)
            out = _clean(df)
        except Exception:
            if cached is not None:
                print(f"[Warning] AKShare fetch failed; using cached data from {cache_path}.")
                return cached
            raise RuntimeError("Failed to fetch index data from AKShare (primary + fallback).")

    try:
        out.to_csv(cache_path, index=False)
    except Exception:
        pass

    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using exponential moving averages."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def estimate_annual_params(prices: pd.Series) -> tuple[float, float]:
    """Estimate annual arithmetic mean return and annualized volatility from daily close."""
    daily_returns = prices.pct_change().dropna()
    if daily_returns.empty:
        raise RuntimeError("Insufficient history to estimate return/volatility.")
    annual_mean_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    annual_volatility = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return annual_mean_return, annual_volatility


def garch_fit_and_forecast(
    prices: pd.Series, forecast_horizon: int = 252
) -> tuple[pd.Series, np.ndarray, pd.Timestamp]:
    """Fit GARCH(1,1) on daily log-returns; return historical conditional vol, forward forecast, and forecast start date.

    Returns:
        cond_vol_annual: historical daily conditional volatility, annualized (%)
        forecast_vol_annual: forecasted vol for next `forecast_horizon` trading days, annualized (%)
        forecast_start: first forecast date (day after last fitted date in the return series)
    """
    from arch import arch_model

    log_returns = np.log(prices).diff()
    returns_pct = (log_returns * 100.0).dropna()
    if len(returns_pct) < 100:
        raise ValueError(f"Not enough data for GARCH fitting: {len(returns_pct)} points.")

    am = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Constant", dist="t", rescale=False)
    res = am.fit(disp="off")

    cond_vol_annual = res.conditional_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    fc = res.forecast(horizon=forecast_horizon)
    var_next = fc.variance.iloc[-1].to_numpy()
    vol_next_annual = np.sqrt(var_next) * np.sqrt(TRADING_DAYS_PER_YEAR)

    forecast_start = returns_pct.index[-1] + pd.Timedelta(days=1)

    return cond_vol_annual, vol_next_annual, forecast_start


def daily_vol_to_monthly(
    daily_vol_annual: np.ndarray,
    months: pd.DatetimeIndex,
    forecast_start: pd.Timestamp,
) -> np.ndarray:
    """Convert daily annualized vol forecast to monthly resolution by year-month grouping.

    Uses calendar-year-month labels rather than searchsorted to correctly map each
    forecasted trading day to its calendar month regardless of month-end exact dates.

    Args:
        daily_vol_annual: shape (n_forecast_days,) — annualized vol % for each forecast day
        months: DatetimeIndex of month-end dates (e.g. [2025-09-30, 2025-10-31, ...])
        forecast_start: timestamp of the first forecast day
    Returns:
        shape (len(months),) — monthly annualized vol % via RMS
    """
    n_days = len(daily_vol_annual)
    if n_days == 0:
        fallback = np.nanmean(daily_vol_annual) if hasattr(daily_vol_annual, '__len__') else daily_vol_annual
        return np.full(len(months), fallback)

    forecast_dates = pd.bdate_range(start=forecast_start, periods=n_days, freq="B")
    forecast_ym = forecast_dates.strftime("%Y-%m")
    months_ym = pd.Series(months).dt.strftime("%Y-%m")
    mapping = {m: i for i, m in enumerate(months_ym)}

    month_idx_list = []
    for ym in forecast_ym:
        idx = mapping.get(ym, -1)
        month_idx_list.append(idx)

    month_idx = np.array(month_idx_list, dtype=int)

    monthly_vol = np.full(len(months), np.nan, dtype=float)
    for i in range(len(months)):
        mask = month_idx == i
        if mask.sum() > 0:
            monthly_vol[i] = np.sqrt(np.mean(daily_vol_annual[mask] ** 2))

    # Forward-fill any months that received no forecast days
    last_valid = np.nan
    for i in range(len(monthly_vol)):
        if not np.isnan(monthly_vol[i]):
            last_valid = monthly_vol[i]
        elif not np.isnan(last_valid):
            monthly_vol[i] = last_valid
        else:
            monthly_vol[i] = np.nanmean(daily_vol_annual) if n_days > 0 else 0.0

    return monthly_vol


def fit_asymmetric_vol_models(
    returns_pct: pd.Series, forecast_horizon: int = 252
) -> dict[str, dict]:
    """Fit GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1); return results with BIC ranking.

    Returns dict keyed by model name, each containing:
        'cond_vol' — conditional volatility series (annualized %)
        'forecast_vol' — forecasted vol array (annualized %)
        'forecast_start' — first forecast date
        'params' — fitted params dict
        'aic', 'bic' — information criteria
        'converged' — bool
    """
    from arch import arch_model

    pct_returns = returns_pct.dropna()
    if len(pct_returns) < 100:
        raise ValueError(f"Not enough data: {len(pct_returns)} points.")

    models = {
        "GARCH(1,1)": {"vol": "Garch", "kwargs": {}},
        "EGARCH(1,1)": {"vol": "EGARCH", "kwargs": {}},
        "GJR-GARCH(1,1)": {"vol": "Garch", "kwargs": {"o": 1}},
    }

    results = {}

    for name, spec in models.items():
        try:
            am = arch_model(
                pct_returns,
                vol=spec["vol"], p=1, q=1,
                mean="Constant", dist="t", rescale=False,
                **spec["kwargs"],
            )
            res = am.fit(disp="off")
            cond_vol = res.conditional_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

            fc = res.forecast(horizon=forecast_horizon)
            var_next = fc.variance.iloc[-1].to_numpy()
            vol_next = np.sqrt(var_next) * np.sqrt(TRADING_DAYS_PER_YEAR)

            forecast_start = pct_returns.index[-1] + pd.Timedelta(days=1)

            results[name] = {
                "cond_vol": cond_vol,
                "forecast_vol": vol_next,
                "forecast_start": forecast_start,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "converged": True,
                "params": {k: v for k, v in res.params.items()},
            }
        except Exception as e:
            results[name] = {
                "cond_vol": None, "forecast_vol": None,
                "forecast_start": None,
                "aic": np.inf, "bic": np.inf,
                "converged": False,
                "params": {},
                "error": str(e)[:200],
            }

    return results


def garch_regime_labels(
    cond_vol: pd.Series, high_vol_pct: float = 70.0
) -> pd.Series:
    """Label each day as 0 (low-vol) or 1 (high-vol) based on conditional vol percentile.

    Args:
        cond_vol: annualized conditional volatility series
        high_vol_pct: percentile threshold above which regime is "high vol"
    Returns:
        Boolean Series (True = high vol regime)
    """
    threshold = np.percentile(cond_vol.dropna(), high_vol_pct)
    return cond_vol >= threshold
