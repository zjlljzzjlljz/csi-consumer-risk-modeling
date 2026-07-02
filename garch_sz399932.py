import sys
import os
import contextlib
import io
import time
from datetime import datetime

import numpy as np
try:
    # Suppress noisy compiled-extension warnings during import (environment-related).
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import `pandas`.\n"
        "This usually indicates a NumPy compatibility problem (e.g. numpy>=2 with pandas built for numpy<2).\n\n"
        "Recommended fix:\n"
        "  pip install 'numpy<2' --force-reinstall\n"
        "  pip install -U pandas numexpr bottleneck --force-reinstall"
    ) from e


def fetch_sz399932_daily_close(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily close prices for CSI Consumer Index (sz399932) via AKShare.

    Notes
    - AKShare's `index_zh_a_hist` expects the index code without the `sz/SH` prefix.
      For `sz399932` we pass `399932`.
    - The returned DataFrame is expected to contain columns: `日期`, `收盘`.
    """
    try:
        import akshare as ak  # local import so the script fails with a clear error message
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `akshare`. Install it first, e.g. `pip install akshare`."
        ) from e

    symbol_sz = "sz399932"
    # AKShare's `index_zh_a_hist` expects numeric index code (without market prefix)
    symbol_num = symbol_sz[2:] if symbol_sz.startswith(("sz", "sh")) else symbol_sz

    cache_file = os.path.join(os.path.dirname(__file__), "sz399932_akshare_cache.csv")

    def load_cache():
        if not os.path.exists(cache_file):
            return None
        try:
            cached = pd.read_csv(cache_file)
            # Accept either cached Chinese columns or English columns
            if "date" in cached.columns and "close" in cached.columns:
                cached = cached.rename(columns={"date": "date", "close": "close"})
            elif "日期" in cached.columns and "收盘" in cached.columns:
                cached = cached.rename(columns={"日期": "date", "收盘": "close"})
            else:
                return None

            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached["close"] = pd.to_numeric(cached["close"], errors="coerce")
            cached = cached.dropna(subset=["date", "close"]).sort_values("date")
            s = pd.to_datetime(start_date, format="%Y%m%d")
            e = pd.to_datetime(end_date, format="%Y%m%d")
            cached = cached[(cached["date"] >= s) & (cached["date"] <= e)]
            if cached.empty:
                return None
            cached = cached.set_index("date").sort_index()
            return cached[["close"]]
        except Exception:
            return None

    cached_df = load_cache()

    def fetch_with_retries(fetch_fn, max_retries: int = 5, base_sleep_s: float = 2.0):
        last_err = None
        for attempt in range(max_retries):
            try:
                return fetch_fn()
            except Exception as e:
                last_err = e
                sleep_s = base_sleep_s * (2**attempt)
                print(
                    f"AKShare fetch failed (attempt {attempt+1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                time.sleep(sleep_s)
        raise last_err  # should not reach

    def parse_df(df_in):
        cols = set(df_in.columns)
        # Main interface: 日期/收盘
        if {"日期", "收盘"}.issubset(cols):
            date_col, close_col = "日期", "收盘"
        # Fallback interface: date/close
        elif {"date", "close"}.issubset(cols):
            date_col, close_col = "date", "close"
        else:
            raise ValueError(f"AKShare response missing required columns. Got: {list(df_in.columns)}")

        out = df_in.loc[:, [date_col, close_col]].copy()
        out["date"] = pd.to_datetime(out[date_col], errors="coerce")
        out["close"] = pd.to_numeric(out[close_col], errors="coerce")
        out = out.dropna(subset=["date", "close"]).sort_values("date")
        out = out.set_index("date").sort_index()
        if out.empty:
            raise ValueError("After cleaning, no valid (date, close) rows remain.")
        return out[["close"]]

    # 1) Primary fetch with index_zh_a_hist
    try:
        def primary_fetch():
            return ak.index_zh_a_hist(
                symbol=symbol_num,
                period="daily",
                start_date=start_date,
                end_date=end_date,
            )

        df = fetch_with_retries(primary_fetch, max_retries=5, base_sleep_s=2.0)
        out = parse_df(df)
    except Exception as e:
        # 2) Fallback fetch
        try:
            def fallback_fetch():
                return ak.stock_zh_index_daily(symbol=symbol_sz)

            df = fetch_with_retries(fallback_fetch, max_retries=3, base_sleep_s=1.5)
            out = parse_df(df)
        except Exception:
            if cached_df is not None:
                print(
                    f"[Warning] AKShare fetch failed; using cached data from {cache_file}."
                )
                return cached_df
            raise RuntimeError(
                "Failed to fetch sz399932 data from AKShare (primary + fallback).\n"
                "Please retry later or check network/AKShare availability."
            ) from e

    # Update cache for future runs
    try:
        # store with simple English columns to simplify future parsing
        to_save = out.reset_index().rename(columns={"date": "date", "close": "close"})
        to_save.to_csv(cache_file, index=False)
    except Exception:
        pass

    return out


def main() -> int:
    symbol_sz = "sz399932"
    start_date = "20100101"
    end_date = datetime.today().strftime("%Y%m%d")
    horizon = 30
    trading_days_per_year = 252

    # 1) Data prep
    prices = fetch_sz399932_daily_close(start_date=start_date, end_date=end_date)
    # prices index is already datetime (from fetch)
    close = prices["close"]

    # daily log returns
    log_returns = np.log(close).diff()

    # Numerical stability: multiply log returns by 100 before fitting
    returns_pct = log_returns * 100.0
    model_input = returns_pct.dropna()

    if len(model_input) < 100:
        raise ValueError(f"Not enough data points for GARCH fitting: {len(model_input)}")

    # 2) GARCH(1,1) modeling with Constant Mean
    try:
        from arch import arch_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `arch`. Install it first, e.g. `pip install arch`."
        ) from e

    am = arch_model(
        model_input,
        vol="Garch",
        p=1,
        q=1,
        mean="Constant",
        dist="Normal",
    )
    res = am.fit(disp="off")

    # 3) Volatility extraction (annualized)
    # `arch` preserves the input DatetimeIndex on `conditional_volatility`.
    cond_vol_annual = res.conditional_volatility * np.sqrt(trading_days_per_year)

    # 4) Forecasting (expected annualized volatility for next 30 trading days)
    fc = res.forecast(horizon=horizon)
    # fc.variance is a (nobs, horizon) DataFrame; we take the last row
    var_next = fc.variance.iloc[-1].to_numpy()
    vol_next_annual = np.sqrt(var_next) * np.sqrt(trading_days_per_year)

    # Trading day calendar differs from generic business days; we use B-days for plotting.
    last_date = model_input.index[-1]
    forecast_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="B")
    forecast_annual_vol = pd.Series(vol_next_annual, index=forecast_dates, name="forecast_annual_vol_pct")

    # 5) Print output: summary + alpha[1]/beta[1]
    print(f"\nGARCH(1,1) on {symbol_sz} (Constant Mean; returns scaled by 100 for stability)\n")
    print(res.summary())
    alpha1 = res.params.get("alpha[1]", np.nan)
    beta1 = res.params.get("beta[1]", np.nan)
    print(f"\nalpha[1] (shock effect): {alpha1}")
    print(f"beta[1] (persistence effect): {beta1}")

    print("\n30-day forecast of annualized conditional volatility (%):")
    print(forecast_annual_vol.to_string())

    # 6) Visualization: two clean subplots
    import matplotlib.pyplot as plt

    abs_daily_returns_pct = np.abs(model_input)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    ax_top.plot(model_input.index, abs_daily_returns_pct, linewidth=1.0)
    ax_top.set_ylabel("Abs. Log Return (%)")
    ax_top.set_title(f"Volatility Clustering - GARCH(1,1) on {symbol_sz}")
    ax_top.grid(True, alpha=0.2)

    # Plot estimated annualized conditional volatility (using its own DatetimeIndex when available)
    if hasattr(cond_vol_annual, "index"):
        x_cond = cond_vol_annual.index
        y_cond = cond_vol_annual
    else:
        # Fallback: if arch returns a numpy array, align length to model_input index.
        y_arr = np.asarray(cond_vol_annual)
        x_cond = model_input.index[-len(y_arr) :]
        y_cond = y_arr

    ax_bottom.plot(x_cond, y_cond, linewidth=1.3, label="Estimated Annualized Conditional Volatility (%)")

    # Highlight stress periods (e.g., 2015, 2021)
    for yr in [2015, 2021]:
        start = pd.Timestamp(f"{yr}-01-01")
        end = pd.Timestamp(f"{yr}-12-31")
        ax_bottom.axvspan(start, end, alpha=0.15)

    # Optional overlay: 30-day forecast
    ax_bottom.plot(
        forecast_dates,
        forecast_annual_vol,
        linestyle="--",
        linewidth=1.2,
        label="30-Day Forecast (Annualized, %)",
    )

    ax_bottom.set_ylabel("Annualized Conditional Volatility (%)")
    ax_bottom.grid(True, alpha=0.2)
    ax_bottom.legend(loc="upper left")

    fig.tight_layout()
    out_file = "sz399932_volatility.png"
    fig.savefig(out_file, dpi=200)
    print(f"\nSaved plot to: {out_file}\n")
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

