from __future__ import annotations

import inspect
import time
from datetime import datetime
import contextlib
import io
import os

import numpy as np


def fetch_sz399932_daily_close(start_date: str, end_date: str):
    """
    Fetch daily close prices for CSI Consumer Index (sz399932) via AKShare.

    Expected columns from AKShare:
    - `日期`
    - `收盘`
    """
    if int(np.__version__.split(".")[0]) >= 2:
        print(
            f"[Warning] Detected numpy {np.__version__}. If pandas/arch fail, you likely need to downgrade to numpy<2 "
            "and reinstall compiled deps."
        )

    try:
        # Suppress noisy compiled-extension warnings during import;
        # we will surface a clean actionable message if it truly fails.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import `pandas` in your environment.\n"
            "This commonly happens when your environment has `numpy>=2` but compiled extensions "
            "(pandas/numexpr/bottleneck/arch) are built for NumPy 1.x.\n\n"
            "Fix (recommended):\n"
            "  pip install 'numpy<2' --force-reinstall\n"
            "  pip install -U pandas numexpr bottleneck arch --force-reinstall"
        ) from e

    try:
        import akshare as ak
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `akshare`. Install it first, e.g. `pip install akshare`."
        ) from e

    symbol_sz = "sz399932"
    # `index_zh_a_hist` expects numeric index code (without market prefix)
    symbol_num = symbol_sz[2:] if symbol_sz.startswith(("sz", "sh")) else symbol_sz

    cache_file = os.path.join(os.path.dirname(__file__), "sz399932_akshare_cache.csv")

    def try_load_cache():
        if not os.path.exists(cache_file):
            return None
        try:
            cached = pd.read_csv(cache_file)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached["close"] = pd.to_numeric(cached["close"], errors="coerce")
            cached = cached.dropna(subset=["date", "close"]).sort_values("date")
            s = pd.to_datetime(start_date, format="%Y%m%d")
            e = pd.to_datetime(end_date, format="%Y%m%d")
            cached = cached[(cached["date"] >= s) & (cached["date"] <= e)]
            if cached.empty:
                return None
            return cached.loc[:, ["date", "close"]].copy()
        except Exception:
            return None

    # 先尝试从缓存出发（可减少你遇到网络故障时的阻塞）
    cached_df = try_load_cache()
    # 但仍会优先尝试 AKShare 拉取更新数据（如果网络正常）。

    # AKShare sometimes fails with network errors (RemoteDisconnected, ConnectionError, ...).
    # Retry to reduce spurious failures.
    def fetch_with_retries(fetch_fn, max_retries=5, base_sleep_s=2.0):
        for attempt in range(max_retries):
            try:
                return fetch_fn()
            except Exception as e:
                sleep_s = base_sleep_s * (2**attempt)
                print(f"AKShare fetch failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(sleep_s)

    def fetch_primary():
        return ak.index_zh_a_hist(
            symbol=symbol_num,
            period="daily",
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_fallback():
        # Alternate AKShare endpoint; still use AKShare per requirement.
        # Expected to include columns like `日期` and `收盘`.
        return ak.stock_zh_index_daily(symbol=symbol_sz)

    df = None
    last_err = None
    try:
        df = fetch_with_retries(fetch_primary)
    except Exception as e:
        last_err = e
        # Try fallback endpoint if primary fails.
        try:
            df = fetch_with_retries(fetch_fallback, max_retries=3, base_sleep_s=1.5)
        except Exception:
            df = None

    if (df is None) or (df is None or df.empty):
        if cached_df is not None and len(cached_df) > 50:
            print(
                f"[Warning] AKShare fetch failed ({type(last_err).__name__ if last_err else 'unknown'}). "
                f"Using cached data from {cache_file}."
            )
            return cached_df
        raise RuntimeError(
            "Failed to fetch sz399932 data from AKShare (and no usable cache found). "
            "Please retry later, or fix your network / AKShare availability."
        ) from last_err

    if df is None or df.empty:
        raise ValueError("AKShare returned empty data for sz399932.")

    # AKShare 不同接口返回的列名可能不同：主接口是 `日期/收盘`，备用接口可能是 `date/close`
    cols = set(df.columns)
    if {"日期", "收盘"}.issubset(cols):
        date_col, close_col = "日期", "收盘"
    elif {"date", "close"}.issubset(cols):
        date_col, close_col = "date", "close"
    else:
        raise ValueError(
            "AKShare response missing required date/close columns. "
            f"Got columns: {list(df.columns)}"
        )

    out = df.loc[:, [date_col, close_col]].copy()
    out = out.rename(columns={date_col: "date", close_col: "close"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.loc[:, ["date", "close"]]

    if out.empty:
        raise ValueError("After cleaning, no valid (date, close) rows remain.")

    # Update cache for future runs.
    try:
        out.to_csv(cache_file, index=False)
    except Exception:
        pass
    return out


def fit_garch_grid(returns_pct, p_values, q_values):
    """
    Fit GARCH(p,q) over the given grid; return a list of result dicts.
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from arch import arch_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import `arch`.\n"
            "This is often caused by the same NumPy 1.x vs 2.x compiled-extension mismatch.\n"
            "Try: downgrade numpy to `numpy<2` and reinstall `arch`, `pandas`, and compiled deps."
        ) from e

    # Try to keep numeric behavior stable across arch versions.
    arch_model_sig = inspect.signature(arch_model)
    fit_params_cache = None

    results = []
    for p in p_values:
        for q in q_values:
            try:
                am_kwargs = dict(
                    vol="Garch",
                    p=p,
                    q=q,
                    mean="Constant",
                    dist="Normal",
                )
                # If supported, disable internal rescaling because we already scale by 100.
                if "rescale" in arch_model_sig.parameters:
                    am_kwargs["rescale"] = False

                am = arch_model(returns_pct, **am_kwargs)

                fit_sig = inspect.signature(am.fit)
                fit_kwargs = {"disp": "off"}
                if "show_warning" in fit_sig.parameters:
                    fit_kwargs["show_warning"] = False
                if "cov_type" in fit_sig.parameters:
                    # Keep defaults; just avoid surprising covariance warnings where applicable.
                    pass

                # Performance/safety: cap iterations to avoid extremely long optimizer runs.
                # If a model fails to converge, we skip it in the outer try/except.
                if "options" in fit_sig.parameters:
                    fit_kwargs["options"] = {"maxiter": 300}

                res = am.fit(**fit_kwargs)

                aic = float(getattr(res, "aic"))
                bic = float(getattr(res, "bic"))

                results.append(
                    {
                        "p": p,
                        "q": q,
                        "AIC": aic,
                        "BIC": bic,
                        "convergence": getattr(res, "convergence", np.nan),
                    }
                )
            except Exception as e:
                # Some combinations fail to converge; we skip them.
                results.append(
                    {
                        "p": p,
                        "q": q,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        "convergence": np.nan,
                        "error": str(e)[:250],
                    }
                )
    return results


def main() -> int:
    symbol_sz = "sz399932"
    start_date = "20100101"
    end_date = datetime.today().strftime("%Y%m%d")

    p_values = [1, 2, 3]
    q_values = [1, 2, 3]

    # 1) Data prep
    prices = fetch_sz399932_daily_close(start_date=start_date, end_date=end_date)
    close = prices["close"]
    dates = prices["date"]

    # Daily log returns * 100 (percentage terms) for numerical stability
    log_returns = np.log(close).diff()
    returns_pct = (log_returns * 100.0).dropna()
    # Keep the index aligned so `arch` can attach its own dates if needed
    returns_pct.index = dates.loc[returns_pct.index].values

    if len(returns_pct) < 200:
        raise ValueError(f"Not enough data points for GARCH grid search: {len(returns_pct)}")

    # 2) Grid search
    results = fit_garch_grid(returns_pct=returns_pct, p_values=p_values, q_values=q_values)
    success_rows = [
        r for r in results if (r.get("BIC") is not None and np.isfinite(r.get("BIC", np.nan)))
        and (r.get("AIC") is not None and np.isfinite(r.get("AIC", np.nan)))
    ]

    if not success_rows:
        raise RuntimeError("All GARCH models in the grid failed. Try different p/q ranges or fix your environment.")

    success_rows.sort(key=lambda r: float(r["BIC"]))

    # 3) Print summary table (rank by lowest BIC)
    print(f"\nGARCH Grid Search (Constant Mean; {symbol_sz}; returns scaled by 100)\n")
    print(
        f"Grid: p in {p_values}, q in {q_values} | Successful models: {len(success_rows)}/{len(results)}\n"
    )

    header = f"{'rank':>4} | {'p':>2} | {'q':>2} | {'AIC':>14} | {'BIC':>14} | {'convergence':>13}"
    print("Ranked by Lowest BIC (best -> worst):")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(success_rows, start=1):
        conv = r.get("convergence", np.nan)
        conv_str = "nan" if conv is None or not np.isfinite(conv) else str(int(conv))
        print(
            f"{i:>4} | {int(r['p']):>2} | {int(r['q']):>2} | {float(r['AIC']):>14.4f} | {float(r['BIC']):>14.4f} | {conv_str:>13}"
        )

    best = success_rows[0]
    best_pq = (int(best["p"]), int(best["q"]))
    is_11_best = best_pq == (1, 1)
    print(f"\nBest by BIC: (p,q)=({best_pq[0]},{best_pq[1]}) | (1,1) is best? {is_11_best}")
    print(
        "(1,1) exists in fitted set?"
        + (" yes" if any(int(r["p"]) == 1 and int(r["q"]) == 1 for r in success_rows) else " no")
    )

    # 4) Optional: print failures (briefly) for transparency
    failed_rows = [r for r in results if np.isnan(r.get("BIC", np.nan)) and np.isnan(r.get("AIC", np.nan))]
    if failed_rows:
        print("\nFailed models (brief error):")
        for r in failed_rows:
            print(f"  (p,q)=({int(r['p'])},{int(r['q'])}) -> {r.get('error', 'unknown error')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

