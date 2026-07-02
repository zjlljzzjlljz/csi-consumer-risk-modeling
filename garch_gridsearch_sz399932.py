#!/usr/bin/env python3
"""GARCH(p,q) grid-search validation for CSI Consumer Index (sz399932).

Confirms that GARCH(1,1) is the optimal lag specification within p,q <= 3.
"""

from __future__ import annotations

import inspect
from datetime import datetime

import numpy as np
import pandas as pd

from modules.core import CSI_CONSUMER_SYMBOL, fetch_index_daily_cached


def fit_garch_grid(returns_pct: pd.Series, p_values: list[int], q_values: list[int]) -> list[dict]:
    """Fit GARCH(p,q) over the given grid; return a list of result dicts."""
    from arch import arch_model

    arch_model_sig = inspect.signature(arch_model)

    results = []
    for p in p_values:
        for q in q_values:
            try:
                am_kwargs = dict(vol="Garch", p=p, q=q, mean="Constant", dist="Normal")
                if "rescale" in arch_model_sig.parameters:
                    am_kwargs["rescale"] = False

                am = arch_model(returns_pct, **am_kwargs)

                fit_sig = inspect.signature(am.fit)
                fit_kwargs = {"disp": "off"}
                if "show_warning" in fit_sig.parameters:
                    fit_kwargs["show_warning"] = False
                if "options" in fit_sig.parameters:
                    fit_kwargs["options"] = {"maxiter": 300}

                res = am.fit(**fit_kwargs)

                results.append({
                    "p": p, "q": q,
                    "AIC": float(getattr(res, "aic")),
                    "BIC": float(getattr(res, "bic")),
                    "convergence": getattr(res, "convergence", np.nan),
                })
            except Exception as e:
                results.append({
                    "p": p, "q": q,
                    "AIC": np.nan, "BIC": np.nan, "convergence": np.nan,
                    "error": str(e)[:250],
                })
    return results


def main() -> int:
    symbol_sz = CSI_CONSUMER_SYMBOL
    start_date = "20100101"
    end_date = datetime.today().strftime("%Y%m%d")

    p_values = [1, 2, 3]
    q_values = [1, 2, 3]

    prices = fetch_index_daily_cached(symbol=symbol_sz, start_date=start_date, end_date=end_date)
    prices = prices.sort_values("date")
    close = prices["close"]
    dates = prices["date"]

    log_returns = np.log(close).diff()
    returns_pct = (log_returns * 100.0).dropna()
    returns_pct.index = dates.loc[returns_pct.index].values

    if len(returns_pct) < 200:
        raise ValueError(f"Not enough data points for GARCH grid search: {len(returns_pct)}")

    results = fit_garch_grid(returns_pct=returns_pct, p_values=p_values, q_values=q_values)
    success_rows = [
        r for r in results
        if (r.get("BIC") is not None and np.isfinite(r.get("BIC", np.nan)))
        and (r.get("AIC") is not None and np.isfinite(r.get("AIC", np.nan)))
    ]

    if not success_rows:
        raise RuntimeError("All GARCH models in the grid failed.")

    success_rows.sort(key=lambda r: float(r["BIC"]))

    print(f"\nGARCH Grid Search (Constant Mean; {symbol_sz}; returns scaled by 100)\n")
    print(f"Grid: p in {p_values}, q in {q_values} | Successful models: {len(success_rows)}/{len(results)}\n")

    header = f"{'rank':>4} | {'p':>2} | {'q':>2} | {'AIC':>14} | {'BIC':>14} | {'convergence':>13}"
    print("Ranked by Lowest BIC (best -> worst):")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(success_rows, start=1):
        conv = r.get("convergence", np.nan)
        conv_str = "nan" if conv is None or not np.isfinite(conv) else str(int(conv))
        print(f"{i:>4} | {int(r['p']):>2} | {int(r['q']):>2} | {float(r['AIC']):>14.4f} | {float(r['BIC']):>14.4f} | {conv_str:>13}")

    best = success_rows[0]
    best_pq = (int(best["p"]), int(best["q"]))
    is_11_best = best_pq == (1, 1)
    print(f"\nBest by BIC: (p,q)=({best_pq[0]},{best_pq[1]}) | (1,1) is best? {is_11_best}")
    print("(1,1) exists in fitted set?"
          + (" yes" if any(int(r["p"]) == 1 and int(r["q"]) == 1 for r in success_rows) else " no"))

    failed_rows = [r for r in results if np.isnan(r.get("BIC", np.nan)) and np.isnan(r.get("AIC", np.nan))]
    if failed_rows:
        print("\nFailed models (brief error):")
        for r in failed_rows:
            print(f"  (p,q)=({int(r['p'])},{int(r['q'])}) -> {r.get('error', 'unknown error')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
