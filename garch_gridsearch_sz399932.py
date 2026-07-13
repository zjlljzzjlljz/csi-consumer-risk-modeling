#!/usr/bin/env python3
"""Volatility model grid-search & asymmetric-model comparison for CSI Consumer Index (sz399932).

Phase 1: GARCH(p,q) grid search over p,q ∈ {1,2,3} → confirms (1,1) optimal.
Phase 2: GARCH(1,1) vs EGARCH(1,1) vs GJR-GARCH(1,1) → captures leverage effect.
"""

from __future__ import annotations

import inspect
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.core import (
    CSI_CONSUMER_SYMBOL,
    TRADING_DAYS_PER_YEAR,
    fetch_index_daily_cached,
    fit_asymmetric_vol_models,
)


def fit_garch_grid(returns_pct: pd.Series, p_values: list[int], q_values: list[int]) -> list[dict]:
    """Fit GARCH(p,q) over the given grid; return list of result dicts."""
    from arch import arch_model

    arch_model_sig = inspect.signature(arch_model)

    results = []
    for p in p_values:
        for q in q_values:
            try:
                am_kwargs = dict(vol="Garch", p=p, q=q, mean="Constant", dist="t")
                if "rescale" in arch_model_sig.parameters:
                    am_kwargs["rescale"] = False

                am = arch_model(returns_pct, **am_kwargs)
                fit_kwargs = {"disp": "off"}
                fit_sig = inspect.signature(am.fit)
                if "show_warning" in fit_sig.parameters:
                    fit_kwargs["show_warning"] = False
                if "options" in fit_sig.parameters:
                    fit_kwargs["options"] = {"maxiter": 300}

                res = am.fit(**fit_kwargs)
                results.append({
                    "p": p, "q": q,
                    "AIC": float(res.aic), "BIC": float(res.bic),
                    "convergence": getattr(res, "convergence", np.nan),
                })
            except Exception as e:
                results.append({
                    "p": p, "q": q,
                    "AIC": np.nan, "BIC": np.nan, "convergence": np.nan,
                    "error": str(e)[:250],
                })
    return results


def plot_asymmetric_news_impact(results: dict[str, dict]) -> str:
    """Plot conditional volatility comparison: GARCH vs EGARCH vs GJR-GARCH."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    first_model = next(iter(results.values()))
    cond_vol_0 = first_model.get("cond_vol")
    if cond_vol_0 is None or not hasattr(cond_vol_0, "index"):
        return ""

    colors = {"GARCH(1,1)": "#1f77b4", "EGARCH(1,1)": "#ff7f0e", "GJR-GARCH(1,1)": "#2ca02c"}

    ax0 = axes[0]
    for name, r in results.items():
        if r["converged"] and r["cond_vol"] is not None:
            ax0.plot(r["cond_vol"].index, r["cond_vol"], linewidth=1.0, alpha=0.85,
                     label=f'{name} (BIC={r["bic"]:.2f})', color=colors.get(name))

    ax0.set_title("Conditional Volatility: GARCH vs EGARCH vs GJR-GARCH (CSI Consumer)")
    ax0.set_ylabel("Annualized Vol (%)")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    for name, r in results.items():
        if r["converged"] and r["forecast_vol"] is not None:
            forecast_start = r["forecast_start"]
            if forecast_start is not None:
                fdates = pd.bdate_range(start=forecast_start, periods=len(r["forecast_vol"]), freq="B")
                ax1.plot(fdates, r["forecast_vol"], linewidth=1.3, alpha=0.85,
                         label=name, color=colors.get(name))

    ax1.set_title("Forward Volatility Forecast Comparison")
    ax1.set_ylabel("Forecast Annualized Vol (%)")
    ax1.set_xlabel("Date")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    out_path = "garch_asymmetric_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Asymmetric vol comparison chart saved to: {out_path}")
    plt.show()
    return out_path


def main() -> int:
    symbol_sz = CSI_CONSUMER_SYMBOL
    start_date = "20100101"
    end_date = datetime.today().strftime("%Y%m%d")

    prices = fetch_index_daily_cached(symbol=symbol_sz, start_date=start_date, end_date=end_date)
    prices = prices.sort_values("date")
    close = prices["close"]
    dates = prices["date"]

    log_returns = np.log(close).diff()
    returns_pct = (log_returns * 100.0).dropna()
    returns_pct.index = dates.loc[returns_pct.index].values

    if len(returns_pct) < 200:
        raise ValueError(f"Not enough data points: {len(returns_pct)}")

    # ===== Phase 1: GARCH(p,q) grid search =====
    p_values = [1, 2, 3]
    q_values = [1, 2, 3]

    grid_results = fit_garch_grid(returns_pct=returns_pct, p_values=p_values, q_values=q_values)
    success_rows = [
        r for r in grid_results
        if (r.get("BIC") is not None and np.isfinite(r.get("BIC", np.nan)))
        and (r.get("AIC") is not None and np.isfinite(r.get("AIC", np.nan)))
    ]

    if not success_rows:
        raise RuntimeError("All GARCH models in the grid failed.")

    success_rows.sort(key=lambda r: float(r["BIC"]))

    print(f"\nGARCH Grid Search (Constant Mean; {symbol_sz}; returns scaled ×100)\n")
    print(f"Grid: p ∈ {p_values}, q ∈ {q_values} | Converged: {len(success_rows)}/{len(grid_results)}\n")

    header = f"{'rank':>4} | {'p':>2} | {'q':>2} | {'AIC':>14} | {'BIC':>14} | {'conv':>5}"
    print("Ranked by Lowest BIC:")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(success_rows, start=1):
        conv = r.get("convergence", np.nan)
        conv_str = "nan" if conv is None or not np.isfinite(conv) else str(int(conv))
        print(f"{i:>4} | {int(r['p']):>2} | {int(r['q']):>2} | {float(r['AIC']):>14.4f} | {float(r['BIC']):>14.4f} | {conv_str:>5}")

    best = success_rows[0]
    print(f"\nBest by BIC: (p,q)=({int(best['p'])},{int(best['q'])}) | (1,1) best? {int(best['p'])==1 and int(best['q'])==1}")

    # ===== Phase 2: Asymmetric model comparison =====
    print("\n" + "=" * 70)
    print("Phase 2: GARCH vs EGARCH vs GJR-GARCH (1,1)")
    print("=" * 70)

    asym_results = fit_asymmetric_vol_models(returns_pct, forecast_horizon=60)

    ranked = sorted(asym_results.items(), key=lambda x: x[1]["bic"])
    print(f"\n{'Model':<20} {'AIC':>12} {'BIC':>12} {'Converged':>10}")
    print("-" * 56)
    for name, r in ranked:
        status = "YES" if r["converged"] else "NO"
        aic_str = f"{r['aic']:,.2f}" if np.isfinite(r['aic']) else "N/A"
        bic_str = f"{r['bic']:,.2f}" if np.isfinite(r['bic']) else "N/A"
        print(f"{name:<20} {aic_str:>12} {bic_str:>12} {status:>10}")

    best_asym = ranked[0]
    print(f"\nBest asymmetric model by BIC: {best_asym[0]} (BIC={best_asym[1]['bic']:,.2f})")

    # Print leverage parameter if present
    for name, r in asym_results.items():
        if r["converged"]:
            params = r.get("params", {})
            if "gamma[1]" in params:
                print(f"  {name} gamma[1] (leverage): {params['gamma[1]']:.4f}")
            if "alpha[1]" in params:
                print(f"  {name} alpha[1]: {params['alpha[1]']:.4f}  beta[1]: {params.get('beta[1]', np.nan):.4f}")

    plot_asymmetric_news_impact(asym_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
