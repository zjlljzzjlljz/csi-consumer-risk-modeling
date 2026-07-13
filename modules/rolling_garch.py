#!/usr/bin/env python3
"""Rolling-window GARCH(1,1) parameter stability analysis for CSI Consumer Index.

Answers the interview question: "How stable are alpha[1] and beta[1] over time?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.core import CSI_CONSUMER_SYMBOL, TRADING_DAYS_PER_YEAR, fetch_index_daily

ROLLING_WINDOW_TRADING_DAYS = 1008  # ~4 years
STEP_TRADING_DAYS = 63  # ~quarterly recalibration


def rolling_garch_estimate(
    returns_pct: pd.Series, window: int, step: int
) -> pd.DataFrame:
    """Fit GARCH(1,1) on overlapping rolling windows; return parameter time series."""
    from arch import arch_model

    n = len(returns_pct)
    rows: list[dict] = []
    start = 0

    while start + window <= n:
        subset = returns_pct.iloc[start : start + window]
        end_date = subset.index[-1]
        try:
            am = arch_model(subset, vol="Garch", p=1, q=1, mean="Constant", dist="t", rescale=False)
            res = am.fit(disp="off")
            rows.append({
                "end_date": end_date,
                "mu": res.params.get("mu", np.nan),
                "omega": res.params.get("omega", np.nan),
                "alpha[1]": res.params.get("alpha[1]", np.nan),
                "beta[1]": res.params.get("beta[1]", np.nan),
                "persistence": res.params.get("alpha[1]", np.nan) + res.params.get("beta[1]", np.nan),
                "last_cond_vol_annual": float(res.conditional_volatility.iloc[-1]) * np.sqrt(TRADING_DAYS_PER_YEAR),
            })
        except Exception as e:
            print(f"  Rolling fit failed at window ending {end_date.date()}: {e}")
        start += step

    return pd.DataFrame(rows).set_index("end_date")


def plot_rolling_params(roll_df: pd.DataFrame, output_name: str = "garch_rolling_params.png") -> Path:
    """Plot rolling alpha[1], beta[1], persistence, and conditional vol."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    param_colors = {"alpha[1]": "#1f77b4", "beta[1]": "#ff7f0e", "persistence": "#2ca02c"}

    ax0 = axes[0]
    for param, color in [("alpha[1]", param_colors["alpha[1]"]), ("beta[1]", param_colors["beta[1]"])]:
        ax0.plot(roll_df.index, roll_df[param], linewidth=1.3, label=param, color=color)
    ax0.set_ylabel("Parameter Value")
    ax0.set_title("Rolling GARCH(1,1): alpha[1] & beta[1] (4-year windows, quarterly step)")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    ax1.plot(roll_df.index, roll_df["persistence"], linewidth=1.3,
             color=param_colors["persistence"], label="alpha[1] + beta[1]")
    ax1.axhline(1.0, linestyle="--", linewidth=1.0, color="red", alpha=0.6, label="Unit Root")
    ax1.set_ylabel("Persistence")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)

    ax2 = axes[2]
    ax2.plot(roll_df.index, roll_df["omega"], linewidth=1.3, color="#d62728")
    ax2.set_ylabel("omega (×1e-4)")
    ax2.set_title("omega (long-run variance floor)")
    ax2.grid(True, alpha=0.25)

    ax3 = axes[3]
    ax3.plot(roll_df.index, roll_df["last_cond_vol_annual"], linewidth=1.3, color="#9467bd")
    ax3.set_ylabel("Annualized Vol (%)")
    ax3.set_title("Terminal Conditional Volatility (annualized %)")
    ax3.set_xlabel("Window End Date")
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.grid(True, alpha=0.25)

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent.parent / output_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Rolling GARCH chart saved to: {output_path}")
    plt.show()
    return output_path


def main() -> None:
    raw = fetch_index_daily(CSI_CONSUMER_SYMBOL, "2005-01-01")
    prices = raw.set_index("date")["close"]

    log_returns = np.log(prices).diff()
    returns_pct = (log_returns * 100.0).dropna()

    print(f"Fitting rolling GARCH(1,1) on {len(returns_pct)} daily returns...")
    print(f"  Window: {ROLLING_WINDOW_TRADING_DAYS} days (~4 years)")
    print(f"  Step:   {STEP_TRADING_DAYS} days (~1 quarter)\n")

    roll_df = rolling_garch_estimate(returns_pct, ROLLING_WINDOW_TRADING_DAYS, STEP_TRADING_DAYS)

    print(f"Total rolling fits: {len(roll_df)}")
    print("\nRolling Parameter Summary:")
    print(roll_df[["alpha[1]", "beta[1]", "persistence"]].describe().to_string())

    plot_rolling_params(roll_df)


if __name__ == "__main__":
    main()
