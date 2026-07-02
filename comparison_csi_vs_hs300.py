#!/usr/bin/env python3
"""Compare CSI Consumer Index (sz399932) vs CSI 300 Index (sh000300) with AkShare."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=r".*Pandas requires version '.*' or newer of 'numexpr'.*")
warnings.filterwarnings("ignore", message=r".*Pandas requires version '.*' or newer of 'bottleneck'.*")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from modules.core import CSI_CONSUMER_SYMBOL, HS300_SYMBOL, fetch_index_daily


START_DATE = "2005-01-01"


def build_aligned_prices(start_date: str) -> pd.DataFrame:
    """Build aligned price DataFrame using date intersection."""
    consumer = fetch_index_daily(CSI_CONSUMER_SYMBOL, start_date)
    hs300 = fetch_index_daily(HS300_SYMBOL, start_date)
    consumer = consumer.set_index("date")["close"].rename("CSI Consumer Index (sz399932)")
    hs300 = hs300.set_index("date")["close"].rename("CSI 300 Index (sh000300)")
    prices = pd.concat([consumer, hs300], axis=1, join="inner").dropna(how="any")
    if prices.empty:
        raise RuntimeError("No overlapping dates between the two indices after alignment.")
    return prices


def calculate_metrics(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Calculate normalized performance, drawdown, and summary statistics."""
    normalized = prices / prices.iloc[0]
    drawdown = normalized / normalized.cummax() - 1.0
    total_return = normalized.iloc[-1] - 1.0
    max_drawdown = drawdown.min()

    years = (normalized.index[-1] - normalized.index[0]).days / 365.25
    annualized_return = (normalized.iloc[-1] ** (1.0 / years)) - 1.0
    return normalized, drawdown, total_return, annualized_return, max_drawdown


def plot_comparison(normalized: pd.DataFrame, drawdown: pd.DataFrame) -> None:
    """Plot normalized performance and drawdown in a clean business style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    normalized.plot(ax=axes[0], linewidth=2.0)
    axes[0].set_title("Normalized Performance Comparison: CSI Consumer vs CSI 300", fontsize=12)
    axes[0].set_ylabel("Normalized Price")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    drawdown.plot(ax=axes[1], linewidth=1.8)
    axes[1].set_title("Drawdown Curves (2015 / 2018 / 2021 Stress Periods)", fontsize=12)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower left")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    axes[1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / "comparison_csi_vs_hs300.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.show()


def print_summary_table(
    total_return: pd.Series, annualized_return: pd.Series, max_drawdown: pd.Series
) -> None:
    """Print a compact business-formal metrics table."""
    summary = pd.DataFrame(
        {
            "Total Return (%)": (total_return * 100).round(2),
            "Annualized Return (%)": (annualized_return * 100).round(2),
            "Max Drawdown (%)": (max_drawdown * 100).round(2),
        }
    )
    print(f"Date range: {START_DATE} to {date.today().isoformat()}")
    print("\nPerformance Summary")
    print(summary.to_string())


def main() -> None:
    prices = build_aligned_prices(START_DATE)
    normalized, drawdown, total_return, annualized_return, max_drawdown = calculate_metrics(prices)
    plot_comparison(normalized, drawdown)
    print_summary_table(total_return, annualized_return, max_drawdown)


if __name__ == "__main__":
    main()
