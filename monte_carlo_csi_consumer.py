#!/usr/bin/env python3
"""Monte Carlo simulation for CSI Consumer Index (sz399932) using AkShare."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import akshare as ak
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDEX_SYMBOL = "sz399932"
HISTORY_START = "2005-01-01"

SIM_PATHS = 10_000
SIM_START = "2025-09-01"
SIM_END = "2030-12-31"

MONTHLY_INVESTMENT = 10_000.0
INVESTMENT_MONTHS = 10
TOTAL_INVESTMENT = MONTHLY_INVESTMENT * INVESTMENT_MONTHS

RNG_SEED = 42


@dataclass
class SimulationStats:
    annual_mean_return: float
    annual_volatility: float
    winning_probability: float
    median_profit_loss: float
    ci_low_value: float
    ci_high_value: float
    ci_low_pl: float
    ci_high_pl: float


def fetch_history(symbol: str, start_date: str) -> pd.Series:
    """Fetch daily close data from AkShare and return a clean price series."""
    df = ak.stock_zh_index_daily(symbol=symbol)
    if df.empty:
        raise RuntimeError(f"No data returned by AkShare for {symbol}.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).set_index("date").sort_index()
    df = df[df.index >= pd.Timestamp(start_date)]

    if df.empty:
        raise RuntimeError(f"{symbol} has no valid data after {start_date}.")
    return df["close"].rename(symbol)


def estimate_annual_params(prices: pd.Series) -> tuple[float, float]:
    """Estimate annual arithmetic mean return and annual volatility from daily returns."""
    daily_returns = prices.pct_change().dropna()
    if daily_returns.empty:
        raise RuntimeError("Insufficient history to estimate return/volatility.")

    annual_mean_return = daily_returns.mean() * 252
    annual_volatility = daily_returns.std(ddof=1) * np.sqrt(252)
    return annual_mean_return, annual_volatility


def run_monte_carlo(
    annual_mean_return: float,
    annual_volatility: float,
    n_paths: int,
    start_date: str,
    end_date: str,
    monthly_investment: float,
    investment_months: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Simulate portfolio value paths with monthly contributions and monthly GBM returns."""
    months = pd.date_range(start=start_date, end=end_date, freq="ME")
    n_steps = len(months)
    if n_steps <= 0:
        raise RuntimeError("Simulation horizon has zero monthly steps.")

    # Convert annual arithmetic moments to monthly lognormal step moments.
    monthly_mu_log = (annual_mean_return - 0.5 * annual_volatility**2) / 12.0
    monthly_sigma_log = annual_volatility / np.sqrt(12.0)

    rng = np.random.default_rng(RNG_SEED)
    shocks = rng.standard_normal((n_steps, n_paths))
    growth_factors = np.exp(monthly_mu_log + monthly_sigma_log * shocks)

    portfolio_paths = np.zeros((n_steps, n_paths), dtype=float)
    portfolio = np.zeros(n_paths, dtype=float)

    for t in range(n_steps):
        if t < investment_months:
            portfolio += monthly_investment
        portfolio *= growth_factors[t]
        portfolio_paths[t] = portfolio

    return months, portfolio_paths


def compute_stats(final_values: np.ndarray, total_investment: float) -> SimulationStats:
    """Compute required summary statistics from final portfolio values."""
    final_profit_loss = final_values - total_investment
    winning_probability = float(np.mean(final_values > total_investment))
    median_profit_loss = float(np.median(final_profit_loss))

    ci_low_value, ci_high_value = np.percentile(final_values, [2.5, 97.5])
    ci_low_pl, ci_high_pl = np.percentile(final_profit_loss, [2.5, 97.5])

    return SimulationStats(
        annual_mean_return=np.nan,  # filled by caller
        annual_volatility=np.nan,  # filled by caller
        winning_probability=winning_probability,
        median_profit_loss=median_profit_loss,
        ci_low_value=float(ci_low_value),
        ci_high_value=float(ci_high_value),
        ci_low_pl=float(ci_low_pl),
        ci_high_pl=float(ci_high_pl),
    )


def plot_results(months: pd.DatetimeIndex, portfolio_paths: np.ndarray, total_investment: float) -> Path:
    """Plot fan chart and final P/L histogram, then save to local file."""
    fan_pct = np.percentile(portfolio_paths, [5, 25, 50, 75, 95], axis=1)
    final_pl = portfolio_paths[-1] - total_investment

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    ax0 = axes[0]
    ax0.fill_between(months, fan_pct[0], fan_pct[4], alpha=0.20, label="5%-95%")
    ax0.fill_between(months, fan_pct[1], fan_pct[3], alpha=0.30, label="25%-75%")
    ax0.plot(months, fan_pct[2], linewidth=2.0, label="Median Path")
    ax0.axhline(total_investment, linestyle="--", linewidth=1.2, label="Total Invested (100,000 CNY)")
    ax0.set_title("Monte Carlo Fan Chart: Portfolio Value (CSI Consumer, sz399932)")
    ax0.set_ylabel("Portfolio Value (CNY)")
    ax0.legend(loc="upper left")
    ax0.xaxis.set_major_locator(mdates.YearLocator(1))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax1 = axes[1]
    ax1.hist(final_pl, bins=60, alpha=0.8, edgecolor="black")
    ax1.axvline(0.0, linestyle="--", linewidth=1.5, label="Break-even")
    ax1.set_title("Distribution of Final Profit/Loss in Dec 2030")
    ax1.set_xlabel("Final Profit/Loss (CNY)")
    ax1.set_ylabel("Frequency")
    ax1.legend(loc="upper right")

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "monte_carlo_csi_consumer.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.show()
    return output_path


def print_summary(stats: SimulationStats) -> None:
    """Print requested simulation statistics."""
    print("\nMonte Carlo Summary (Dec 2030)")
    print("-" * 70)
    print(f"Historical Annual Mean Return: {stats.annual_mean_return:.2%}")
    print(f"Historical Annual Volatility : {stats.annual_volatility:.2%}")
    print(f"Winning Probability          : {stats.winning_probability:.2%}")
    print(f"Median Profit/Loss (CNY)    : {stats.median_profit_loss:,.2f}")
    print(
        "95% Confidence Interval"
        f" (Final Value, CNY)        : [{stats.ci_low_value:,.2f}, {stats.ci_high_value:,.2f}]"
    )
    print(
        "95% Confidence Interval"
        f" (Profit/Loss, CNY)        : [{stats.ci_low_pl:,.2f}, {stats.ci_high_pl:,.2f}]"
    )
    print("-" * 70)


def main() -> None:
    prices = fetch_history(INDEX_SYMBOL, HISTORY_START)
    annual_mean_return, annual_volatility = estimate_annual_params(prices)

    months, portfolio_paths = run_monte_carlo(
        annual_mean_return=annual_mean_return,
        annual_volatility=annual_volatility,
        n_paths=SIM_PATHS,
        start_date=SIM_START,
        end_date=SIM_END,
        monthly_investment=MONTHLY_INVESTMENT,
        investment_months=INVESTMENT_MONTHS,
    )

    final_values = portfolio_paths[-1]
    stats = compute_stats(final_values=final_values, total_investment=TOTAL_INVESTMENT)
    stats.annual_mean_return = annual_mean_return
    stats.annual_volatility = annual_volatility

    plot_results(months, portfolio_paths, TOTAL_INVESTMENT)
    print_summary(stats)


if __name__ == "__main__":
    main()
