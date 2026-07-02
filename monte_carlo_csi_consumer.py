#!/usr/bin/env python3
"""Monte Carlo simulation for CSI Consumer Index (sz399932) with GARCH vol linkage,
DCA scenario comparison, and historical simulation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.core import (
    CSI_CONSUMER_SYMBOL,
    TRADING_DAYS_PER_YEAR,
    daily_vol_to_monthly,
    estimate_annual_params,
    fetch_index_daily,
    garch_fit_and_forecast,
)

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
    label: str
    annual_mean_return: float
    annual_volatility: float
    winning_probability: float
    median_profit_loss: float
    ci_low_value: float
    ci_high_value: float
    ci_low_pl: float
    ci_high_pl: float
    total_investment: float


def run_monte_carlo(
    annual_mean_return: float,
    annual_volatility: float,
    n_paths: int,
    start_date: str,
    end_date: str,
    monthly_investment: float,
    investment_months: int,
    annual_vol_array: np.ndarray | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Simulate portfolio value paths with monthly GBM.

    If `annual_vol_array` is provided (shape: n_steps,), each month uses its own
    annualized volatility. Otherwise uses constant `annual_volatility`.
    """
    months = pd.date_range(start=start_date, end=end_date, freq="ME")
    n_steps = len(months)
    if n_steps <= 0:
        raise RuntimeError("Simulation horizon has zero monthly steps.")

    if annual_vol_array is not None and len(annual_vol_array) < n_steps:
        annual_vol_array = np.pad(annual_vol_array, (0, n_steps - len(annual_vol_array)),
                                  mode="edge")

    rng = np.random.default_rng(RNG_SEED)
    shocks = rng.standard_normal((n_steps, n_paths))

    portfolio_paths = np.zeros((n_steps, n_paths), dtype=float)
    portfolio = np.zeros(n_paths, dtype=float)

    for t in range(n_steps):
        if t < investment_months:
            portfolio += monthly_investment

        vol_t = annual_volatility if annual_vol_array is None else float(annual_vol_array[t])
        monthly_mu_log = (annual_mean_return - 0.5 * vol_t**2) / 12.0
        monthly_sigma_log = vol_t / np.sqrt(12.0)
        growth_factors = np.exp(monthly_mu_log + monthly_sigma_log * shocks[t])
        portfolio *= growth_factors
        portfolio_paths[t] = portfolio

    return months, portfolio_paths


def run_historical_simulation(
    daily_returns: pd.Series,
    n_paths: int,
    start_date: str,
    end_date: str,
    monthly_investment: float,
    investment_months: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Bootstrapped historical simulation: sample daily returns with replacement."""
    months = pd.date_range(start=start_date, end=end_date, freq="ME")
    n_steps = len(months)
    if n_steps <= 0:
        raise RuntimeError("Simulation horizon has zero monthly steps.")

    ret_array = daily_returns.dropna().to_numpy()
    if len(ret_array) == 0:
        raise RuntimeError("No daily returns available for historical simulation.")

    trading_days_per_month = TRADING_DAYS_PER_YEAR // 12

    rng = np.random.default_rng(RNG_SEED)
    portfolio_paths = np.zeros((n_steps, n_paths), dtype=float)
    portfolio = np.zeros(n_paths, dtype=float)

    for t in range(n_steps):
        if t < investment_months:
            portfolio += monthly_investment
        idx = rng.integers(0, len(ret_array), size=(trading_days_per_month, n_paths))
        monthly_return_contrib = np.prod(1 + ret_array[idx], axis=0) - 1
        portfolio *= (1 + monthly_return_contrib)
        portfolio_paths[t] = portfolio

    return months, portfolio_paths


def compute_stats(
    final_values: np.ndarray, total_investment: float, label: str,
    annual_mean_return: float, annual_volatility: float,
) -> SimulationStats:
    """Compute summary statistics from final portfolio values."""
    final_profit_loss = final_values - total_investment
    winning_probability = float(np.mean(final_values > total_investment))
    median_profit_loss = float(np.median(final_profit_loss))

    ci_low_value, ci_high_value = np.percentile(final_values, [2.5, 97.5])
    ci_low_pl, ci_high_pl = np.percentile(final_profit_loss, [2.5, 97.5])

    return SimulationStats(
        label=label,
        annual_mean_return=annual_mean_return,
        annual_volatility=annual_volatility,
        winning_probability=winning_probability,
        median_profit_loss=median_profit_loss,
        ci_low_value=float(ci_low_value),
        ci_high_value=float(ci_high_value),
        ci_low_pl=float(ci_low_pl),
        ci_high_pl=float(ci_high_pl),
        total_investment=total_investment,
    )


def plot_results(
    months: pd.DatetimeIndex,
    portfolio_paths: np.ndarray,
    total_investment: float,
    title_suffix: str = "",
    output_name: str = "monte_carlo_csi_consumer.png",
    label: str = "",
) -> Path:
    """Plot fan chart and final P/L histogram."""
    fan_pct = np.percentile(portfolio_paths, [5, 25, 50, 75, 95], axis=1)
    final_pl = portfolio_paths[-1] - total_investment

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    ax0 = axes[0]
    ax0.fill_between(months, fan_pct[0], fan_pct[4], alpha=0.20, label="5%-95%")
    ax0.fill_between(months, fan_pct[1], fan_pct[3], alpha=0.30, label="25%-75%")
    ax0.plot(months, fan_pct[2], linewidth=2.0, label="Median Path")
    ax0.axhline(total_investment, linestyle="--", linewidth=1.2,
                label=f"Total Invested ({total_investment:,.0f} CNY)")
    ax0.set_title(f"Monte Carlo Fan Chart: Portfolio Value {title_suffix}")
    ax0.set_ylabel("Portfolio Value (CNY)")
    ax0.legend(loc="upper left")
    ax0.xaxis.set_major_locator(mdates.YearLocator(1))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax1 = axes[1]
    ax1.hist(final_pl, bins=60, alpha=0.8, edgecolor="black")
    ax1.axvline(0.0, linestyle="--", linewidth=1.5, label="Break-even")
    ax1.set_title(f"Distribution of Final Profit/Loss {title_suffix}")
    ax1.set_xlabel("Final Profit/Loss (CNY)")
    ax1.set_ylabel("Frequency")
    ax1.legend(loc="upper right")

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / output_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.show()
    return output_path


def plot_dca_comparison(
    all_stats: list[SimulationStats],
    output_name: str = "monte_carlo_dca_comparison.png",
) -> Path:
    """Plot bar chart comparing winning probability and median P/L across DCA scenarios."""
    labels = [s.label for s in all_stats]
    win_probs = [s.winning_probability for s in all_stats]
    med_pls = [s.median_profit_loss for s in all_stats]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    ax1.barh(labels, win_probs, color=colors, alpha=0.85)
    ax1.set_title("Winning Probability by Scenario")
    ax1.set_xlabel("P(Profit > 0)")
    ax1.set_xlim(0, 1)
    for i, v in enumerate(win_probs):
        ax1.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)

    ax2.barh(labels, med_pls, color=colors, alpha=0.85)
    ax2.set_title("Median Profit/Loss by Scenario (CNY)")
    ax2.set_xlabel("Median P/L (CNY)")
    for i, v in enumerate(med_pls):
        sign = "+" if v >= 0 else ""
        ax2.text(v + max(abs(v) * 0.02, 500), i, f"{sign}{v:,.0f}", va="center", fontsize=9)

    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / output_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"DCA comparison chart saved to: {output_path}")
    plt.show()
    return output_path


def print_summary(stats: SimulationStats) -> None:
    """Print simulation statistics."""
    print(f"\n{stats.label}")
    print("-" * 70)
    print(f"Total Investment (CNY)      : {stats.total_investment:,.0f}")
    print(f"Historical Annual Mean Return: {stats.annual_mean_return:.2%}")
    print(f"Annual Volatility            : {stats.annual_volatility:.2%}")
    print(f"Winning Probability          : {stats.winning_probability:.2%}")
    print(f"Median Profit/Loss (CNY)    : {stats.median_profit_loss:,.2f}")
    print(f"95% CI Final Value (CNY)    : [{stats.ci_low_value:,.2f}, {stats.ci_high_value:,.2f}]")
    print(f"95% CI Profit/Loss (CNY)    : [{stats.ci_low_pl:,.2f}, {stats.ci_high_pl:,.2f}]")
    print("-" * 70)


def main() -> None:
    raw = fetch_index_daily(CSI_CONSUMER_SYMBOL, HISTORY_START)
    prices = raw.set_index("date")["close"].rename(CSI_CONSUMER_SYMBOL)
    annual_mean_return, annual_volatility = estimate_annual_params(prices)

    # ── 1) Constant-vol GBM (original baseline) ─────────────────────────────
    dca_months_options = [10, 60, 120]
    all_stats: list[SimulationStats] = []

    for inv_months in dca_months_options:
        total_inv = MONTHLY_INVESTMENT * inv_months
        label = f"DCA {inv_months}mo (const vol)"
        months, paths = run_monte_carlo(
            annual_mean_return=annual_mean_return,
            annual_volatility=annual_volatility,
            n_paths=SIM_PATHS,
            start_date=SIM_START,
            end_date=SIM_END,
            monthly_investment=MONTHLY_INVESTMENT,
            investment_months=inv_months,
        )
        stats = compute_stats(paths[-1], total_inv, label,
                              annual_mean_return, annual_volatility)
        print_summary(stats)
        all_stats.append(stats)
        if inv_months == 10:
            plot_results(months, paths, total_inv,
                         title_suffix="(Constant Vol, 10mo DCA)",
                         output_name="monte_carlo_csi_consumer.png",
                         label=label)

    # ── 2) Lump-sum comparison (60-month investment upfront) ───────────────
    lump_months = 60
    total_lump = MONTHLY_INVESTMENT * lump_months
    label_lump = f"Lump-sum {lump_months}mo upfront (const vol)"
    months_lump, paths_lump = run_monte_carlo(
        annual_mean_return=annual_mean_return,
        annual_volatility=annual_volatility,
        n_paths=SIM_PATHS,
        start_date=SIM_START,
        end_date=SIM_END,
        monthly_investment=total_lump,
        investment_months=1,
    )
    stats_lump = compute_stats(paths_lump[-1], total_lump, label_lump,
                               annual_mean_return, annual_volatility)
    print_summary(stats_lump)
    all_stats.append(stats_lump)

    # ── 3) GARCH time-varying vol GBM ──────────────────────────────────────
    try:
        cond_vol, fc_vol_daily, forecast_start = garch_fit_and_forecast(
            prices, forecast_horizon=2520
        )

        months_ref = pd.date_range(start=SIM_START, end=SIM_END, freq="ME")
        # GARCH returns vol in % (returns scaled ×100 for fitting); MC expects decimal
        monthly_vol_array = daily_vol_to_monthly(fc_vol_daily, months_ref, forecast_start) / 100.0

        for inv_months in dca_months_options:
            total_inv = MONTHLY_INVESTMENT * inv_months
            label_garch = f"DCA {inv_months}mo (GARCH vol)"
            months_g, paths_g = run_monte_carlo(
                annual_mean_return=annual_mean_return,
                annual_volatility=annual_volatility,
                n_paths=SIM_PATHS,
                start_date=SIM_START,
                end_date=SIM_END,
                monthly_investment=MONTHLY_INVESTMENT,
                investment_months=inv_months,
                annual_vol_array=monthly_vol_array,
            )
            stats_g = compute_stats(paths_g[-1], total_inv, label_garch,
                                    annual_mean_return, annual_volatility)
            print_summary(stats_g)
            all_stats.append(stats_g)
            if inv_months == 10:
                plot_results(months_g, paths_g, total_inv,
                             title_suffix="(GARCH Time-Varying Vol, 10mo DCA)",
                             output_name="monte_carlo_garch_vol.png",
                             label=label_garch)

        # Lump-sum with GARCH vol
        label_garch_lump = f"Lump-sum {lump_months}mo upfront (GARCH vol)"
        months_gl, paths_gl = run_monte_carlo(
            annual_mean_return=annual_mean_return,
            annual_volatility=annual_volatility,
            n_paths=SIM_PATHS,
            start_date=SIM_START,
            end_date=SIM_END,
            monthly_investment=total_lump,
            investment_months=1,
            annual_vol_array=monthly_vol_array,
        )
        stats_gl = compute_stats(paths_gl[-1], total_lump, label_garch_lump,
                                 annual_mean_return, annual_volatility)
        print_summary(stats_gl)
        all_stats.append(stats_gl)
    except Exception as e:
        print(f"[GARCH-MC linkage skipped: {e}]")

    # ── 4) Historical simulation benchmark ─────────────────────────────────
    daily_returns = prices.pct_change().dropna()
    daily_returns = daily_returns[daily_returns.index < pd.Timestamp(SIM_START)]
    for inv_months in dca_months_options:
        total_inv = MONTHLY_INVESTMENT * inv_months
        label_hs = f"DCA {inv_months}mo (Hist Sim)"
        try:
            months_hs, paths_hs = run_historical_simulation(
                daily_returns=daily_returns,
                n_paths=SIM_PATHS,
                start_date=SIM_START,
                end_date=SIM_END,
                monthly_investment=MONTHLY_INVESTMENT,
                investment_months=inv_months,
            )
            stats_hs = compute_stats(paths_hs[-1], total_inv, label_hs,
                                     annual_mean_return, annual_volatility)
            print_summary(stats_hs)
            all_stats.append(stats_hs)
            if inv_months == 10:
                plot_results(months_hs, paths_hs, total_inv,
                             title_suffix="(Historical Simulation, 10mo DCA)",
                             output_name="monte_carlo_hist_sim.png",
                             label=label_hs)
        except Exception as e:
            print(f"[Historical simulation skipped for {inv_months}mo: {e}]")

    # ── 5) DCA scenario comparison chart ───────────────────────────────────
    if len(all_stats) > 1:
        plot_dca_comparison(all_stats)


if __name__ == "__main__":
    main()
