#!/usr/bin/env python3
"""Monte Carlo simulation for CSI Consumer Index (sz399932) with GARCH vol linkage,
regime-switching GBM, DCA scenario comparison, strategic allocation, VaR/CVaR backtesting,
and historical simulation benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from modules.core import (
    CSI_CONSUMER_SYMBOL,
    TRADING_DAYS_PER_YEAR,
    daily_vol_to_monthly,
    estimate_annual_params,
    fetch_index_daily,
    garch_fit_and_forecast,
    garch_regime_labels,
)

HISTORY_START = "2005-01-01"

SIM_PATHS = 10_000
SIM_START = "2025-09-01"
SIM_END = "2030-12-31"

MONTHLY_INVESTMENT = 10_000.0
INVESTMENT_MONTHS = 10
TOTAL_INVESTMENT = MONTHLY_INVESTMENT * INVESTMENT_MONTHS

RNG_SEED = 42
VAR_LEVELS = [0.95, 0.99]
GARCH_CALIBRATION_WINDOW = 1008  # latest ~4 trading years, matching rolling analysis


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
    var_95: float = np.nan
    cvar_95: float = np.nan
    var_99: float = np.nan
    cvar_99: float = np.nan


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
    months = pd.date_range(start=start_date, end=end_date, freq="M")
    n_steps = len(months)
    if n_steps <= 0:
        raise RuntimeError("Simulation horizon has zero monthly steps.")

    if annual_vol_array is not None and len(annual_vol_array) < n_steps:
        annual_vol_array = np.pad(annual_vol_array, (0, n_steps - len(annual_vol_array)),
                                  mode="edge")

    rng = np.random.default_rng(RNG_SEED)
    shocks = t_dist.rvs(df=5, size=(n_steps, n_paths), random_state=RNG_SEED)

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


def run_regime_switching_mc(
    annual_mean_return: float,
    annual_volatility: float,
    n_paths: int,
    start_date: str,
    end_date: str,
    monthly_investment: float,
    investment_months: int,
    regime_vols: dict[int, float],
    transition_matrix: np.ndarray,
    initial_regime: int = 0,
    regime_means: dict[int, float] | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Simulate portfolio paths with two-regime Markov-switching GBM.

    Args:
        regime_vols: {0: low_vol, 1: high_vol} — annualized vol for each regime
        transition_matrix: 2×2 Markov transition matrix P[i,j] = P(state j | state i)
        initial_regime: starting regime (0 or 1)
        regime_means: optional annualized mean return for each regime
    Returns:
        months, portfolio_paths, regime_paths (n_steps × n_paths of regime labels)
    """
    months = pd.date_range(start=start_date, end=end_date, freq="M")
    n_steps = len(months)
    if n_steps <= 0:
        raise RuntimeError("Simulation horizon has zero monthly steps.")

    rng = np.random.default_rng(RNG_SEED)
    shocks = t_dist.rvs(df=5, size=(n_steps, n_paths), random_state=RNG_SEED)
    regime_draws = rng.uniform(0, 1, (n_steps, n_paths))

    portfolio_paths = np.zeros((n_steps, n_paths), dtype=float)
    regime_paths = np.zeros((n_steps, n_paths), dtype=int)
    portfolio = np.zeros(n_paths, dtype=float)
    current_regime = np.full(n_paths, initial_regime, dtype=int)

    for t in range(n_steps):
        if t < investment_months:
            portfolio += monthly_investment

        vol_path = np.array([regime_vols[r] for r in current_regime], dtype=float)
        if regime_means is not None:
            mu_path = np.array([regime_means[r] for r in current_regime], dtype=float)
        else:
            mu_path = np.full(n_paths, annual_mean_return, dtype=float)

        monthly_mu_log = (mu_path - 0.5 * vol_path**2) / 12.0
        monthly_sigma_log = vol_path / np.sqrt(12.0)
        growth_factors = np.exp(monthly_mu_log + monthly_sigma_log * shocks[t])
        portfolio *= growth_factors
        portfolio_paths[t] = portfolio
        regime_paths[t] = current_regime

        # Markov transition: for each path, if uniform draw < P(stay), stay; else switch
        stay_probs = np.array([transition_matrix[r, r] for r in current_regime])
        switch = regime_draws[t] > stay_probs
        current_regime = np.where(switch, 1 - current_regime, current_regime)

    return months, portfolio_paths, regime_paths


def run_strategy_mc(
    annual_mean_return: float,
    annual_volatility: float,
    n_paths: int,
    start_date: str,
    end_date: str,
    monthly_contribution: float,
    investment_months: int,
    total_budget: float,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """MC with valuation-driven allocation and OU mean-reverting PE signal.

    Simulates a PE percentile path following an Ornstein-Uhlenbeck process
    (mean-reverts to 0.50 with half-life ~2 years). When PE > 80th percentile
    (expensive), defers 50% contribution to cash for later deployment. When PE
    < 20th percentile (cheap), deploys accumulated cash on top of base contribution.

    Budget-aware: total deployed never exceeds `total_budget`; final value
    includes remaining cash. This ensures fair comparison with passive DCA.

    Returns:
        months, portfolio_paths, alloc_paths, cash_paths
    """
    months = pd.date_range(start=start_date, end=end_date, freq="M")
    n_steps = len(months)
    if n_steps <= 0:
        raise RuntimeError("Simulation horizon has zero monthly steps.")

    rng = np.random.default_rng(RNG_SEED)
    shocks = t_dist.rvs(df=5, size=(n_steps, n_paths), random_state=RNG_SEED)

    # OU PE percentile path: mean-reverts to 0.50, half-life ~24 months
    theta = 0.50
    kappa = np.log(2) / 24.0  # half-life of 24 months
    sigma_ou = 0.08  # monthly noise
    ou_corr = -0.3  # PE-price shock correlation
    pe_noise_coef = np.sqrt(max(1 - ou_corr**2, 0.01))
    pe_paths = np.full(n_paths, theta, dtype=float)

    portfolio_paths = np.zeros((n_steps, n_paths), dtype=float)
    alloc_paths = np.zeros((n_steps, n_paths), dtype=float)
    cash_paths = np.zeros((n_steps, n_paths), dtype=float)
    portfolio = np.zeros(n_paths, dtype=float)
    cash = np.zeros(n_paths, dtype=float)
    invested_total = np.zeros(n_paths, dtype=float)

    for t in range(n_steps):
        if t < investment_months:
            cash += monthly_contribution

        # Link the OU signal to the same shock driving this month's price move.
        indep_noise = rng.standard_normal(n_paths)
        pe_noise = ou_corr * shocks[t] + pe_noise_coef * indep_noise
        pe_paths = pe_paths + kappa * (theta - pe_paths) + sigma_ou * pe_noise
        pe_paths = np.clip(pe_paths, 0.01, 0.99)

        alloc_mult = np.ones(n_paths, dtype=float)
        cheap_mask = pe_paths < 0.20
        expensive_mask = pe_paths > 0.80

        base_invest = monthly_contribution if t < investment_months else 0.0

        # Conservative (PE > 0.80): defer 50% — invest less, build cash
        alloc_mult[expensive_mask] = 0.5

        # Aggressive (PE < 0.20): deploy up to 2× base from accumulated cash
        alloc_mult[cheap_mask] = np.minimum(
            2.0,
            1.0 + cash[cheap_mask] / np.maximum(base_invest, 1.0)
        )

        invest_amount = base_invest * alloc_mult
        # Cannot invest more than available cash or remaining budget
        invest_amount = np.minimum(invest_amount, cash)
        remaining_budget = np.maximum(total_budget - invested_total, 0.0)
        invest_amount = np.minimum(invest_amount, remaining_budget)

        portfolio += invest_amount
        cash -= invest_amount
        invested_total += invest_amount
        alloc_paths[t] = invest_amount
        cash_paths[t] = cash

        monthly_mu_log = (annual_mean_return - 0.5 * annual_volatility**2) / 12.0
        monthly_sigma_log = annual_volatility / np.sqrt(12.0)
        growth_factors = np.exp(monthly_mu_log + monthly_sigma_log * shocks[t])
        portfolio *= growth_factors
        portfolio_paths[t] = portfolio

    return months, portfolio_paths, alloc_paths, cash_paths


def run_historical_simulation(
    daily_returns: pd.Series,
    n_paths: int,
    start_date: str,
    end_date: str,
    monthly_investment: float,
    investment_months: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Bootstrapped historical simulation: sample daily returns with replacement."""
    months = pd.date_range(start=start_date, end=end_date, freq="M")
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


def compute_var_cvar(final_values: np.ndarray, total_investment: float) -> dict[str, float]:
    """Compute VaR and CVaR (Expected Shortfall) from terminal values.

    VaR: dollar loss at given confidence level. CVaR: expected loss beyond VaR.
    Loss = total_investment - final_value (positive = loss).
    """
    losses = total_investment - final_values
    result = {}
    for level in VAR_LEVELS:
        alpha = 1 - level
        var = float(np.percentile(losses, level * 100))
        cvar = float(losses[losses >= var].mean()) if np.any(losses >= var) else var
        result[f"var_{int(level*100)}"] = var
        result[f"cvar_{int(level*100)}"] = cvar
    return result


def kupiec_test(
    losses: np.ndarray, var_level: float, confidence: float = 0.95
) -> dict[str, float]:
    """Run the Kupiec proportion-of-failures test for a VaR model."""
    from scipy.stats import chi2

    del confidence  # The POF statistic itself is independent of the decision cutoff.
    n_total = len(losses)
    if n_total == 0:
        raise ValueError("Kupiec test requires at least one loss observation.")

    var_value = float(np.percentile(losses, var_level * 100))
    n_exceed = int((losses > var_value).sum())

    expected_rate = 1 - var_level
    actual_rate = n_exceed / n_total

    if n_exceed == 0:
        lr_stat = -2 * n_total * np.log(1 - expected_rate)
    elif n_exceed == n_total:
        lr_stat = -2 * n_total * np.log(expected_rate)
    else:
        lr_stat = -2 * (
            (n_total - n_exceed)
            * np.log((1 - expected_rate) / (1 - actual_rate))
            + n_exceed * np.log(expected_rate / actual_rate)
        )

    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "n_total": n_total,
        "n_exceed": n_exceed,
        "expected_rate": expected_rate,
        "actual_rate": actual_rate,
        "p_value": float(p_value),
    }


def compute_stats(
    final_values: np.ndarray, total_investment: float, label: str,
    annual_mean_return: float, annual_volatility: float,
) -> SimulationStats:
    """Compute summary statistics including VaR/CVaR from final portfolio values."""
    final_profit_loss = final_values - total_investment
    winning_probability = float(np.mean(final_values > total_investment))
    median_profit_loss = float(np.median(final_profit_loss))

    ci_low_value, ci_high_value = np.percentile(final_values, [2.5, 97.5])
    ci_low_pl, ci_high_pl = np.percentile(final_profit_loss, [2.5, 97.5])

    var_cvar = compute_var_cvar(final_values, total_investment)

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
        var_95=var_cvar["var_95"],
        cvar_95=var_cvar["cvar_95"],
        var_99=var_cvar["var_99"],
        cvar_99=var_cvar["cvar_99"],
    )


def plot_results(
    months: pd.DatetimeIndex,
    portfolio_paths: np.ndarray,
    total_investment: float,
    title_suffix: str = "",
    output_name: str = "monte_carlo_csi_consumer.png",
    label: str = "",
    var_stats: dict | None = None,
) -> Path:
    """Plot fan chart and final P/L histogram with VaR overlay."""
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

    if var_stats:
        var_95 = var_stats.get("var_95", np.nan)
        var_99 = var_stats.get("var_99", np.nan)
        if np.isfinite(var_95):
            ax1.axvline(-var_95, linestyle="-.", linewidth=1.2, color="orange",
                        label=f"VaR 95% ({var_95:,.0f} loss)")
        if np.isfinite(var_99):
            ax1.axvline(-var_99, linestyle=":", linewidth=1.2, color="red",
                        label=f"VaR 99% ({var_99:,.0f} loss)")

    ax1.set_title(f"Distribution of Final Profit/Loss {title_suffix}")
    ax1.set_xlabel("Final Profit/Loss (CNY)")
    ax1.set_ylabel("Frequency")
    ax1.legend(loc="upper right", fontsize=8)

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
    """Plot bar chart comparing winning probability and median P/L across scenarios."""
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


def plot_var_backtest(
    all_stats: list[SimulationStats],
    historical_worst_drawdown_pct: float,
    output_name: str = "monte_carlo_var_backtest.png",
) -> Path:
    """Plot VaR/CVaR bar chart across scenarios, benchmarked against historical worst drawdown."""
    labels = [s.label for s in all_stats]
    var_95s = np.array([s.var_95 for s in all_stats], dtype=float)
    cvar_95s = np.array([s.cvar_95 for s in all_stats], dtype=float)
    var_99s = np.array([s.var_99 for s in all_stats], dtype=float)
    cvar_99s = np.array([s.cvar_99 for s in all_stats], dtype=float)

    mask = np.isfinite(var_95s)
    labels = [l for l, m in zip(labels, mask) if m]
    var_95s = var_95s[mask]
    cvar_95s = cvar_95s[mask]
    var_99s = var_99s[mask]
    cvar_99s = cvar_99s[mask]

    x = np.arange(len(labels))
    width = 0.2

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - 1.5 * width, var_95s, width, label="VaR 95%", alpha=0.85, color="#ff7f0e")
    ax.bar(x - 0.5 * width, cvar_95s, width, label="CVaR 95%", alpha=0.85, color="#1f77b4")
    ax.bar(x + 0.5 * width, var_99s, width, label="VaR 99%", alpha=0.85, color="#d62728")
    ax.bar(x + 1.5 * width, cvar_99s, width, label="CVaR 99%", alpha=0.85, color="#9467bd")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Loss at Risk (CNY)")
    ax.set_title(f"VaR / CVaR Backtest — Historical Worst Drawdown: {historical_worst_drawdown_pct:.1%}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / output_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"VaR backtest chart saved to: {output_path}")
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
    print(f"VaR 95% (CNY)               : {stats.var_95:,.2f}")
    print(f"CVaR 95% (CNY)              : {stats.cvar_95:,.2f}")
    print(f"VaR 99% (CNY)               : {stats.var_99:,.2f}")
    print(f"CVaR 99% (CNY)              : {stats.cvar_99:,.2f}")
    print("-" * 70)


def estimate_regime_transition_matrix(
    cond_vol: pd.Series, high_vol_pct: float = 70.0
) -> np.ndarray:
    """Estimate a two-state Markov transition matrix from conditional volatility."""
    threshold = np.percentile(cond_vol.dropna(), high_vol_pct)
    regimes = (cond_vol >= threshold).astype(int).dropna()

    if len(regimes) < 100:
        return np.array([[0.85, 0.15], [0.15, 0.85]])

    trans_counts = np.zeros((2, 2), dtype=int)
    regime_prev = regimes.iloc[:-1].values
    regime_curr = regimes.iloc[1:].values

    for i in range(len(regime_prev)):
        trans_counts[regime_prev[i], regime_curr[i]] += 1

    trans_matrix = np.zeros((2, 2), dtype=float)
    for i in range(2):
        row_sum = trans_counts[i].sum()
        if row_sum > 0:
            trans_matrix[i] = trans_counts[i] / row_sum
        else:
            trans_matrix[i] = (
                np.array([0.85, 0.15]) if i == 0 else np.array([0.15, 0.85])
            )

    return trans_matrix


def estimate_regime_returns(
    daily_returns: pd.Series, cond_vol: pd.Series, high_vol_pct: float = 70.0
) -> dict[int, float]:
    """Estimate annualized mean return for low- and high-volatility regimes."""
    threshold = np.percentile(cond_vol.dropna(), high_vol_pct)
    regimes = (cond_vol >= threshold).astype(int)

    common_index = regimes.index.intersection(daily_returns.index)
    aligned_ret = daily_returns.loc[common_index]
    aligned_reg = regimes.loc[common_index]

    regime_means = {}
    for state in [0, 1]:
        mask = aligned_reg == state
        if mask.sum() > 1:
            regime_means[state] = float(
                aligned_ret[mask].mean() * TRADING_DAYS_PER_YEAR
            )
        else:
            regime_means[state] = float(
                daily_returns.mean() * TRADING_DAYS_PER_YEAR
            )

    return regime_means


def main() -> None:
    raw = fetch_index_daily(CSI_CONSUMER_SYMBOL, HISTORY_START)
    prices = raw.set_index("date")["close"].rename(CSI_CONSUMER_SYMBOL)
    annual_mean_return, annual_volatility = estimate_annual_params(prices)

    # Historical worst drawdown for VaR backtest benchmark
    cummax = prices.cummax()
    drawdowns = prices / cummax - 1.0
    historical_worst_dd = float(drawdowns.min())

    # ── 1) Constant-vol GBM (baseline) ────────────────────────────────────
    dca_months_options = [10, 60, 120]
    all_stats: list[SimulationStats] = []

    for inv_months in dca_months_options:
        total_inv = MONTHLY_INVESTMENT * inv_months
        label = f"DCA {inv_months}mo (const vol)"
        months, paths = run_monte_carlo(
            annual_mean_return=annual_mean_return,
            annual_volatility=annual_volatility,
            n_paths=SIM_PATHS, start_date=SIM_START, end_date=SIM_END,
            monthly_investment=MONTHLY_INVESTMENT, investment_months=inv_months,
        )
        stats = compute_stats(paths[-1], total_inv, label,
                              annual_mean_return, annual_volatility)
        print_summary(stats)
        all_stats.append(stats)
        if inv_months == 10:
            plot_results(months, paths, total_inv,
                         title_suffix="(Constant Vol, 10mo DCA)",
                         output_name="monte_carlo_csi_consumer.png", label=label,
                         var_stats={"var_95": stats.var_95, "var_99": stats.var_99})

    # ── 2) Lump-sum comparison ───────────────────────────────────────────
    lump_months = 60
    total_lump = MONTHLY_INVESTMENT * lump_months
    label_lump = f"Lump-sum {lump_months}mo upfront (const vol)"
    months_lump, paths_lump = run_monte_carlo(
        annual_mean_return=annual_mean_return,
        annual_volatility=annual_volatility,
        n_paths=SIM_PATHS, start_date=SIM_START, end_date=SIM_END,
        monthly_investment=total_lump, investment_months=1,
    )
    stats_lump = compute_stats(paths_lump[-1], total_lump, label_lump,
                               annual_mean_return, annual_volatility)
    print_summary(stats_lump)
    all_stats.append(stats_lump)

    # ── 3) GARCH time-varying vol GBM ────────────────────────────────────
    monthly_vol_array_garch = None
    try:
        cond_vol, fc_vol_daily, forecast_start = garch_fit_and_forecast(
            prices,
            forecast_horizon=2520,
            calibration_window=GARCH_CALIBRATION_WINDOW,
        )
        print(
            f"\n[GARCH-MC] Calibrated on latest {len(cond_vol)} trading days "
            f"({cond_vol.index[0].date()} to {cond_vol.index[-1].date()})."
        )
        months_ref = pd.date_range(start=SIM_START, end=SIM_END, freq="M")
        monthly_vol_array_garch = daily_vol_to_monthly(fc_vol_daily, months_ref, forecast_start) / 100.0

        for inv_months in dca_months_options:
            total_inv = MONTHLY_INVESTMENT * inv_months
            label_garch = f"DCA {inv_months}mo (GARCH vol)"
            months_g, paths_g = run_monte_carlo(
                annual_mean_return=annual_mean_return,
                annual_volatility=annual_volatility,
                n_paths=SIM_PATHS, start_date=SIM_START, end_date=SIM_END,
                monthly_investment=MONTHLY_INVESTMENT, investment_months=inv_months,
                annual_vol_array=monthly_vol_array_garch,
            )
            stats_g = compute_stats(paths_g[-1], total_inv, label_garch,
                                    annual_mean_return, annual_volatility)
            print_summary(stats_g)
            all_stats.append(stats_g)

        # Lump-sum with GARCH vol
        label_gl = f"Lump-sum {lump_months}mo upfront (GARCH vol)"
        months_gl, paths_gl = run_monte_carlo(
            annual_mean_return=annual_mean_return,
            annual_volatility=annual_volatility,
            n_paths=SIM_PATHS, start_date=SIM_START, end_date=SIM_END,
            monthly_investment=total_lump, investment_months=1,
            annual_vol_array=monthly_vol_array_garch,
        )
        stats_gl = compute_stats(paths_gl[-1], total_lump, label_gl,
                                 annual_mean_return, annual_volatility)
        print_summary(stats_gl)
        all_stats.append(stats_gl)

        # ── 3b) Regime-switching MC (two-state, data-driven) ─────────────
        # Regime parameters need a complete market cycle, while forward GARCH
        # volatility remains calibrated on the latest four-year window above.
        seq_cond_vol_full, _, _ = garch_fit_and_forecast(
            prices,
            forecast_horizon=1,
            calibration_window=None,
        )
        cond_vol_hist = seq_cond_vol_full / 100.0
        trans = estimate_regime_transition_matrix(cond_vol_hist)
        daily_ret = prices.pct_change().dropna()
        regime_return_estimates = estimate_regime_returns(daily_ret, cond_vol_hist)

        vol_threshold = np.percentile(cond_vol_hist.dropna(), 70)
        low_vol_mask = cond_vol_hist < vol_threshold
        high_vol_mask = cond_vol_hist >= vol_threshold
        low_vol = (
            float(cond_vol_hist[low_vol_mask].mean())
            if low_vol_mask.any()
            else float(cond_vol_hist.mean()) * 0.7
        )
        high_vol = (
            float(cond_vol_hist[high_vol_mask].mean())
            if high_vol_mask.any()
            else float(cond_vol_hist.mean()) * 1.3
        )
        regime_vols = {0: low_vol, 1: high_vol}

        print(f"\n[Regime-Switching] Estimated transition matrix:\n{trans}")
        print(
            f"  Low-vol ({low_vol:.1%}): mean ret = "
            f"{regime_return_estimates.get(0, annual_mean_return):.2%}"
        )
        print(
            f"  High-vol ({high_vol:.1%}): mean ret = "
            f"{regime_return_estimates.get(1, annual_mean_return):.2%}"
        )

        label_rs = "DCA 60mo (Regime-Switching, data-driven)"
        total_inv = MONTHLY_INVESTMENT * 60
        months_rs, paths_rs, regimes_rs = run_regime_switching_mc(
            annual_mean_return=annual_mean_return,
            annual_volatility=annual_volatility,
            n_paths=SIM_PATHS, start_date=SIM_START, end_date=SIM_END,
            monthly_investment=MONTHLY_INVESTMENT, investment_months=60,
            regime_vols=regime_vols, transition_matrix=trans, initial_regime=0,
            regime_means=regime_return_estimates,
        )
        stats_rs = compute_stats(paths_rs[-1], total_inv, label_rs,
                                 annual_mean_return, annual_volatility)
        print_summary(stats_rs)
        all_stats.append(stats_rs)
    except Exception as e:
        print(f"[GARCH-MC linkage skipped: {e}]")

    # ── 4) Historical simulation benchmark ───────────────────────────────
    daily_returns = prices.pct_change().dropna()
    daily_returns = daily_returns[daily_returns.index < pd.Timestamp(SIM_START)]
    for inv_months in dca_months_options:
        total_inv = MONTHLY_INVESTMENT * inv_months
        label_hs = f"DCA {inv_months}mo (Hist Sim)"
        try:
            months_hs, paths_hs = run_historical_simulation(
                daily_returns=daily_returns, n_paths=SIM_PATHS,
                start_date=SIM_START, end_date=SIM_END,
                monthly_investment=MONTHLY_INVESTMENT, investment_months=inv_months,
            )
            stats_hs = compute_stats(paths_hs[-1], total_inv, label_hs,
                                     annual_mean_return, annual_volatility)
            print_summary(stats_hs)
            all_stats.append(stats_hs)
        except Exception as e:
            print(f"[Historical simulation skipped for {inv_months}mo: {e}]")

    # ── 5) Strategy MC (valuation-driven allocation, OU PE signal) ──────────
    label_strat = "DCA 60mo (OU PE Active)"
    total_inv = MONTHLY_INVESTMENT * 60
    try:
        months_st, paths_st, alloc_st, cash_st = run_strategy_mc(
            annual_mean_return=annual_mean_return,
            annual_volatility=annual_volatility,
            n_paths=SIM_PATHS, start_date=SIM_START, end_date=SIM_END,
            monthly_contribution=MONTHLY_INVESTMENT, investment_months=60,
            total_budget=total_inv,
        )
        final_with_cash = paths_st[-1] + cash_st[-1]
        stats_st = compute_stats(final_with_cash, total_inv, label_strat,
                                 annual_mean_return, annual_volatility)
        print_summary(stats_st)
        all_stats.append(stats_st)
    except Exception as e:
        print(f"[Strategy MC skipped: {e}]")

    # ── 6) Comparison & VaR backtest charts ───────────────────────────────
    if len(all_stats) > 1:
        plot_dca_comparison(all_stats)
        plot_var_backtest(all_stats, historical_worst_dd)

    # ── 7) Kupiec VaR backtest ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Kupiec POF VaR Backtest (Constant Vol, 10mo DCA)")
    print("=" * 70)
    months_ref, paths_ref = run_monte_carlo(
        annual_mean_return=annual_mean_return,
        annual_volatility=annual_volatility,
        n_paths=SIM_PATHS, start_date=SIM_START, end_date=SIM_END,
        monthly_investment=MONTHLY_INVESTMENT, investment_months=10,
    )
    total_ref = MONTHLY_INVESTMENT * 10
    losses_ref = total_ref - paths_ref[-1]
    for level in [0.95, 0.99]:
        kp = kupiec_test(losses_ref, level)
        verdict = "(PASS)" if kp["p_value"] > 0.05 else "(FAIL)"
        print(
            f"  VaR {level*100:.0f}%: expected exceed={kp['expected_rate']:.1%}, "
            f"actual={kp['actual_rate']:.1%} ({kp['n_exceed']}/{kp['n_total']}), "
            f"p-value={kp['p_value']:.4f} {verdict}"
        )


if __name__ == "__main__":
    main()
