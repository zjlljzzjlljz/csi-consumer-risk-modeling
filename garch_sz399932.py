#!/usr/bin/env python3
"""GARCH(1,1) volatility modeling for CSI Consumer Index (sz399932)."""

from __future__ import annotations

import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.core import (
    CSI_CONSUMER_SYMBOL,
    TRADING_DAYS_PER_YEAR,
    fetch_index_daily_cached,
)


def main() -> int:
    symbol_sz = CSI_CONSUMER_SYMBOL
    start_date = "20100101"
    end_date = datetime.today().strftime("%Y%m%d")
    horizon = 30

    prices = fetch_index_daily_cached(symbol=symbol_sz, start_date=start_date, end_date=end_date)
    close = prices.set_index("date")["close"]

    log_returns = np.log(close).diff()
    returns_pct = (log_returns * 100.0).dropna()

    if len(returns_pct) < 100:
        raise ValueError(f"Not enough data points for GARCH fitting: {len(returns_pct)}")

    from arch import arch_model

    am = arch_model(
        returns_pct, vol="Garch", p=1, q=1, mean="Constant", dist="t", rescale=False,
    )
    res = am.fit(disp="off")

    cond_vol_annual = res.conditional_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)

    fc = res.forecast(horizon=horizon)
    var_next = fc.variance.iloc[-1].to_numpy()
    vol_next_annual = np.sqrt(var_next) * np.sqrt(TRADING_DAYS_PER_YEAR)

    last_date = returns_pct.index[-1]
    forecast_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="B")
    forecast_annual_vol = pd.Series(vol_next_annual, index=forecast_dates, name="forecast_annual_vol_pct")

    print(f"\nGARCH(1,1) on {symbol_sz} (Constant Mean; returns scaled by 100 for stability)\n")
    print(res.summary())
    alpha1 = res.params.get("alpha[1]", np.nan)
    beta1 = res.params.get("beta[1]", np.nan)
    print(f"\nalpha[1] (shock effect): {alpha1}")
    print(f"beta[1] (persistence effect): {beta1}")

    print("\n30-day forecast of annualized conditional volatility (%):")
    print(forecast_annual_vol.to_string())

    abs_daily_returns_pct = np.abs(returns_pct)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]},
    )

    ax_top.plot(returns_pct.index, abs_daily_returns_pct, linewidth=1.0)
    ax_top.set_ylabel("Abs. Log Return (%)")
    ax_top.set_title(f"Volatility Clustering - GARCH(1,1) on {symbol_sz}")
    ax_top.grid(True, alpha=0.2)

    x_cond = cond_vol_annual.index if hasattr(cond_vol_annual, "index") else returns_pct.index[-len(cond_vol_annual):]
    ax_bottom.plot(x_cond, cond_vol_annual, linewidth=1.3, label="Estimated Annualized Conditional Volatility (%)")

    for yr in [2015, 2021]:
        ax_bottom.axvspan(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31"), alpha=0.15)

    ax_bottom.plot(forecast_dates, forecast_annual_vol, linestyle="--", linewidth=1.2,
                   label="30-Day Forecast (Annualized, %)")

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
