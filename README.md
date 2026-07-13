# CSI Consumer Index — Quantitative Analysis Toolkit

A research-grade quantitative finance project analyzing the CSI Consumer Index (sz399932) with volatility modeling, Monte Carlo simulation, and machine learning directional prediction.

## Motivation

This project demonstrates end-to-end quantitative research capability: from raw data ingestion to model selection, risk quantification, and strategy evaluation. It targets the FinTech quant interview narrative: "I can build, validate, and productionize quant models."

## Quick Start

```bash
pip install -r requirements.txt
```

### Reproduce Main Results

| Script | Output | What It Shows |
|--------|--------|--------------|
| `comparison_csi_vs_hs300.py` | `comparison_csi_vs_hs300.png` | CSI Consumer outperformance vs CSI 300 since 2005 |
| `garch_sz399932.py` | `sz399932_volatility.png` | GARCH(1,1) conditional vol + 30-day forecast |
| `garch_gridsearch_sz399932.py` | `garch_asymmetric_comparison.png` | Grid search (p,q∈{1,2,3}) + GARCH vs EGARCH vs GJR-GARCH |
| `modules/rolling_garch.py` | `garch_rolling_params.png` | 4-year rolling GARCH parameter stability |
| `monte_carlo_csi_consumer.py` | `monte_carlo_csi_consumer.png`, `monte_carlo_dca_comparison.png`, `monte_carlo_var_backtest.png` | MC simulation: GBM, GARCH vol, regime-switching, Hist Sim, strategic DCA, VaR/CVaR |
| `ml_direction_csi_consumer.py` | `ml_direction_csi_consumer.png`, `ml_feature_importance_csi_consumer.png` | XGBoost/RandomForest 5-day direction prediction |

## Project Architecture

```
modules/core.py          # Shared library: data fetch, GARCH, vol models, regime labels
modules/rolling_garch.py # Rolling-window GARCH parameter analysis
```

All analysis scripts import from `modules/core.py` — no duplicated fetch or model code.

## Data

- **Source**: AkShare (`stock_zh_index_daily`, `index_zh_a_hist`)
- **Coverage**: CSI Consumer Index (sz399932) from 2005-01-01 to present
- **Caching**: CSV cache at `sz399932_akshare_cache.csv` with automatic retry (5 attempts, exponential backoff)
- **Benchmark**: CSI 300 Index (sh000300)

### Data Quality Notes

- **PE/Valuation**: Uses a quality-tagged fallback chain. The currently available tier 2 source is a Moutai price/120-day-MA trend proxy, not a true PE series. Absolute levels are not meaningful and should only be used as a directional valuation signal. Production deployment would use Wind or the official CSIndex PE(TTM) series.
- **GARCH calibration**: Monte Carlo volatility is calibrated on the latest 1,008 trading days (about four years), matching the terminal rolling window in `modules/rolling_garch.py`. The full rolling analysis remains available for parameter stability monitoring.
- **Distribution**: GARCH fitting and parametric Monte Carlo shocks use Student's t distribution (`df=5` for MC) to capture fat tails. Historical bootstrap simulation is retained as a non-parametric benchmark.

## Methodology Choices

### Why GARCH(1,1) and not EWMA?

EWMA imposes a fixed decay factor λ with no statistical foundation. GARCH(1,1) is data-driven — α[1] (shock sensitivity) and β[1] (persistence) are estimated by maximum likelihood. Grid search over p,q ∈ {1,2,3} confirms (1,1) minimizes BIC.

### Why BIC and not AIC for model selection?

AIC asymptotically selects the true model only if it's in the candidate set. BIC is consistent for model selection with large samples (4,000+ trading days) and penalizes over-parameterization more heavily, which aligns with the production concern: simpler models generalize better.

### Why EGARCH and GJR-GARCH?

Standard GARCH assumes symmetric volatility response to positive and negative shocks. A-share markets exhibit a pronounced leverage effect — volatility rises more after declines than after equivalent gains. EGARCH (Nelson 1991) models log-variance directly (no positivity constraints) and captures the sign of innovations via the gamma[1] parameter. GJR-GARCH (Glosten-Jagannathan-Runkle 1993) adds an asymmetric term that only activates for negative returns. The grid search script compares all three by BIC.

### Why Regime-Switching MC?

Naive constant-vol GBM produces symmetric, thin-tailed terminal distributions. GARCH time-varying vol is an improvement, but the path-dependence of vol clustering implies that consecutive high-vol months cluster together — a Markov-switching framework explicitly models this persistence, producing fatter tails and more realistic VaR estimates.

### Why Historical Simulation?

Bootstrapped historical simulation makes zero parametric assumptions — it draws directly from the empirical return distribution. The divergence between GBM-based and HistSim-based risk metrics is itself a model risk diagnostic. Large discrepancies signal that parametric assumptions may be misspecified.

### Why Strategic (OU PE Active) MC?

Passive DCA ignores valuation. The strategic MC simulates a PE percentile signal following an Ornstein-Uhlenbeck (mean-reverting) process — the signal drifts stochastically around 0.50 with a ~2-year half-life. When PE > 80th percentile (expensive), contributions are reduced to 50% and cash accumulates. When PE < 20th percentile (cheap), accumulated cash is deployed at up to 2× the base contribution rate. A fixed total budget cap ensures fair comparison against passive DCA: final portfolio value includes both invested assets and remaining cash.

## Key Outputs

_Values depend on current AkShare data and will vary by run date. Below are representative figures from a July 2026 run with 10,000-path MC, 60-month horizon._

| Metric | GBM const vol | GARCH vol GBM | Regime-Switching | Hist Sim | OU PE Active |
|--------|:---:|:---:|:---:|:---:|:---:|
| Win Probability | ~78% | ~75% | ~72% | ~80% | ~76% |
| Median P/L (CNY) | ~+85K | ~+72K | ~+60K | ~+92K | ~+78K |
| VaR 95% (CNY) | ~-45K | ~-58K | ~-72K | ~-40K | ~-48K |
| CVaR 95% (CNY) | ~-68K | ~-82K | ~-98K | ~-62K | ~-70K |

_Key pattern: GARCH time-varying vol produces wider tail risk (fatter left tail, more conservative VaR). Regime-switching further amplifies tail risk through vol-clustering persistence. Historical simulation tends to show lower VaR due to limited sample extreme events in the in-sample window._

## Limitations (Honest Assessment)

1. **No transaction costs**: Real DCA incurs ~0.03% commission per trade + bid-ask spread
2. **Valuation proxy quality**: Free PE endpoints can fall back to the tier 2 Moutai price/MA trend proxy. It is directional only; production requires Wind or official CSIndex PE(TTM)
3. **No regime persistence validation**: Markov transition probabilities are calibrated from GARCH conditional vol percentiles, not a proper EM-estimated HMM
4. **GBM drift assumption**: Long-run mean return μ is assumed constant; real markets exhibit structural breaks
5. **MC path count**: 10,000 paths balance runtime vs precision; 50,000+ would tighten VaR 99% estimates
6. **AkShare availability**: Data pipeline falls back to cached CSV when endpoints are unavailable; PE proxy may use trend-based fallback when Moutai PE endpoints return 404
7. **Stationarity**: GARCH assumes variance stationarity (α+β < 1); rolling analysis monitors persistence and should trigger recalibration if it approaches the unit-root boundary

## Environment

- Python 3.9+
- Core: `akshare`, `arch`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`

## License

This project is for educational and portfolio demonstration purposes.
