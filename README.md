# Conformal Stochastic Model Predictive Control for Grid-Scale Energy Arbitrage

**Team `aniMLs` — Datathon FME 2026 (UPC Sustainable Campus Energy Optimization Challenge)**  
*2nd Place Overall (Tied Causal Frontier) in Standard Regime (`6.44 pts`) · Top Tier in Hard Stochastic Regime (`23.40 pts`) · **Zero Risk Violations***

---

## Executive Summary

Grid-scale battery storage systems face a classic **optimal inventory control problem under asymmetric transaction costs and non-stationary price/emission uncertainty**. Every charge–discharge cycle incurs an irreversible round-trip energy loss ($`1 - \eta_{\text{rt}} = 10\%`$, imposing a strict no-arbitrage spread threshold of $`1/\eta_{\text{rt}} - 1 \approx 11.11\%`$), while excess solar generation creates a piecewise-linear (*kinked*) payoff where grid exports yield zero marginal revenue.

This repository implements a two-tier **Receding-Horizon Optimal Execution Engine**:
1. **Standard Regime (`src/agent_standard.py`):** Exact 24-hour Deterministic Linear Program (HiGHS) with real-time exponential moving average (EMA) campus baseline calibration.
2. **Hard Regime (`src/agent_hard.py`):** Multi-scenario **Stochastic Model Predictive Control (SMPC)** formulated as a continuous newsvendor LP over heteroscedastic empirical error quantiles, coupled with **online conformal volatility scaling** and a **grey-box physics + residual GBDT forecasting pipeline**.

---

## Empirical Results (Out-of-Sample Evaluation)

Evaluated across a 6-week held-out horizon (1,008 hours) with zero lookahead bias:

| Evaluation Regime | Battery Specs | Forecast Visibility | Score | $`\Delta\text{CO}_2`$ | $`\Delta\text{Cost}`$ | $`\text{CO}_2`$ Abated | Cost Saved | Hard Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Mode** | 2 MWh / ±500 kW | 24h Market | **`6.44`** | `6.01%` | `7.43%` | `4,316.5 kg` | `€8,824.37` | **`0`** |
| **Hard Mode (×12 Solar)** | 6 MWh / ±1500 kW | 3h Market + Meteo | **`23.40`** | `22.17%` | `26.28%` | `13,357.8 kg` | `€24,961.44` | **`0`** |

### Causal Information Frontier Analysis
Using offline hindsight Linear Programming (168h full-week oracle), we established the theoretical upper bound on the validation set at **`6.83 pts`** (Standard) and **`25.61 pts`** (Hard). Our causal agents capture **94.3%** and **91.4%** of the theoretical clairvoyant optimum respectively, operating directly on the causal information frontier $`\mathcal{F}_t`$.

---

## Exploratory Market & Load Structure

![Campus Load, Solar Generation, and PVPC/CO2 Arbitrage Regimes](eda_resum.jpg)

* **Midday Solar Valley (`10:00–16:00`):** Simultaneous trough in grid carbon intensity ($\approx 0.052\text{ kg CO}_2/\text{kWh}$) and spot PVPC price, establishing the optimal charging window.
* **Evening Peak (`19:00–22:00`):** Coincident surge in marginal emissions ($\approx 0.110\text{ kg CO}_2/\text{kWh}$) and electricity price ($\approx €0.235/\text{kWh}$), exceeding the $11.11\%$ round-trip friction hurdle ($1/\eta_{\text{rt}} - 1$).
* **Weekday vs. Weekend Regime Shift:** Campus base load drops by $\sim 45\%$ during weekends ($\sim 750\text{ kW}$ vs. $\sim 1,550\text{ kW}$ weekday peak), motivating regime-aware baseline calibration.

---

## Mathematical Formulation

### 1. Piecewise-Linear Net Load & Asymmetric Payoffs
At each hourly step $`t`$, let $`d_t = L_t - P^{\text{PV}}_t`$ denote net campus load, let $`u^+_t \in [0, P_{\max}]`$ and $`u^-_t \in [0, P_{\max}]`$ denote charging and discharging power, and let $`g_t, x_t \ge 0`$ denote grid import and zero-remuneration export. Power balance enforces:

$$g_t - x_t - u^+_t + u^-_t = d_t$$

State-of-Charge $`\text{SoC}_t \in [0, C_{\max}]`$ evolves under symmetric one-way efficiencies $`\eta_c = \eta_d = \sqrt{0.90} \approx 0.9487`$:

$$\text{SoC}_{t+1} = \text{SoC}_t + \eta_c u^+_t - \frac{1}{\eta_d} u^-_t$$

The unified marginal cost $`c_t`$ aligns directly with the competition's normalized objective (70% carbon intensity, 30% PVPC spot price):

$$c_t = 0.70 \frac{\text{CO}_{2,t}}{\text{CO}_{2,\text{ref}}} + 0.30 \frac{P_t}{P_{\text{ref}}}$$

### 2. Conformal Stochastic MPC (Hard Regime)
In the Hard Regime, solar generation is scaled by $`12\times`$ and market forecasts beyond $`t+2`$ are unobserved. Discharging beyond actual net demand ($`u^-_t > d_t`$) spills energy to the grid at zero value—an asymmetric **newsvendor** payoff.

We model net load uncertainty across $`H = 24`$ steps using $`S`$ empirical error quantiles $`E_{h,s}`$ conditioned on hour-of-day ($`q_{\text{hod}}`$), dynamically rescaled online via an adaptive conformal volatility multiplier $`\nu_t \in [0.3, 3.0]`$:

$$\nu_{t+1} = \text{clip}\left((1 - \alpha)\nu_t + \alpha \frac{\lvert d^{\text{real}}_t - \hat{d}_t \rvert}{\bar{\epsilon}_{\text{typ}}(h_t)},\; 0.3,\; 3.0\right)$$

The Stochastic LP minimizes expected grid import cost plus terminal inventory shadow value $`V_{\text{term}} = \eta_d \, Q_{0.5}(c_{1:H})`$:

$$\min_{u^+, u^-, g \ge 0} \; \sum_{h=1}^{H} c_h \sum_{s=1}^{S} w_s \, g_{h,s} - V_{\text{term}} \, \text{SoC}_H \quad \text{s.t.} \quad g_{h,s} \ge \hat{d}_h + \nu_t E_{h,s} + u^+_h - u^-_h$$

### 3. Grey-Box Forecasting & Latency-Bounded Execution
- **Physics + Residual ML:** Solar generation is decomposed into clear-sky astronomical irradiance (`cos_zenith` at $`41.39^\circ\text{N}`$) multiplied by an exponentially decaying clearness index $`k_t`$, refined via residual XGBoost / Scikit-Learn regressors.
- **Graceful Latency Degradation:** Strict execution time budgets (`time_budget_s`, `stoch_budget_s`, `lp_budget_s`) progressively transition from Stochastic LP $\to$ Deterministic LP $\to$ Greedy analytical fallback, followed by a hard analytical feasibility clamp (`safe_clamp`) guaranteeing **0 physical violations**.

---

## Repository Structure

```text
├── src/
│   ├── agent_standard.py   # 24h Deterministic LP + EMA campus load calibration
│   └── agent_hard.py       # Conformal Stochastic LP + Physics/GBDT hybrid forecaster
├── backtest/
│   └── simulate.py         # Standalone closed-loop simulator & benchmark script
├── requirements.txt
└── README.md
