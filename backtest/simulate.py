"""
Standalone closed-loop backtest harness for BatteryAgent (Standard & Hard modes).
Generates synthetic realistic campus load, PV generation, and PVPC/CO2 spot curves
to verify execution latency, zero-violation constraints, and arbitrage performance.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_standard import BatteryAgent as StandardAgent
from agent_hard import BatteryAgent as HardAgent

def generate_synthetic_week(seed: int = 42, hard_mode: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-05-04 00:00:00", periods=168, freq="1h")
    hours = idx.hour.to_numpy()
    is_wk = (idx.dayofweek >= 5).to_numpy()

    # Campus base load (kW)
    load = np.where(
        is_wk,
        780.0 + 90.0 * np.cos((hours - 2) * np.pi / 12.0),
        1220.0 + 240.0 * np.sin(np.clip((hours - 6) * np.pi / 14.0, 0, np.pi)),
    ) + rng.normal(0, 35.0, size=168)

    # Clear-sky solar curve (scaled 12x in Hard Mode)
    solar_unit = np.maximum(0.0, 180.0 * np.sin(np.clip((hours - 6) * np.pi / 13.0, 0, np.pi)))
    cloud_factor = np.clip(0.85 + 0.15 * np.sin(np.arange(168) / 11.0) + rng.normal(0, 0.08, 168), 0.2, 1.1)
    solar = solar_unit * cloud_factor * (12.0 if hard_mode else 1.0)

    # Spot PVPC price (€/kWh) & Grid CO2 intensity (kg/kWh)
    price = np.clip(
        0.14 - 0.06 * np.sin(np.clip((hours - 8) * np.pi / 9.0, 0, np.pi))
        + 0.08 * np.exp(-((hours - 20.5) ** 2) / 4.5)
        - 0.03 * is_wk
        + rng.normal(0, 0.012, 168),
        0.01,
        0.38,
    )
    co2 = np.clip(
        0.095 - 0.045 * np.sin(np.clip((hours - 8) * np.pi / 9.0, 0, np.pi))
        + 0.035 * np.exp(-((hours - 21.0) ** 2) / 5.0)
        + rng.normal(0, 0.008, 168),
        0.02,
        0.22,
    )

    return pd.DataFrame({
        "timestamp": idx,
        "hour": hours,
        "day_of_week": idx.dayofweek,
        "load_kw": load,
        "solar_kw": solar,
        "price": price,
        "co2": co2,
    })

def run_backtest(hard_mode: bool = False):
    df = generate_synthetic_week(hard_mode=hard_mode)
    agent = HardAgent() if hard_mode else StandardAgent()

    cap = 6000.0 if hard_mode else 2000.0
    pmax = 1500.0 if hard_mode else 500.0
    soc = 0.5 * cap
    eta = 0.90 ** 0.5
    known_h = 3 if hard_mode else 24

    base_co2 = base_cost = opt_co2 = opt_cost = 0.0
    violations = 0
    latencies_ms = []

    for t in range(len(df)):
        row = df.iloc[t]
        end_idx = min(len(df), t + known_h)
        obs = {
            "timestamp": row["timestamp"],
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "soc_kwh": soc,
            "cap_max_kwh": cap,
            "power_max_kw": pmax,
            "preu_actual": float(row["price"]),
            "co2_actual": float(row["co2"]),
            "preus_properes_24h": df["price"].iloc[t:end_idx].tolist(),
            "co2_properes_24h": df["co2"].iloc[t:end_idx].tolist(),
            "noves_dades_5min": {
                "ultim_consum_kw": float(df["load_kw"].iloc[max(0, t - 1)]),
                "ultima_solar_kw": float(df["solar_kw"].iloc[max(0, t - 1)]),
            },
        }

        t0 = time.perf_counter()
        u = agent.act(obs)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        next_soc = soc + (u * eta if u >= 0 else u / eta)
        if abs(u) > pmax + 1e-3 or next_soc < -1e-3 or next_soc > cap + 1e-3:
            violations += 1
        soc = float(np.clip(next_soc, 0.0, cap))

        net_base = max(0.0, row["load_kw"] - row["solar_kw"])
        net_opt = max(0.0, row["load_kw"] - row["solar_kw"] + u)
        base_co2 += net_base * row["co2"]
        base_cost += net_base * row["price"]
        opt_co2 += net_opt * row["co2"]
        opt_cost += net_opt * row["price"]

    d_co2 = (base_co2 - opt_co2) / base_co2 * 100.0
    d_cost = (base_cost - opt_cost) / base_cost * 100.0
    score = 0.70 * d_co2 + 0.30 * d_cost
    mode_name = "HARD MODE (6 MWh / 12x Solar)" if hard_mode else "STANDARD MODE (2 MWh)"
    print(f"[{mode_name}] Score: {score:.2f} | dCO2: {d_co2:.2f}% | dCost: {d_cost:.2f}% | "
          f"Violations: {violations} | Avg Latency: {np.mean(latencies_ms):.2f} ms")

if __name__ == "__main__":
    run_backtest(hard_mode=False)
    run_backtest(hard_mode=True)
