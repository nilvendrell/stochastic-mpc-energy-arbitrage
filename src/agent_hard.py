"""
agent.py — BatteryAgent per al MODE DIFÍCIL (compatible també amb el Mode Estàndard).
Repte UPC Sostenible · Datathon 2026

MPC amb LP exacte (HiGHS) sobre un horitzó de 24 h construït amb:
  * preu i CO2: 3 h conegudes + previsió (model / estacional) per a la resta
  * càrrega neta = consum previst - solar prevista (x12), amb índex de cel clar
Cost unificat normalitzat igual que el Score; clamp exacte -> 0 infraccions.
Llibreries: numpy, pandas, scipy, json, pathlib, math, time (+ joblib/xgboost opcionals).
"""
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
try:
    from scipy.optimize import linprog
except Exception:
    linprog = None
try:
    import joblib
except Exception:
    joblib = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

ETA = 0.9
ETA_C = math.sqrt(ETA)
ETA_D = math.sqrt(ETA)
MARGIN_KW = 0.01

DEFAULT_PARAMS = {
    "w_co2": 0.70, "w_cost": 0.30,
    # referències ponderades pel baseline (mode difícil: max(consum - 12*solar, 0))
    "co2_ref_hard": 0.0933, "price_ref_hard": 0.1528,
    "co2_ref_std": 0.0840, "price_ref_std": 0.1430,
    "plan_h": 24,
    "terminal_q": 0.5, "terminal_mult": 1.0,
    "load_tau": 6.0,          # decaïment (h) de la correcció per persistència del consum
    "kt_tau": 4.0,            # decaïment (h) de l'índex de cel clar
    "mkt_tau": 24.0,          # decaïment (h) del biaix de nivell de preu/CO2
    "mkt_mode": "blend",
    "mkt_blend": 0.5,         # pes d'"ahir" a la previsió base del preu (la resta: perfil)
    "co2_blend": 0.5,         # idem per al CO2
    "co2_tau": 24.0,          # decaïment (h) del biaix de nivell del CO2      # blend | yesterday | profile
    "net_safety": 0.05,       # fracció de marge sobre la càrrega neta de l'hora actual en descarregar
    "time_budget_s": 25.0,    # si el temps acumulat supera això -> sense models ML (LP segueix)
    "stoch_budget_s": 35.0,   # si el supera -> LP determinista (més ràpid)
    "lp_budget_s": 50.0,      # si el supera -> heurística ràpida
    "resid_w": 1.0,
    "stochastic": True,       # MPC estocàstic amb escenaris d'error de la càrrega neta
    "err_by_hour": True,
    "adapt_vol": True,        # escala adaptativa dels escenaris segons l'error recent (estil conformal)
    "vol_alpha": 0.3, "vol_lo": 0.3, "vol_hi": 3.0, "vol_min_typ": 20.0,      # quantils d'error condicionats a l'hora del dia
    "stoch_h": 24,            # hores amb incertesa explícita
    "err_mult": 1.0,
    "err_scale_std": 0.5,
    "drain_at_end": True,
    "use_models": True,
}

HOLIDAYS = {
    "2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08",
    "2025-12-25", "2025-12-26", "2026-01-01", "2026-01-06", "2026-04-03", "2026-04-06",
    "2026-05-01", "2026-05-25", "2026-06-24", "2026-08-15", "2026-09-11", "2026-09-24",
}


def daytype(ts):
    d = ts.strftime("%Y-%m-%d")
    return 1 if (ts.dayofweek >= 5 or d in HOLIDAYS) else 0


def cos_zenith(ts):
    lat = math.radians(41.39)
    doy = ts.dayofyear
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365.0)
    m, d = ts.month, ts.day
    tz = 2.0 if (4 <= m <= 9 or (m == 3 and d >= 29) or (m == 10 and d < 26)) else 1.0
    st = ts.hour + 0.5 - tz + 2.11 / 15.0
    ha = math.radians(15.0 * (st - 12.0))
    return max(0.0, math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha))


# ------------------------------------------------------------------ optimitzador
class LPSolver:
    def __init__(self):
        self.cache = {}

    def _mats(self, H):
        if H not in self.cache:
            I, Z, L = np.eye(H), np.zeros((H, H)), np.tril(np.ones((H, H)))
            Aeq = np.hstack([-I, I, I, -I])                       # g - x - ch + dis = n
            Asoc = np.hstack([ETA_C * L, -L / ETA_D, Z, Z])
            self.cache[H] = (Aeq, np.vstack([Asoc, -Asoc]))
        return self.cache[H]

    def solve(self, soc0, c, n, v_term, cap, pmax, e=0.0):
        H = len(c)
        Aeq, Aub = self._mats(H)
        cost = np.concatenate([np.full(H, -v_term * ETA_C), np.full(H, v_term / ETA_D),
                               np.asarray(c, float), np.full(H, -e)])
        bub = np.concatenate([np.full(H, cap - soc0), np.full(H, soc0)])
        pm = pmax - MARGIN_KW
        r = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=np.asarray(n, float),
                    bounds=[(0, pm)] * (2 * H) + [(0, None)] * (2 * H), method="highs")
        if r.status != 0 or r.x is None:
            raise RuntimeError("LP")
        return float(r.x[0] - r.x[H])


class StochLPSolver:
    """MPC estocàstic: minimitza el cost ESPERAT sobre S escenaris d'error de la càrrega neta.
    g[h,s] >= n_h + E[h,s] + ch_h - dis_h ,  cost = sum_h c_h sum_s w_s g[h,s]
    (problema tipus newsvendor: descarregar més que la càrrega real llença energia)."""

    def __init__(self):
        self.cache = {}

    def _mats(self, H, S):
        key = (H, S)
        if key not in self.cache:
            I, L = np.eye(H), np.tril(np.ones((H, H)))
            rep = np.repeat(I, S, axis=0)                          # (H*S, H)
            G = -np.eye(H * S)
            A1 = np.hstack([rep, -rep, G])                         # ch - dis - g <= -(n+E)
            Asoc = np.hstack([ETA_C * L, -L / ETA_D, np.zeros((H, H * S))])
            self.cache[key] = np.vstack([A1, Asoc, -Asoc])
        return self.cache[key]

    def solve(self, soc0, c, n, E, w, v_term, cap, pmax):
        H, S = E.shape
        Aub = self._mats(H, S)
        cost = np.concatenate([np.full(H, -v_term * ETA_C), np.full(H, v_term / ETA_D),
                               (np.asarray(c, float)[:, None] * np.asarray(w, float)[None, :]).ravel()])
        b1 = -(np.asarray(n, float)[:, None] + E).ravel()
        bub = np.concatenate([b1, np.full(H, cap - soc0), np.full(H, soc0)])
        pm = pmax - MARGIN_KW
        r = linprog(cost, A_ub=Aub, b_ub=bub, bounds=[(0, pm)] * (2 * H) + [(0, None)] * (H * S),
                    method="highs")
        if r.status != 0 or r.x is None:
            raise RuntimeError("SLP")
        return float(r.x[0] - r.x[H])


def greedy_fallback(soc, c, n, cap, pmax):
    """Reserva sense scipy: carrega amb excedent o hores barates, descarrega fins a la càrrega neta en hores cares."""
    c0, med = c[0], float(np.median(c))
    if n[0] < 0:
        return min(-n[0], pmax)
    if c0 < 0.85 * med:
        return pmax
    if c0 > 1.1 * med:
        return -min(pmax, max(n[0], 0.0))
    return 0.0


def safe_clamp(p, soc, cap, pmax):
    if not np.isfinite(p):
        return 0.0
    if p > 0:
        return float(max(0.0, min(p, min(pmax, (cap - soc) / ETA_C) - MARGIN_KW)))
    if p < 0:
        return float(-max(0.0, min(-p, min(pmax, soc * ETA_D) - MARGIN_KW)))
    return 0.0


# ------------------------------------------------------------------ agent
class BatteryAgent:
    def __init__(self, params=None):
        self.p = dict(DEFAULT_PARAMS)
        base = Path(__file__).parent
        try:
            if (base / "params.json").exists():
                self.p.update(json.loads((base / "params.json").read_text(encoding="utf-8")))
        except Exception:
            pass
        if params:
            self.p.update(params)
        self.prof = None
        try:
            pf = Path(self.p.get("model_dir") or base) / "profiles_hard.json"
            if pf.exists():
                self.prof = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            self.prof = None
        self.models = {}
        if self.p["use_models"]:
            self._load_models(Path(self.p.get("model_dir") or base))
            keep = self.p.get("models_use")
            if keep is not None:
                self.models = {k: v for k, v in self.models.items() if k in keep}
        self.lp = LPSolver() if linprog is not None else None
        self.slp = StochLPSolver() if linprog is not None else None
        self.hist = {}       # ts -> dict(load, pv, irr, tamb, tpan)
        self.mkt = {}        # ts -> (price, co2)
        self.prev_ts = None
        self.elapsed = 0.0
        self.fast = False
        self.oracle_net = None   # backtest: fn(ts, H) -> net real
        self.oracle_mkt = None   # backtest: fn(ts, H) -> (preus, co2) reals

    # ---------------------------------------------------------- models
    def _load_models(self, d):
        for name in ("load", "pv", "price", "co2"):
            try:
                fj = d / f"m_{name}.json"
                fp = d / f"m_{name}.pkl"
                if xgb is not None and fj.exists():
                    b = xgb.Booster()
                    b.load_model(str(fj))
                    self.models[name] = ("xgb", b)
                elif joblib is not None and fp.exists():
                    self.models[name] = ("skl", joblib.load(fp))
            except Exception:
                pass

    def _predict(self, name, X):
        kind, m = self.models[name]
        if kind == "xgb":
            return np.asarray(m.inplace_predict(X), float).reshape(-1)
        return np.asarray(m.predict(X), float).reshape(-1)

    # ---------------------------------------------------------- perfils
    def _pf(self, table, ts, default):
        try:
            return float(self.prof[table][f"{ts.month}_{daytype(ts)}_{ts.hour}"])
        except Exception:
            return default

    # ---------------------------------------------------------- previsions
    def _hours(self, ts, H):
        one = pd.Timedelta(hours=1)
        return [ts + h * one for h in range(H)]

    def simple_market(self, ts, H, known_p, known_c):
        """Previsió base: barreja (ahir, perfil) + biaix de nivell de les hores conegudes."""
        k = len(known_p)
        hrs = self._hours(ts, H)
        one = pd.Timedelta(hours=1)
        out = []
        for idx, known, dflt in ((0, known_p, 0.14), (1, known_c, 0.09)):
            base = []
            for T in hrs:
                y = self.mkt.get(T - 24 * one)
                prof = self._pf("price" if idx == 0 else "co2", T, dflt)
                by = self.p["mkt_blend"] if idx == 0 else self.p["co2_blend"]
                if self.p["mkt_mode"] == "profile" or y is None:
                    base.append(prof)
                elif self.p["mkt_mode"] == "yesterday":
                    base.append(y[idx])
                else:
                    base.append(by * y[idx] + (1 - by) * prof)
            base = np.asarray(base, float)
            bias = float(np.mean(np.asarray(known, float) - base[:k])) if k else 0.0
            h = np.arange(H)
            tau = self.p["mkt_tau"] if idx == 0 else self.p["co2_tau"]
            v = base + bias * np.exp(-(h - (k - 1)).clip(min=0) / tau)
            v[:k] = np.asarray(known, float)[:H]
            out.append(np.maximum(v, 1e-4))
        return out[0], out[1]

    def forecast_market(self, ts, H, known_p, known_c):
        k = len(known_p)
        bp, bc = self.simple_market(ts, H, known_p, known_c)
        if k >= H or self.fast:
            return bp, bc
        hrs = self._hours(ts, H)
        out = [bp, bc]
        if "price" in self.models or "co2" in self.models:
            try:
                F = self.market_features(ts, hrs, known_p, known_c)
                for j, name in ((0, "price"), (1, "co2")):
                    if name in self.models:
                        r = self._predict(name, np.column_stack([F, out[j]]))
                        v = out[j].copy()
                        rw = self.p.get(f"resid_w_{name}", self.p["resid_w"])
                        v[k:] = np.maximum(v[k:] + rw * r[k:], 1e-4)
                        out[j] = v
            except Exception:
                out = [bp, bc]
        return out[0], out[1]

    def market_features(self, ts, hrs, known_p, known_c):
        one = pd.Timedelta(hours=1)
        k = len(known_p)
        kp = list(known_p) + [np.nan] * (3 - k)
        kc = list(known_c) + [np.nan] * (3 - k)
        past = [self.mkt.get(ts - i * one) for i in range(1, 25)]
        pp = [x[0] for x in past if x is not None]
        pc = [x[1] for x in past if x is not None]
        mp = float(np.mean(pp)) if pp else np.nan
        mc = float(np.mean(pc)) if pc else np.nan
        rows = []
        for h, T in enumerate(hrs):
            y = self.mkt.get(T - 24 * one)
            rows.append([h, T.hour, T.dayofweek, T.month, daytype(T), cos_zenith(T),
                         kp[0], kp[1], kp[2], kc[0], kc[1], kc[2], mp, mc,
                         y[0] if y else np.nan, y[1] if y else np.nan,
                         self._pf("price", T, np.nan), self._pf("co2", T, np.nan)])
        return np.asarray(rows, float)

    def simple_net(self, ts, H, pv_scale):
        """Previsió base: consum = perfil + persistència; solar = cel clar x índex de claredat."""
        one = pd.Timedelta(hours=1)
        hrs = self._hours(ts, H)
        last = self.hist.get(ts - one)
        lp = np.array([self._pf("load", T, 1100.0) for T in hrs])
        if last is not None and np.isfinite(last["load"]):
            if last["load"] < 200:          # tall de càrrega -> persistència
                lp = np.full(H, last["load"])
            else:
                b = last["load"] - self._pf("load", ts - one, last["load"])
                lp = lp + b * np.exp(-np.arange(H) / self.p["load_tau"])
        cs = np.array([self._pf("pvcs", T, 0.0) for T in hrs])
        kclim = np.array([self._pf("kt", T, 0.7) for T in hrs])
        kt_now = None
        if last is not None:
            cs_last = self._pf("pvcs", ts - one, 0.0)
            if cs_last > 5.0:
                kt_now = float(np.clip(last["pv"] / pv_scale / cs_last, 0.0, 1.3))
        if kt_now is None:
            kt_now = self._last_kt
        kt = kclim if kt_now is None else kclim + (kt_now - kclim) * np.exp(-np.arange(1, H + 1) / self.p["kt_tau"])
        if kt_now is not None:
            self._last_kt = kt_now
        pvu = np.clip(cs * kt, 0, None)
        return lp, pvu

    def forecast_net(self, ts, H, pv_scale):
        """Càrrega neta = consum - solar (escalada). ML = correcció residual sobre la base."""
        lp, pvu = self.simple_net(ts, H, pv_scale)
        if not self.fast and ("load" in self.models or "pv" in self.models):
            try:
                F = self.net_features(ts, self._hours(ts, H), pv_scale)
                if "load" in self.models:
                    lp = lp + self.p["resid_w"] * self._predict("load", np.column_stack([F, lp]))
                if "pv" in self.models:
                    pvu = np.clip(pvu + self.p["resid_w"] * self._predict("pv", np.column_stack([F, pvu])), 0, None)
            except Exception:
                pass
        pv = pvu * pv_scale
        return lp - pv, lp, pv

    _last_kt = None

    def net_features(self, ts, hrs, pv_scale):
        one = pd.Timedelta(hours=1)
        last = self.hist.get(ts - one) or {}
        vals = [self.hist.get(ts - i * one) for i in range(1, 25)]
        ml = [v["load"] for v in vals if v is not None]
        mean_load = float(np.mean(ml)) if ml else np.nan
        cs_last = self._pf("pvcs", ts - one, np.nan)
        lpv = last.get("pv", np.nan) / pv_scale if last else np.nan
        kt_now = lpv / cs_last if (cs_last and cs_last > 5) else np.nan
        rows = []
        for h, T in enumerate(hrs):
            y = self.hist.get(T - 24 * one)
            rows.append([h, T.hour, T.dayofweek, T.month, daytype(T), cos_zenith(T),
                         last.get("load", np.nan), lpv, last.get("irr", np.nan),
                         last.get("tamb", np.nan), last.get("tpan", np.nan), kt_now, mean_load,
                         y["load"] if y else np.nan, (y["pv"] / pv_scale) if y else np.nan,
                         self._pf("load", T, np.nan), self._pf("pvcs", T, np.nan), self._pf("kt", T, np.nan)])
        return np.asarray(rows, float)

    _pred0 = None
    vol = 1.0

    def _typ_err(self, h, hod):
        try:
            qh = self._qh if self._qh is not None else np.asarray(self.prof["net_err"]["q_hod"], float)
            self._qh = qh
            return float(np.mean(np.abs(qh[h, hod])))
        except Exception:
            return 0.0

    _qh = None

    def _update_vol(self, ts, pv_scale):
        """Escalat adaptatiu de la incertesa: compara la previsió de l'hora anterior amb el real."""
        if not self.p["adapt_vol"] or self._pred0 is None or ts is None:
            return
        pts, pred = self._pred0
        real = self.hist.get(pts)
        if real is None or not np.isfinite(real["load"]):
            return
        typ = self._typ_err(0, pts.hour)
        if typ < self.p["vol_min_typ"]:
            return
        err = abs((real["load"] - real["pv"]) - pred)
        a = self.p["vol_alpha"]
        self.vol = float(np.clip((1 - a) * self.vol + a * err / typ, self.p["vol_lo"], self.p["vol_hi"]))

    def _err_scen(self, H, pv_scale, ts=None):
        """Escenaris d'error de la càrrega neta per horitzó (quantils empírics del train).
        Si n'hi ha, usa quantils condicionats a l'hora del dia objectiu (heteroscedàstics)."""
        try:
            w = np.asarray(self.prof["net_err"]["w"], float)
            qh = self.prof["net_err"].get("q_hod") if self.p["err_by_hour"] else None
            if qh is not None and ts is not None:
                qh = np.asarray(qh, float)                                # (24 h, 24 hora, S)
                q = np.vstack([qh[min(h, 23), (ts.hour + h) % 24] for h in range(H)])
            else:
                q = np.asarray(self.prof["net_err"]["q"], float)        # (24, S) en kW (mode difícil)
            if pv_scale < 12.0:
                q = q * self.p["err_scale_std"]
            q = q * self.p["err_mult"]
            if self.p["adapt_vol"]:
                q = q * self.vol
            Hq = q.shape[0]
            E = q[:H] if Hq >= H else np.vstack([q[min(h, Hq - 1)] for h in range(H)])
            nh = int(self.p["stoch_h"])
            if nh < H:
                E[nh:] = 0.0
            return E, w
        except Exception:
            return None

    # ---------------------------------------------------------- memòria
    def _record(self, obs, ts, prices, co2):
        d = obs.get("noves_dades_5min") or {}

        def g(k, dflt=np.nan):
            try:
                v = float(d.get(k, dflt))
                return v if np.isfinite(v) else dflt
            except Exception:
                return dflt

        if self.prev_ts is not None and ts is not None:
            self.hist[self.prev_ts] = {"load": g("ultim_consum_kw"), "pv": g("ultima_solar_kw", 0.0),
                                       "irr": g("irradiancia_wm2"), "tamb": g("temp_ambient_c"),
                                       "tpan": g("temp_panel_c")}
        if ts is not None and len(prices):
            self.mkt[ts] = (float(prices[0]), float(co2[0]))
        self.prev_ts = ts

    # ---------------------------------------------------------- política
    def act(self, obs: dict) -> float:
        t0 = time.perf_counter()
        cap = float(obs.get("cap_max_kwh", 2000.0) or 2000.0)
        pmax = float(obs.get("power_max_kw", 500.0) or 500.0)
        soc = 0.5 * cap
        try:
            soc = float(obs.get("soc_kwh", float("nan")))
            if not np.isfinite(soc):
                soc = float(obs.get("soc_pct", 0.5)) * cap
            soc = min(max(soc, 0.0), cap)
            hard = cap > 2500.0
            self.fast = self.elapsed > self.p["time_budget_s"]
            pv_scale = 12.0 if hard else 1.0
            ts = pd.Timestamp(obs.get("timestamp")) if pd is not None else None

            p_now = float(obs.get("preu_actual", 0.14))
            c_now = float(obs.get("co2_actual", 0.09))
            kp = obs.get("preus_properes_24h")
            kc = obs.get("co2_properes_24h")
            kp = np.asarray(kp if kp is not None and len(kp) else [p_now], float).reshape(-1)
            kc = np.asarray(kc if kc is not None and len(kc) else [c_now], float).reshape(-1)
            k = int(min(len(kp), len(kc)))
            kp = np.where(np.isfinite(kp[:k]), kp[:k], p_now)
            kc = np.where(np.isfinite(kc[:k]), kc[:k], c_now)
            self._record(obs, ts, kp, kc)

            full_len = int(self.p.get("known_h", 3 if hard else 24))
            end = k < full_len and self.p["drain_at_end"]       # llistes escurçades -> final
            H = k if end else int(self.p["plan_h"])

            if self.oracle_mkt is not None:
                prices, co2 = self.oracle_mkt(ts, H)
            else:
                prices, co2 = self.forecast_market(ts, H, kp, kc)
            if self.oracle_net is not None:
                n = np.asarray(self.oracle_net(ts, H), float)
            else:
                n = self.forecast_net(ts, H, pv_scale)[0]
                self._update_vol(ts, pv_scale)
                self._pred0 = (ts, float(n[0]))

            if self.p["net_safety"] > 0 and n[0] > 0:
                n = n.copy()
                n[0] = n[0] * (1.0 - self.p["net_safety"])

            cr = self.p["co2_ref_hard"] if hard else self.p["co2_ref_std"]
            pr = self.p["price_ref_hard"] if hard else self.p["price_ref_std"]
            c = self.p["w_co2"] * np.asarray(co2) / cr + self.p["w_cost"] * np.asarray(prices) / pr
            v_term = 0.0 if end else ETA_D * float(np.quantile(c, self.p["terminal_q"])) * self.p["terminal_mult"]

            p = None
            E = self._err_scen(H, pv_scale, ts) if (self.p["stochastic"] and self.oracle_net is None
                                                 and self.elapsed < self.p["stoch_budget_s"]) else None
            if E is not None and self.slp is not None and self.elapsed < self.p["lp_budget_s"]:
                try:
                    p = self.slp.solve(soc, c, n, E[0], E[1], v_term, cap, pmax)
                except Exception:
                    p = None
            if p is None and self.lp is not None and self.elapsed < self.p["lp_budget_s"]:
                try:
                    p = self.lp.solve(soc, c, n, v_term, cap, pmax)
                except Exception:
                    p = None
            if p is None:
                p = greedy_fallback(soc, c, n, cap, pmax)
            return safe_clamp(p, soc, cap, pmax)
        except Exception:
            return 0.0
        finally:
            self.elapsed += time.perf_counter() - t0
