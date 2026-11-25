"""
agent.py
Solució òptima exacta mitjançant Programació Lineal (MPC a 24h vista)
amb calibratge dinàmic de la línia base del campus (0 infraccions garantides).
"""
import numpy as np
from scipy.optimize import linprog


class BatteryAgent:
    def __init__(self):
        self.potencia_max = 500.0
        self.capacitat_max = 2000.0
        self.eta = 0.90 ** 0.5  # Rendiment one-way (~0.9486833)

        # Perfil horari mitjà de demanda neta (kW) après de les dades històriques del campus
        # [00:00 .. 23:00] per dies laborables i caps de setmana
        self.perfil_laborable = np.array([
            998.0, 998.0, 998.0, 998.0, 998.0, 998.0,
            1165.0, 1255.0, 1345.0, 1410.0, 1455.0, 1465.0,
            1445.0, 1415.0, 1390.0, 1365.0, 1345.0, 1330.0,
            1285.0, 1225.0, 1135.0, 1065.0, 1048.0, 1048.0
        ])
        self.perfil_cap_setmana = np.array([
            922.0, 922.0, 922.0, 922.0, 922.0, 922.0,
            803.0, 780.0, 765.0, 752.0, 740.0, 735.0,
            730.0, 725.0, 732.0, 728.0, 750.0, 775.0,
            798.0, 812.0, 810.0, 798.0, 795.0, 795.0
        ])

        # Factor d'escala adaptatiu observat en temps real
        self.factor_escala_campus = 1.0
        self.historial = []

    def _actualitzar_escala_campus(self, obs: dict):
        """Calibra el nivell de demanda real del campus usant la telemetria recent."""
        dades_5m = obs.get("noves_dades_5min")
        if not dades_5m:
            return
        consum = dades_5m.get("ultim_consum_kw")
        solar = dades_5m.get("ultima_solar_kw")
        if consum is None or solar is None or consum <= 0:
            return

        demanda_obs = max(200.0, float(consum) - float(solar))
        hora_ant = (int(obs["hour"]) - 1) % 24
        dow_ant = int(obs["day_of_week"]) if obs["hour"] > 0 else (int(obs["day_of_week"]) - 1) % 7
        perfil = self.perfil_cap_setmana if dow_ant >= 5 else self.perfil_laborable
        esperada = perfil[hora_ant]

        rati_actual = np.clip(demanda_obs / esperada, 0.5, 1.8)
        # Mitjana mòbil exponencial suau per adaptar-se a setmanes festives o d'estiu
        self.factor_escala_campus = 0.85 * self.factor_escala_campus + 0.15 * rati_actual

    def act(self, obs: dict) -> float:
        self.historial.append(obs)
        self._actualitzar_escala_campus(obs)

        p24 = np.asarray(obs["preus_properes_24h"], dtype=float)
        c24 = np.asarray(obs["co2_properes_24h"], dtype=float)
        H = len(p24)
        soc0 = float(obs["soc_kwh"])

        # Estimem la demanda neta de les properes 24h per ponderar exactament
        # el denominador relatiu de ΔCO2 (%) i ΔCost (%)
        h0 = int(obs["hour"])
        dow0 = int(obs["day_of_week"])
        dem_24 = np.empty(H, dtype=float)
        for k in range(H):
            hk = (h0 + k) % 24
            dow_k = (dow0 + ((h0 + k) // 24)) % 7
            perfil = self.perfil_cap_setmana if dow_k >= 5 else self.perfil_laborable
            dem_24[k] = perfil[hk] * self.factor_escala_campus

        base_co2_24 = max(1.0, float(np.sum(dem_24 * c24)))
        base_cost_24 = max(1.0, float(np.sum(dem_24 * p24)))

        # Funció objectiu exacta alineada amb 0.70 * ΔCO2(%) + 0.30 * ΔCost(%)
        w = 0.70 * (c24 / base_co2_24) + 0.30 * (p24 / base_cost_24)
        c_obj = np.concatenate([w, -w])

        # Restriccions físiques del SOC al llarg de les 24h
        tril = np.tril(np.ones((H, H)))
        A_ub = np.vstack([
            np.hstack([self.eta * tril, -(1.0 / self.eta) * tril]),
            np.hstack([-self.eta * tril, (1.0 / self.eta) * tril]),
        ])
        b_ub = np.concatenate([
            np.full(H, self.capacitat_max - soc0),
            np.full(H, soc0),
        ])

        bounds = [(0.0, self.potencia_max)] * (2 * H)
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success:
            consigna = float(res.x[0] - res.x[H])
        else:
            consigna = 0.0

        # Clamping físic estricte per garantir 0 infraccions
        eps = 1e-4
        espai_disp = max(0.0, self.capacitat_max - soc0)
        energia_disp = max(0.0, soc0)
        max_carrega = max(0.0, min(self.potencia_max, espai_disp / self.eta - eps))
        max_descarrega = max(0.0, min(self.potencia_max, energia_disp * self.eta - eps))

        return float(np.clip(consigna, -max_descarrega, max_carrega))
