from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .data import CACHE, DATA, PERIODS, THRESHOLD, load_static


class Features:
    """Variables que consultan exclusivamente meses anteriores al objetivo."""

    def __init__(self, panel_path=None):
        self.panel = np.load(CACHE if panel_path is None else panel_path, mmap_mode="r")
        self.static, self.factor = load_static()
        self.n = len(self.factor)
        self.risk = self.panel[:, :, 1] / self.factor
        self.names = {}
        self.neighbors = None

    def _neighbors(self):
        if self.neighbors is None:
            path = DATA / "vecinos_32.npy"
            if path.exists():
                self.neighbors = np.load(path)
            else:
                xy = self.static[:, [1, 0]].astype(np.float64)
                xy[:, 0] *= 111_320 * np.cos(np.deg2rad(-12))
                xy[:, 1] *= 111_320
                _, self.neighbors = cKDTree(xy).query(xy, k=33, workers=4)
                self.neighbors = self.neighbors[:, 1:].astype(np.int32)
                np.save(path, self.neighbors)
        return self.neighbors

    def make(self, month, kind):
        if month < 3:
            raise ValueError("Se requieren al menos tres meses anteriores completos")
        if kind in ("aumentado", "aumentado_contexto"):
            base = self.make(month, "legacy")
            extra_kind = "contexto" if kind == "aumentado_contexto" else "temporal"
            extra = self.make(month, extra_kind)
            self.names[kind] = self.names["legacy"] + self.names[extra_kind][3:]
            return np.column_stack([base, extra[:, 3:]]).astype(np.float32)
        if kind == "legacy":
            history = self.panel[month - 3:month].sum(axis=0)
            x = np.column_stack([self.static, history,
                                 history[:, 0] / self.factor, history[:, 1] / self.factor]).astype(np.float32)
            self.names[kind] = ["latitud", "longitud", "longitud_m"] + [f"suma3_{i}" for i in range(11)] + ["densidad3", "gravedad3"]
            return x
        cols, names = [], []

        def add(name, values):
            cols.append(np.broadcast_to(values, (self.n,)))
            names.append(name)

        if kind != "sin_geografia":
            add("latitud", self.static[:, 0])
            add("longitud", self.static[:, 1])
        add("log_longitud", np.log1p(self.static[:, 2]))
        add("meses_historia_disponible", min(month, 12))
        for lag in (1, 2, 3):
            for var in range(11):
                add(f"lag{lag}_variable{var}", self.panel[month-lag, :, var] / self.factor)
        for window in (3, 6, 12):
            hist = self.panel[max(0, month-window):month]
            risk = self.risk[max(0, month-window):month]
            add(f"conteo_media{window}", hist[:, :, 0].mean(axis=0) / self.factor)
            add(f"riesgo_media{window}", risk.mean(axis=0))
            add(f"riesgo_std{window}", risk.std(axis=0))
            add(f"riesgo_max{window}", risk.max(axis=0))
            add(f"meses_activos{window}", (risk > 0).mean(axis=0))
            add(f"meses_altos{window}", (risk >= THRESHOLD).mean(axis=0))
        add("riesgo_lag6", self.risk[month-6] if month >= 6 else np.nan)
        add("riesgo_lag12", self.risk[month-12] if month >= 12 else np.nan)
        add("tendencia1_2", self.risk[month-1] - self.risk[month-2])
        add("tendencia3_6", self.risk[month-3:month].mean(axis=0) - self.risk[max(0, month-6):max(1, month-3)].mean(axis=0))
        add("ratio1_6", (self.risk[month-1] + .05) / (self.risk[max(0, month-6):month].mean(axis=0) + .05))
        add("mes_sin", np.sin(2*np.pi*(month % 12)/12))
        add("mes_cos", np.cos(2*np.pi*(month % 12)/12))
        if kind in ("contexto", "sin_geografia"):
            neighbors = self._neighbors()
            recent = self.risk[month-1]
            mean3 = self.risk[month-3:month].mean(axis=0)
            for k in (8, 32):
                add(f"vecinos{k}_riesgo_lag1", recent[neighbors[:, :k]].mean(axis=1))
                add(f"vecinos{k}_riesgo_media3", mean3[neighbors[:, :k]].mean(axis=1))
                add(f"vecinos{k}_activos_lag1", (recent[neighbors[:, :k]] > 0).mean(axis=1))
            for lag in (1, 2, 3):
                add(f"cobertura_global_lag{lag}", float((self.risk[month-lag] > 0).mean()))
                add(f"altos_global_lag{lag}", float((self.risk[month-lag] >= THRESHOLD).mean()))
        self.names[kind] = names
        return np.column_stack(cols).astype(np.float32)

    def matrix(self, months, kind):
        return np.concatenate([self.make(m, kind) for m in months])

    def target(self, months):
        from .data import labels
        return labels(self.risk[list(months)]).ravel()

    def verify_causality(self):
        # Todas las ramas de make indexan como máximo month-1. Se comprueba que
        # cambiar el mes objetivo y el futuro no altera ninguna variable.
        month = PERIODS.index("2025-08")
        original_panel, original_risk = self.panel, self.risk
        class PastOnly:
            def __init__(self, values):
                self.values = values
            def __getitem__(self, key):
                first = key[0] if isinstance(key, tuple) else key
                if isinstance(first, slice):
                    assert first.stop is not None and first.stop <= month
                else:
                    assert first < month
                return self.values[key]
        try:
            self.panel, self.risk = PastOnly(original_panel), PastOnly(original_risk)
            for kind in ("legacy", "temporal", "contexto", "sin_geografia", "aumentado", "aumentado_contexto"):
                x = self.make(month, kind)
                assert np.isfinite(x).all()
        finally:
            self.panel, self.risk = original_panel, original_risk
