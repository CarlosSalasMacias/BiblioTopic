"""HJ-Biplot (Galindo, 1986)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class HJBiplot:
    """X (filas x columnas) -> marcadores fila J = U·D y columna H = V·D (misma calidad para ambos)."""

    def __init__(self, data: pd.DataFrame, scale=True):
        self.data = data
        X = data.values.astype(float)
        X = X - X.mean(axis=0)
        if scale:
            sd = X.std(axis=0, ddof=1)
            sd[sd == 0] = 1
            X = X / sd
        U, d, Vt = np.linalg.svd(X, full_matrices=False)
        self.d = d
        self.inertia = d ** 2 / (d ** 2).sum()
        self.rows = pd.DataFrame(U * d, index=data.index)
        self.cols = pd.DataFrame(Vt.T * d, index=data.columns)
        # calidad de representación en el plano 1-2 (contribución relativa del factor al elemento)
        rq = (self.rows ** 2).div((self.rows ** 2).sum(axis=1) + 1e-12, axis=0)
        cq = (self.cols ** 2).div((self.cols ** 2).sum(axis=1) + 1e-12, axis=0)
        self.row_quality = rq.iloc[:, :2].sum(axis=1)
        self.col_quality = cq.iloc[:, :2].sum(axis=1)

    def summary(self):
        return pd.DataFrame({
            "Axis": [f"Axis {i + 1}" for i in range(len(self.d))],
            "Eigenvalue": (self.d ** 2).round(3),
            "Explained inertia (%)": (100 * self.inertia).round(2),
            "Cumulative (%)": (100 * np.cumsum(self.inertia)).round(2),
        })

    def column_angles(self):
        """Cosenos entre vectores columna en el plano 1-2 (≈ correlaciones)."""
        H = self.cols.iloc[:, :2].values
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        return pd.DataFrame(Hn @ Hn.T, index=self.cols.index, columns=self.cols.index)


def make_periods(years: pd.Series, width=5, min_share=0.05) -> pd.Series:
    """Agrupa años en periodos de `width` años terminando en el último año;
    los periodos iniciales con menos de `min_share` del corpus se fusionan."""
    y = years.astype(int)
    last = int(y.max())
    edges = []
    hi = last
    while hi >= y.min():
        edges.append((hi - width + 1, hi))
        hi -= width
    edges = edges[::-1]
    # fusionar los primeros periodos hasta alcanzar min_share
    n = len(y)
    while len(edges) > 1 and ((y >= edges[0][0]) & (y <= edges[0][1])).sum() < min_share * n:
        edges[1] = (edges[0][0], edges[1][1])
        edges.pop(0)
    edges[0] = (int(y.min()), edges[0][1])
    labels = []
    for lo, hi in edges:
        labels.append(f"{lo}-{hi}" if lo != hi else str(lo))

    def lab(v):
        for (lo, hi), l in zip(edges, labels):
            if lo <= v <= hi:
                return l
        return labels[-1]
    return y.map(lab), labels
