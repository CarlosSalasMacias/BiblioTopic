"""Mapa temático (Callon et al., 1991) y evolución temática (diagrama de Sankey) a partir de palabras clave."""
from __future__ import annotations

import itertools
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle

from . import figures as F
from .biblio import split_list


def keyword_clusters(kw_lists, n_terms=250, min_freq=3, seed=7):
    """Comunidades de Louvain sobre la red de co-ocurrencia ponderada por índice de equivalencia."""
    freq = Counter(itertools.chain.from_iterable(kw_lists))
    keep = {k for k, f in freq.most_common(n_terms) if f >= min_freq}
    co = Counter()
    for ks in kw_lists:
        ks = sorted(set(k for k in ks if k in keep))
        for a, b in itertools.combinations(ks, 2):
            co[(a, b)] += 1
    G = nx.Graph()
    for (a, b), c in co.items():
        e = c * c / (freq[a] * freq[b])  # índice de equivalencia
        G.add_edge(a, b, weight=e, count=c)
    if G.number_of_edges() == 0:
        return [], freq, G
    comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    comms = [c for c in comms if len(c) >= 3]
    comms.sort(key=lambda c: -sum(freq[k] for k in c))
    return comms, freq, G


def callon_map(kw_lists):
    comms, freq, G = keyword_clusters(kw_lists)
    cid = {k: i for i, c in enumerate(comms) for k in c}
    rows = []
    for i, c in enumerate(comms):
        internal = sum(d["weight"] for a, b, d in G.edges(c, data=True) if cid.get(a) == i and cid.get(b) == i)
        external = sum(d["weight"] for a, b, d in G.edges(c, data=True) if (cid.get(a) == i) != (cid.get(b) == i))
        top = sorted(c, key=lambda k: -freq[k])
        rows.append({"Cluster": i + 1, "Label": ", ".join(top[:3]), "Keywords": "; ".join(top[:10]),
                     "Centrality": 10 * external, "Density": 100 * internal / len(c),
                     "Occurrences": sum(freq[k] for k in c), "Size": len(c)})
    t = pd.DataFrame(rows)
    if len(t):
        cx, cy = t["Centrality"].median(), t["Density"].median()
        t["Quadrant"] = np.select(
            [(t["Centrality"] >= cx) & (t["Density"] >= cy), (t["Centrality"] < cx) & (t["Density"] >= cy),
             (t["Centrality"] < cx) & (t["Density"] < cy)],
            ["Motor themes", "Niche themes", "Emerging or declining themes"], "Basic themes")
    return t


def plot_callon(path, t, title="Thematic map"):
    fig, ax = plt.subplots(figsize=(7, 6))
    x, y = t["Centrality"].rank(pct=True), t["Density"].rank(pct=True)  # rangos, como bibliometrix
    ax.axvline(0.5, color="#999999", lw=0.8, ls="--")
    ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
    s = 200 + 2600 * (t["Occurrences"] / t["Occurrences"].max())
    ax.scatter(x, y, s=s, c=[F.PALETTE[i % len(F.PALETTE)] for i in range(len(t))], alpha=0.55,
               edgecolors="white", lw=1, zorder=3)
    texts = []
    for xi, yi, lab in zip(x, y, t["Label"]):
        tx = ax.text(xi, yi, F.wrap(lab, 22), ha="center", va="center", fontsize=7.5, zorder=4)
        tx.set_path_effects([F.pe.withStroke(linewidth=1.2, foreground=(1, 1, 1, 0.75))])
        texts.append(tx)
    F._repel(ax, texts)
    for (qx, qy, txt) in [(0.98, 0.98, "Motor themes"), (0.02, 0.98, "Niche themes"),
                          (0.02, 0.02, "Emerging or\ndeclining themes"), (0.98, 0.02, "Basic themes")]:
        ax.text(qx, qy, txt, transform=ax.transAxes, ha="right" if qx > 0.5 else "left",
                va="top" if qy > 0.5 else "bottom", fontsize=9, color="#777777", style="italic")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Relevance degree (centrality, rank)")
    ax.set_ylabel("Development degree (density, rank)")
    return F._save(fig, path)


# ------------------------------------------------------------- evolución temática
def evolution(df, kw_field, periods, plabels, min_inclusion=0.1):
    """Clusters por periodo y vínculos por índice de inclusión ponderado (Cobo et al., 2011)."""
    layers = []
    for p in plabels:
        sub = df[periods.values == p][kw_field].map(split_list).tolist()
        comms, freq, _ = keyword_clusters(sub, n_terms=150, min_freq=2)
        layers.append([{"period": p, "label": sorted(c, key=lambda k: -freq[k])[0],
                        "top": ", ".join(sorted(c, key=lambda k: -freq[k])[:3]), "kw": set(c),
                        "occ": sum(freq[k] for k in c)} for c in comms[:8]])
    links = []
    for t in range(len(layers) - 1):
        for i, a in enumerate(layers[t]):
            for j, b in enumerate(layers[t + 1]):
                inter = a["kw"] & b["kw"]
                if not inter:
                    continue
                inc = len(inter) / min(len(a["kw"]), len(b["kw"]))
                if inc >= min_inclusion:
                    links.append({"t": t, "i": i, "j": j, "inclusion": inc, "shared": ", ".join(sorted(inter)[:5])})
    return layers, links


def plot_sankey(path, layers, links, title="Thematic evolution"):
    nper = len(layers)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    gap = 0.02
    pos = {}
    for t, layer in enumerate(layers):
        tot = sum(n["occ"] for n in layer) or 1
        h_avail = 1 - gap * (len(layer) - 1)
        y = 1.0
        for i, n in enumerate(layer):
            h = h_avail * n["occ"] / tot
            pos[(t, i)] = (t, y - h, h)
            y -= h + gap
    w = 0.12
    colors, palette_of = {}, {}
    for (t, i), (x, y0, h) in pos.items():
        lab = layers[t][i]["label"]
        if lab not in palette_of:  # mismo color para el mismo tema en todos los periodos
            palette_of[lab] = F.PALETTE[len(palette_of) % len(F.PALETTE)]
        c = palette_of[lab]
        colors[(t, i)] = c
        ax.add_patch(Rectangle((x - w / 2, y0), w, h, color=c, zorder=3))
        last = t == nper - 1
        ax.text(x + (-w / 2 - 0.02 if last else w / 2 + 0.02), y0 + h / 2, F.wrap(lab, 16),
                ha="right" if last else "left", va="center", fontsize=6.8, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    out_off = {k: 0.0 for k in pos}
    in_off = {k: 0.0 for k in pos}
    for lk in sorted(links, key=lambda l: -l["inclusion"]):
        a, b = (lk["t"], lk["i"]), (lk["t"] + 1, lk["j"])
        (xa, ya, ha), (xb, yb, hb) = pos[a], pos[b]
        th = min(ha, hb) * lk["inclusion"] * 0.9
        y1 = ya + ha - out_off[a] - th
        y2 = yb + hb - in_off[b] - th
        out_off[a] += th
        in_off[b] += th
        x1, x2 = xa + w / 2, xb - w / 2
        xm = (x1 + x2) / 2
        verts = [(x1, y1), (xm, y1), (xm, y2), (x2, y2), (x2, y2 + th), (xm, y2 + th), (xm, y1 + th),
                 (x1, y1 + th), (x1, y1)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4,
                 MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), color=colors[a], alpha=0.35, lw=0, zorder=2))
    for t, layer in enumerate(layers):
        ax.text(t, 1.04, layer[0]["period"] if layer else "", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.4, nper - 0.6)
    ax.set_ylim(-0.02, 1.08)
    ax.axis("off")
    return F._save(fig, path)
