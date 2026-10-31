"""Figuras en inglés, 300 dpi, listas para revista."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DPI = 300
BLUE, ORANGE, GREEN, GREY, RED = "#2a6f97", "#e07a1f", "#3a8d5c", "#7a7a7a", "#b23a48"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "figure.dpi": 100, "pdf.fonttype": 42, "legend.frameon": False,
})


def _save(fig, path):
    """PNG a 300 dpi para el Word y PDF vectorial (texto editable) para la revista."""
    fig.savefig(path, dpi=DPI)
    try:
        fig.savefig(str(path).rsplit(".", 1)[0] + ".pdf")
    except Exception:
        pass
    plt.close(fig)
    return path


def wrap(s, w=28):
    return "\n".join(textwrap.wrap(str(s), w))


# ------------------------------------------------------------ PRISMA
def prisma(path, by_db, total, dups, unique, steps, final):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(2.9, 9.8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef4f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=BLUE, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=GREY))

    src = "\n".join(f"{k}: n = {v:,}" for k, v in by_db.items())
    box(0.3, 8.2, 4.4, 1.4, f"Records identified\n{src}\nTotal: n = {total:,}")
    box(5.6, 8.2, 4.1, 1.4, f"Duplicates removed\n(DOI / title + year)\nn = {dups:,}", "#fbeee6")
    arrow(4.7, 8.9, 5.6, 8.9)
    box(0.3, 5.7, 4.4, 1.3, f"Unique records screened\nn = {unique:,}")
    arrow(2.5, 8.2, 2.5, 7.0)
    excl = "\n".join(wrap(f"{name}: n = {n:,}", 34) for name, n in steps if n) or "None"
    box(5.6, 5.4, 4.1, 1.9, f"Records excluded\n{excl}", "#fbeee6")
    arrow(4.7, 6.35, 5.6, 6.35)
    box(0.3, 3.2, 4.4, 1.3, f"Records included in the\nbibliometric and topic analyses\nn = {final:,}",
        "#e8f3ec")
    arrow(2.5, 5.7, 2.5, 4.5)
    for y, lab in [(8.9, "Identification"), (6.35, "Screening"), (3.85, "Included")]:
        ax.text(-0.25, y, lab, rotation=90, ha="center", va="center", fontsize=9, color=BLUE,
                fontweight="bold")
    return _save(fig, path)


# ------------------------------------------------------- bibliometría
def annual_production(path, ap):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(ap["Year"], ap["Articles"], color=BLUE, width=0.8, label="Articles")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of documents")
    if "MeanTCperYear" in ap:
        ax2 = ax.twinx()
        ax2.plot(ap["Year"], ap["MeanTCperYear"], color=ORANGE, lw=1.5, label="Mean citations/year")
        ax2.set_ylabel("Mean citations per year", color=ORANGE)
        ax2.spines["top"].set_visible(False)
    ax.set_title("Annual scientific production")
    return _save(fig, path)


def barh(path, labels, values, title, xlabel, color=BLUE, n=15):
    labels, values = list(labels)[:n][::-1], list(values)[:n][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(labels) + 1))
    ax.barh([wrap(l, 45) for l in labels], values, color=color)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:g}", va="center", fontsize=7.5)
    return _save(fig, path)


def country_scp_mcp(path, cp):
    cp = cp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(cp) + 1))
    ax.barh(cp["Country"], cp["SCP"], color=BLUE, label="SCP (single-country)")
    ax.barh(cp["Country"], cp["MCP"], left=cp["SCP"], color=ORANGE, label="MCP (multi-country)")
    ax.set_xlabel("Documents (corresponding author's country)")
    ax.set_title("Most productive countries")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path)



def k_selection(path, t, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharex=True)

    def mm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    for col, mk in [("CaoJuan2009", "o"), ("Arun2010", "s")]:
        axes[0].plot(t["K"], mm(t[col]), marker=mk, ms=4, label=col)
    axes[0].set_title("Minimize")
    for col, mk in [("Deveaud2014", "^"), ("NPMI", "D")]:
        axes[1].plot(t["K"], mm(t[col]), marker=mk, ms=4, label=col)
    axes[1].set_title("Maximize")
    for ax in axes:
        ax.axvline(chosen, color=RED, ls="--", lw=1)
        ax.set_xlabel("Number of topics (K)")
        ax.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("Normalized metric")
    return _save(fig, path)


def topic_words(path, weights, words, labels, title, color=BLUE, ncols=4, n=8):
    k = len(labels)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 1.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        ax = axes[i]
        if i >= k:
            ax.axis("off")
            continue
        w = words[i][:n][::-1]
        v = weights[i][:n][::-1]
        ax.barh(w, v, color=color)
        ax.set_title(wrap(labels[i], 30), fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=6)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def topic_heatmap_time(path, m, labels, title, cmap="Blues"):
    """m: DataFrame periodos x temas."""
    fig, ax = plt.subplots(figsize=(0.55 * m.shape[0] + 3.5, 0.3 * m.shape[1] + 1.2))
    data = m.T.values
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(m.shape[0]), m.index, rotation=45, ha="right")
    ax.set_yticks(range(m.shape[1]), [wrap(l, 40) for l in labels], fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean topic prevalence", shrink=0.7)
    ax.set_title(title)
    return _save(fig, path)



def doc_map(path, xy, topics, labels, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab20")
    out = topics < 0
    ax.scatter(xy[out, 0], xy[out, 1], s=2, c="#d0d0d0", label="Outliers")
    for t in range(len(labels)):
        m = topics == t
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=3, color=cmap(t % 20), alpha=0.7)
            cx, cy = np.median(xy[m], axis=0)
            ax.text(cx, cy, labels[t].split(":")[0], fontsize=7.5, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    return _save(fig, path)


def similarity_heatmap(path, S, matches, title):
    fig, ax = plt.subplots(figsize=(0.35 * S.shape[1] + 3, 0.32 * S.shape[0] + 1.5))
    im = ax.imshow(S.values, cmap="viridis", vmin=0, vmax=max(S.values.max(), 0.5), aspect="auto")
    ax.set_xticks(range(S.shape[1]), S.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(S.shape[0]), S.index, fontsize=7)
    for _, r in matches.iterrows():
        if r["_i"] >= 0 and r["_j"] >= 0:
            ax.add_patch(plt.Rectangle((r["_j"] - .5, r["_i"] - .5), 1, 1, fill=False, ec="white",
                                       lw=1.4))
    ax.set_xlabel("BERTopic topics")
    ax.set_ylabel("LDA topics")
    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.7)
    ax.set_title(title)
    return _save(fig, path)


def contingency(path, cont, title):
    P = cont.div(cont.sum(axis=1) + 1e-12, axis=0)
    fig, ax = plt.subplots(figsize=(0.35 * P.shape[1] + 3, 0.32 * P.shape[0] + 1.5))
    im = ax.imshow(P.values, cmap="magma_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(P.shape[1]), P.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(P.shape[0]), P.index, fontsize=7)
    ax.set_xlabel("BERTopic topic (document assignment)")
    ax.set_ylabel("LDA dominant topic")
    fig.colorbar(im, ax=ax, label="Row proportion", shrink=0.7)
    ax.set_title(title)
    return _save(fig, path)


def trends(path, tl, tb, lda_labels, bert_labels):
    panels = [(t, labs, name) for t, labs, name in
              [(tl, lda_labels, "LDA"), (tb, bert_labels, "BERTopic")] if t is not None]
    n = max(len(p[0]) for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 0.28 * n + 1.6))
    axes = np.atleast_1d(axes)
    for ax, (t, labs, name) in zip(axes, panels):
        order = np.argsort(t["Slope"].values)
        cols = [RED if t["Trend"][i] == "Hot" else BLUE if t["Trend"][i] == "Cold" else "#bbbbbb"
                for i in order]
        ax.barh([wrap(labs[i], 34) for i in order], t["Slope"].values[order] * 100, color=cols)
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_title(f"{name}: hot (red) and cold (blue) topics")
        ax.set_xlabel("Slope of annual prevalence (×100)")
        ax.tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    return _save(fig, path)


def strategic(path, sm, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x, y = sm["Growth (slope)"] * 100, sm["Impact (citations/year)"]
    ax.scatter(x, y, s=60 + 3000 * sm["Prevalence"], color=BLUE, alpha=0.55, ec="white")
    ax.axvline(0, color=GREY, lw=0.7, ls="--")
    ax.axhline(np.median(y), color=GREY, lw=0.7, ls="--")
    for xi, yi, l in zip(x, y, sm["Topic"]):
        ax.text(xi, yi, " " + l.split(":")[0], fontsize=7.5)
    ax.set_xlabel("Growth: slope of annual prevalence (×100)")
    ax.set_ylabel("Impact: mean citations per year")
    ax.set_title(title)
    return _save(fig, path)


# ------------------------------------------------------ utilidades de estilo
# Paleta Okabe-Ito (apta para daltónicos) ampliada
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000",
           "#8C6BB1", "#66A61E", "#A6761D", "#E7298A", "#1B9E77", "#7570B3", "#999999"]


def _halo(text_obj, w=2.5):
    text_obj.set_path_effects([pe.withStroke(linewidth=w, foreground="white")])
    return text_obj


def _repel(ax, texts):
    """Evita que las etiquetas se solapen (adjustText si está disponible)."""
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.15, 1.3), arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4))
    except Exception:
        pass


def _curved_edge(ax, p1, p2, color, lw, alpha, rad=0.18, z=1):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                 color=color, lw=lw, alpha=alpha, zorder=z, shrinkA=0, shrinkB=0))


# ----------------------------------------------------- redes (estilo bibliometrix)
def _separate(pos, nodes, sizes, fig_pts=500, iters=300):
    """Aleja los nodos que se solapan (radio aproximado en unidades de datos)."""
    xy = np.array([pos[n_] for n_ in nodes], dtype=float)
    xy -= xy.mean(axis=0)
    xy /= np.abs(xy).max() + 1e-12
    r = np.sqrt(sizes) / 2 / (fig_pts / 2.3) * 1.15
    for _ in range(iters):
        moved = False
        d = xy[:, None, :] - xy[None, :, :]
        dist = np.linalg.norm(d, axis=2) + 1e-9
        need = r[:, None] + r[None, :]
        over = (dist < need) & ~np.eye(len(xy), dtype=bool)
        if not over.any():
            break
        push = ((need - dist) / dist * over)[:, :, None] * d * 0.5
        xy += push.sum(axis=1)
        moved = True
        if not moved:
            break
    return {n_: xy[i] for i, n_ in enumerate(nodes)}


def network(path, nodes, edges, title, top_nodes=50, normalize="association", min_weight=1,
            max_edges=250, seed=7, size_label="Occurrences"):
    """nodes: [(nombre, frecuencia)], edges: {(a, b): peso}.
    Fuerza de asociación (van Eck y Waltman, 2009) para el diseño; Louvain para las comunidades."""
    freq = dict(sorted(nodes, key=lambda x: -x[1])[:top_nodes])
    G = nx.Graph()
    for n_, f_ in freq.items():
        G.add_node(n_, freq=f_)
    for (a, b), w in edges.items():
        if a in freq and b in freq and w >= min_weight:
            s = w / (freq[a] * freq[b]) if normalize == "association" else w
            G.add_edge(a, b, weight=w, strength=s)
    if G.number_of_edges() > max_edges:
        keep = sorted(G.edges(data=True), key=lambda e: -e[2]["weight"])[:max_edges]
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        H.add_edges_from(keep)
        G = H
    G.remove_nodes_from([n_ for n_ in list(G) if G.degree(n_) == 0])
    if len(G) < 3:
        return None
    try:
        comms = nx.algorithms.community.louvain_communities(G, weight="strength", seed=seed, resolution=1.0)
    except Exception:
        comms = [set(G)]
    comms = sorted(comms, key=lambda c: -sum(freq[n_] for n_ in c))
    cid = {n_: i for i, c in enumerate(comms) for n_ in c}
    col = {n_: PALETTE[cid[n_] % len(PALETTE)] for n_ in G}
    # Fruchterman-Reingold con pesos normalizados
    smax = max(d["strength"] for *_, d in G.edges(data=True))
    for u, v, d in G.edges(data=True):
        d["w_layout"] = 0.1 + d["strength"] / smax
    pos = nx.spring_layout(G, k=2.2 / np.sqrt(len(G)), iterations=300, seed=seed, weight="w_layout")
    f = np.array([freq[n_] for n_ in G])
    sizes = 60 + 1400 * (f / f.max()) ** 0.8
    pos = _separate(pos, list(G), sizes)
    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    wmax = max(d["weight"] for *_, d in G.edges(data=True))
    for u, v, d in G.edges(data=True):
        same = cid[u] == cid[v]
        _curved_edge(ax, pos[u], pos[v], col[u] if same else "#9a9a9a",
                     lw=0.3 + 3.2 * (d["weight"] / wmax) ** 0.8, alpha=0.45 if same else 0.18)
    xy = np.array([pos[n_] for n_ in G])
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=[col[n_] for n_ in G], alpha=0.85, edgecolors="white",
               linewidths=0.8, zorder=3)
    for (x, y), n_, s_ in zip(xy, G, sizes):
        fs = 6 + 5 * (s_ - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
        t = ax.text(x, y, n_, fontsize=fs, ha="center", va="center", zorder=4, color="#111111",
                    fontweight="bold" if s_ > np.percentile(sizes, 80) else "normal")
        t.set_path_effects([pe.withStroke(linewidth=1.2, foreground=(1, 1, 1, 0.7))])
    handles = [Line2D([], [], marker="o", ls="", color=PALETTE[i % len(PALETTE)], markersize=7,
                      label=f"Cluster {i + 1}: " + ", ".join(sorted(c, key=lambda n_: -freq[n_])[:2]))
               for i, c in enumerate(comms[:8])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False,
              fontsize=7.5)
    ax.set_title(title)
    ax.axis("off")
    ax.margins(0.08)
    _save(fig, path)
    return [sorted(c, key=lambda n_: -freq[n_]) for c in comms]


# --------------------------------------------- mapa mundial de colaboración
_WORLD = None
ISO3 = {"USA": "USA", "United Kingdom": "GBR", "South Korea": "KOR", "Russia": "RUS", "Vietnam": "VNM",
        "Iran": "IRN", "Czech Republic": "CZE", "Cote d'Ivoire": "CIV", "Taiwan": "TWN", "DR Congo": "COD",
        "Congo": "COG", "Laos": "LAO", "Bolivia": "BOL", "Venezuela": "VEN", "Tanzania": "TZA",
        "Syria": "SYR", "Moldova": "MDA", "China": "CHN", "Netherlands": "NLD"}


def _world():
    global _WORLD
    if _WORLD is None:
        import json
        from pathlib import Path
        _WORLD = json.loads((Path(__file__).parent / "data" / "world_110m.json").read_text())
    return _WORLD


def _iso3(name):
    if name in ISO3:
        return ISO3[name]
    try:
        import pycountry
        return pycountry.countries.lookup(name).alpha_3
    except Exception:
        for c in _world():
            if c["name"].lower() == name.lower():
                return c["iso3"]
    return None


def collaboration_map(path, production, edges, title, max_edges=60):
    """production: {país: documentos}; edges: [(a, b, peso)]."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm
    world = _world()
    prod = {_iso3(k): v for k, v in production.items() if _iso3(k)}
    cent = {c["iso3"]: (c["x"], c["y"]) for c in world}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cmap = plt.get_cmap("Blues")
    vmax = max(prod.values()) if prod else 1
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))
    polys, colors = [], []
    for c in world:
        v = prod.get(c["iso3"], 0)
        fc = cmap(0.15 + 0.85 * norm(v)) if v else "#eeeeee"
        for poly in c["polys"]:
            polys.append(np.array(poly[0]))
            colors.append(fc)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors="white", linewidths=0.3))
    top = sorted(edges, key=lambda e: -e[2])[:max_edges]
    wmax = max((e[2] for e in top), default=1)
    for a, b, w in top[::-1]:
        ia, ib = _iso3(a), _iso3(b)
        if ia in cent and ib in cent:
            _curved_edge(ax, cent[ia], cent[ib], "#D55E00", lw=0.4 + 2.6 * (w / wmax),
                         alpha=0.25 + 0.5 * (w / wmax), rad=0.2, z=3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("Documents (log scale)")
    ax.add_line(Line2D([], [], color="#D55E00", lw=2, label=f"Top {len(top)} collaboration links"))
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    ax.set_xlim(-170, 190)
    ax.set_ylim(-58, 85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    return _save(fig, path)


# --------------------------------------------------- HJ-Biplot con conglomerados
def biplot(path, hj, title, row_kind="period", col_clusters=None, row_clusters=None, row_color="#0072B2"):
    """Vectores = temas (coloreados por conglomerado, con envolvente); puntos = periodos o países.
    En periodos se dibuja la trayectoria cronológica."""
    from scipy.spatial import ConvexHull
    R = hj.rows.iloc[:, :2].values
    C = hj.cols.iloc[:, :2].values
    sc = np.linalg.norm(R, axis=1).max() / (np.linalg.norm(C, axis=1).max() + 1e-12)
    Cs = C * sc
    lim = 1.22 * max(np.abs(R).max(), np.abs(Cs).max())
    fig, ax = plt.subplots(figsize=(7, 6.4))
    ax.axhline(0, color="#d0d0d0", lw=0.6, zorder=0)
    ax.axvline(0, color="#d0d0d0", lw=0.6, zorder=0)
    cc = col_clusters if col_clusters is not None else np.zeros(len(C), int)
    texts = []
    for k in sorted(set(cc)):
        idx = np.where(cc == k)[0]
        color = PALETTE[k % len(PALETTE)]
        pts = np.vstack([Cs[idx], [[0, 0]]])
        if len(idx) >= 2:
            try:
                hull = ConvexHull(pts)
                ax.fill(*pts[hull.vertices].T, color=color, alpha=0.10, zorder=1, lw=0)
            except Exception:
                pass
        for i in idx:
            ax.annotate("", xy=Cs[i], xytext=(0, 0), zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=8))
            texts.append(_halo(ax.text(*(Cs[i] * 1.07), hj.cols.index[i], color=color, fontsize=7.5,
                                       ha="center", va="center", fontweight="bold", zorder=5)))
    if row_kind == "period":
        ax.plot(R[:, 0], R[:, 1], color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=3)
        for i in range(len(R) - 1):
            ax.annotate("", xy=R[i + 1], xytext=R[i], zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0, mutation_scale=9))
    rc = row_clusters if row_clusters is not None else None
    for i, (x, y) in enumerate(R):
        c = PALETTE[(rc[i] + 4) % len(PALETTE)] if rc is not None else row_color
        ax.scatter(x, y, s=46, color=c, edgecolors="white", linewidths=0.8, zorder=4,
                   marker="s" if row_kind == "period" else "o")
        texts.append(_halo(ax.text(x, y, " " + str(hj.rows.index[i]), fontsize=7.5, color="#1a1a1a",
                                   zorder=5)))
    _repel(ax, texts)
    if col_clusters is not None:
        handles = [Line2D([], [], color=PALETTE[k % len(PALETTE)], lw=2.5, label=f"Topic cluster {k + 1}")
                   for k in sorted(set(cc))]
        if rc is not None:
            handles += [Line2D([], [], marker="o", ls="", color=PALETTE[(k + 4) % len(PALETTE)], markersize=7,
                               label=f"Country cluster {k + 1}") for k in sorted(set(rc))]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Axis 1 ({100 * hj.inertia[0]:.1f}%)")
    ax.set_ylabel(f"Axis 2 ({100 * hj.inertia[1]:.1f}%)")
    ax.set_title(title)
    return _save(fig, path)
