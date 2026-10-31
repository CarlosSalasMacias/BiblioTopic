"""Figuras en inglés, 300 dpi, listas para revista."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DPI = 300
BLUE, ORANGE, GREEN, GREY, RED = "#2a6f97", "#e07a1f", "#3a8d5c", "#7a7a7a", "#b23a48"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "figure.dpi": 100,
})


def _save(fig, path):
    fig.savefig(path, dpi=DPI)
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


def network(path, nodes, edges, title, top_edges=150, seed=7):
    """nodes: [(name, size)], edges: {(a,b): w}."""
    G = nx.Graph()
    for n, s in nodes:
        G.add_node(n, size=s)
    for (a, b), w in sorted(edges.items(), key=lambda x: -x[1])[:top_edges]:
        if a in G and b in G:
            G.add_edge(a, b, weight=w)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    if len(G) == 0:
        return None
    try:
        comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    except Exception:
        comms = [set(G)]
    cmap = plt.get_cmap("tab10")
    col = {n: cmap(i % 10) for i, c in enumerate(comms) for n in c}
    pos = nx.spring_layout(G, k=1.2 / np.sqrt(len(G)), seed=seed, weight="weight")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sizes = np.array([G.nodes[n]["size"] for n in G])
    ns = 80 + 1500 * sizes / sizes.max()
    ws = np.array([G[u][v]["weight"] for u, v in G.edges])
    nx.draw_networkx_edges(G, pos, width=0.3 + 3 * ws / ws.max(), alpha=0.35, edge_color=GREY, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=ns, node_color=[col[n] for n in G], alpha=0.85, ax=ax,
                           linewidths=0.5, edgecolors="white")
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    _save(fig, path)
    return comms


# ------------------------------------------------------------- temas
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


def biplot(path, hj, title, row_name="Period", col_color=RED, row_color=BLUE, label_rows=True):
    R = hj.rows.iloc[:, :2].values
    C = hj.cols.iloc[:, :2].values
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(0, color="#cccccc", lw=0.6)
    ax.axvline(0, color="#cccccc", lw=0.6)
    rmax = np.linalg.norm(R, axis=1).max()
    sc = rmax / (np.linalg.norm(C, axis=1).max() + 1e-12)
    Cs = C * sc
    lim = 1.18 * max(np.abs(R).max(), np.abs(Cs).max())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    for (x, y), lab in zip(Cs, hj.cols.index):
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col_color, lw=0.9, alpha=0.8))
        ax.text(x * 1.06, y * 1.06, wrap(lab, 22), color=col_color, fontsize=6.5, ha="center")
    ax.scatter(R[:, 0], R[:, 1], color=row_color, s=28, zorder=3)
    if label_rows:
        for (x, y), lab in zip(R, hj.rows.index):
            ax.text(x, y, f" {lab}", color=row_color, fontsize=7.5, fontweight="bold")
    ax.set_xlabel(f"Axis 1 ({100 * hj.inertia[0]:.1f}%)")
    ax.set_ylabel(f"Axis 2 ({100 * hj.inertia[1]:.1f}%)")
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
