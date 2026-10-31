"""Validación de los temas por expertos: hoja para evaluadores, lectura y acuerdo inter-evaluador.

Diseño (ciego respecto al modelo):
  * Calificación 1-5 de interpretabilidad y coherencia de cada tema (Mimno et al., 2011).
  * Prueba de la palabra intrusa (Chang et al., 2009): 5 términos principales + 1 intruso.
  * Acuerdo entre evaluadores: alfa de Krippendorff (intervalo para calificaciones, nominal para intrusos).
"""
from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd

KEY_SHEET = "_key"


def _representative(theta_col, titles, n=3):
    idx = np.argsort(-theta_col)[:n]
    return [titles[i] for i in idx]


def build_sheet(session, n_docs=3, seed=42) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    rng = random.Random(seed)
    titles = session.df["TI"].tolist()
    items = []
    for model, res in (("LDA", session.lda), ("BERTopic", session.bert)):
        if res is None:
            continue
        for k in range(res.k):
            # intruso: palabra frecuente en otro tema y poco probable en este
            others = [w for j in range(res.k) if j != k for w in res.top10[j][:5]]
            own = set(res.top25[k])
            cands = [w for w in others if w not in own]
            intruder = rng.choice(cands) if cands else res.top25[(k + 1) % res.k][0]
            words = res.top10[k][:5] + [intruder]
            rng.shuffle(words)
            score = res.theta[:, k]
            items.append({"model": model, "topic": res.ids[k], "terms": ", ".join(res.top10[k]),
                          "docs": _representative(score, titles, n_docs), "words": words,
                          "intruder": intruder})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["code"] = f"T{i + 1:02d}"

    wb = Workbook()
    ws = wb.active
    ws.title = "Instructions"
    head = Font(bold=True, size=12)
    lines = [
        ("Topic validation sheet / Hoja de validación de temas", head),
        ("1. Write your name in cell B3 / Escriba su nombre en la celda B3.", None),
        ("Rater / Evaluador:", Font(bold=True)),
        ("2. Sheet 'Ratings': rate each topic from 1 (very poor) to 5 (excellent) for interpretability and "
         "coherence, and suggest a short label. / Califique cada tema de 1 a 5 y sugiera un nombre.", None),
        ("3. Sheet 'Word intrusion': choose the word that does NOT belong to the topic. / Elija la palabra "
         "que NO pertenece al tema.", None),
        ("4. Do not change topic codes or sheet names. Topics are shown in random order and the model is "
         "hidden on purpose. / No modifique los códigos; el modelo se oculta a propósito.", None),
        ("5. Save the file and send it back. / Guarde y devuelva el archivo.", None),
    ]
    for r, (txt, font) in enumerate(lines, start=1):
        c = ws.cell(row=r, column=1, value=txt)
        if font:
            c.font = font
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 40
    ws["B3"].fill = PatternFill("solid", fgColor="FFF2CC")

    fill = PatternFill("solid", fgColor="FFF2CC")
    wr = wb.create_sheet("Ratings")
    cols = ["Code", "Top terms", "Representative documents", "Interpretability (1-5)",
            "Coherence (1-5)", "Suggested label", "Comments"]
    wr.append(cols)
    for c in wr[1]:
        c.font = Font(bold=True)
    for it in items:
        wr.append([it["code"], it["terms"], "\n".join(f"• {t}" for t in it["docs"]), None, None, None, None])
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True)
    wr.add_data_validation(dv)
    dv.add(f"D2:E{len(items) + 1}")
    for i, w in enumerate([7, 45, 70, 16, 14, 30, 30], start=1):
        wr.column_dimensions[chr(64 + i)].width = w
    for row in wr.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in row[3:6]:
            c.fill = fill

    wi = wb.create_sheet("Word intrusion")
    wi.append(["Code", "W1", "W2", "W3", "W4", "W5", "W6", "Intruder word"])
    for c in wi[1]:
        c.font = Font(bold=True)
    for r, it in enumerate(items, start=2):
        wi.append([it["code"], *it["words"], None])
        dvw = DataValidation(type="list", formula1=f"=B{r}:G{r}", allow_blank=True)
        wi.add_data_validation(dvw)
        dvw.add(f"H{r}")
        wi[f"H{r}"].fill = fill
    for col, w in zip("ABCDEFGH", [7, 18, 18, 18, 18, 18, 18, 20]):
        wi.column_dimensions[col].width = w

    wk = wb.create_sheet(KEY_SHEET)
    wk.append(["code", "model", "topic", "intruder"])
    for it in items:
        wk.append([it["code"], it["model"], it["topic"], it["intruder"]])
    wk.sheet_state = "veryHidden"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def read_sheets(files):
    """files: [(nombre, bytes)] -> (calificaciones largas, intrusos largos, clave)."""
    from openpyxl import load_workbook
    ratings, intr, key = [], [], None
    for i, (name, raw) in enumerate(files):
        wb = load_workbook(io.BytesIO(raw), data_only=True)
        rater = str(wb["Instructions"]["B3"].value or f"Rater {i + 1}").strip()
        k = pd.DataFrame(list(wb[KEY_SHEET].values)[1:], columns=["code", "model", "topic", "intruder"])
        key = k if key is None else key
        for row in list(wb["Ratings"].values)[1:]:
            if row and row[0]:
                ratings.append({"rater": rater, "code": row[0], "interpretability": row[3],
                                "coherence": row[4], "label": row[5]})
        for row in list(wb["Word intrusion"].values)[1:]:
            if row and row[0]:
                intr.append({"rater": rater, "code": row[0], "choice": row[7]})
    r = pd.DataFrame(ratings)
    for c in ("interpretability", "coherence"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    return r, pd.DataFrame(intr), key


def krippendorff_alpha(data, level="interval"):
    """data: matriz evaluadores × unidades (np.nan = sin dato). Krippendorff (2004)."""
    data = np.asarray(data, dtype=object)
    units = []
    for j in range(data.shape[1]):
        vals = [v for v in data[:, j] if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(vals) >= 2:
            units.append(vals)
    if not units:
        return np.nan
    if level == "interval":
        delta = lambda a, b: (float(a) - float(b)) ** 2  # noqa: E731
    else:
        delta = lambda a, b: 0.0 if a == b else 1.0  # noqa: E731
    n = sum(len(u) for u in units)
    do = sum(sum(delta(a, b) for i, a in enumerate(u) for j, b in enumerate(u) if i != j) / (len(u) - 1)
             for u in units) / n
    allv = [v for u in units for v in u]
    de = sum(delta(a, b) for i, a in enumerate(allv) for j, b in enumerate(allv) if i != j) / (n * (n - 1))
    return float(1 - do / de) if de > 0 else np.nan


def analyse(files):
    from .stats import compare_topics, mean_ci
    r, wi, key = read_sheets(files)
    raters = sorted(r["rater"].unique())
    codes = list(key["code"])
    out = {"raters": raters, "n_raters": len(raters)}
    alphas = {}
    for c in ("interpretability", "coherence"):
        M = r.pivot_table(index="rater", columns="code", values=c, aggfunc="first").reindex(columns=codes)
        alphas[c] = krippendorff_alpha(M.values.astype(float), "interval")
    W = wi.pivot_table(index="rater", columns="code", values="choice", aggfunc="first").reindex(columns=codes)
    alphas["intrusion"] = krippendorff_alpha(W.values, "nominal")
    out["alpha"] = alphas
    wi = wi.merge(key, on="code")
    wi["correct"] = (wi["choice"].astype(str).str.strip().str.lower()
                     == wi["intruder"].astype(str).str.strip().str.lower()).astype(float)
    wi.loc[wi["choice"].isna(), "correct"] = np.nan
    per_topic = (r.merge(key, on="code").groupby(["model", "topic"])[["interpretability", "coherence"]].mean()
                 .join(wi.groupby(["model", "topic"])["correct"].mean().rename("model_precision")))
    out["per_topic"] = per_topic.reset_index()
    rows = []
    for model, g in per_topic.reset_index().groupby("model"):
        row = {"Model": model}
        for c, name, bnd in (("interpretability", "Interpretability (1-5)", (1, 5)),
                             ("coherence", "Coherence (1-5)", (1, 5)),
                             ("model_precision", "Word-intrusion precision", (0, 1))):
            m, lo, hi = mean_ci(g[c].values, bounds=bnd)
            row[name] = f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
        rows.append(row)
    table = pd.DataFrame(rows)
    tests = {}
    pt = per_topic.reset_index()
    if pt["model"].nunique() == 2:
        a, b = (pt[pt["model"] == m] for m in ("LDA", "BERTopic"))
        for c in ("interpretability", "coherence", "model_precision"):
            tests[c] = compare_topics(a[c].values, b[c].values)
    out["table"], out["tests"] = table, tests
    labels = (r.merge(key, on="code").dropna(subset=["label"]).groupby(["model", "topic"])["label"]
              .apply(lambda s: " | ".join(str(x) for x in s)).reset_index())
    out["labels"] = labels
    return out
