import pandas as pd

from bibliotopic import io as bio


def rec(**kw):
    base = {f: "" for f in bio.FIELDS}
    base.update(DB="WoS", FILE="f", TC=None, PY=pd.NA)
    base.update(kw)
    return base


def frame(rows):
    df = pd.DataFrame(rows)
    df["PY"] = pd.array(df["PY"], dtype="Int64")
    df["TC"] = pd.to_numeric(df["TC"])
    return df


LONG = "Effects of shade trees on coffee leaf rust incidence"


def test_unknown_language_is_not_other_language():
    df = frame([rec(TI="a", AB="w " * 30, LA="English", PY=2020),
                rec(TI="b", AB="w " * 30, LA="Spanish", PY=2020),
                rec(TI="c", AB="w " * 30, LA="", PY=2020, DB="Scopus")])
    kept, steps = bio.apply_filters(df, languages=["English"])
    assert sorted(kept["TI"]) == ["a", "c"]
    assert dict(steps)["Non-eligible language"] == 1
    assert "Other languages" not in dict(steps)
    kept2, steps2 = bio.apply_filters(df, languages=["English"], keep_unknown_language=False)
    assert list(kept2["TI"]) == ["a"]
    assert dict(steps2)["Unknown language (excluded)"] == 1


def test_status_by_db_counts_three_states():
    df = frame([rec(LA="English", DB="WoS"), rec(LA="Spanish", DB="WoS"), rec(LA="", DB="Scopus"),
                rec(LA="", DB="Scopus")])
    st = bio.status_by_db(df, "LA", ["English"]).set_index("Source")
    assert st.loc["WoS"].tolist() == [1, 1, 0]
    assert st.loc["Scopus"].tolist() == [0, 0, 2]


def test_chained_duplicates_form_one_group_in_any_order():
    # 1 y 3 comparten título; 2 y 3 comparten DOI: los tres son el mismo documento
    a = rec(TI=LONG, PY=2020, DI="", DB="Scopus", UT="s1")
    b = rec(TI="Different title from another export", PY=2020, DI="10.1/x", UT="w1")
    c = rec(TI=LONG, PY=2020, DI="10.1/x", DB="Scopus", UT="s2")
    for order in ([a, b, c], [c, b, a], [b, a, c]):
        m, st = bio.merge([frame(order)])
        assert len(m) == 1 and st["duplicates_removed"] == 2


def test_title_does_not_merge_different_dois_but_merges_variants():
    a = rec(TI=LONG, PY=2020, DI="10.1/aaa")
    b = rec(TI=LONG, PY=2020, DI="10.1/bbb", DB="Scopus")
    assert len(bio.merge([frame([a, b])])[0]) == 2
    c = rec(TI=LONG, PY=2020, DI="10.1/v014n2")
    d = rec(TI=LONG, PY=2020, DI="10.1/v14n2", DB="Scopus")
    assert len(bio.merge([frame([c, d])])[0]) == 1


def test_adjacent_year_and_missing_doi():
    a = rec(TI=LONG, PY=2020, DI="")
    b = rec(TI=LONG, PY=2021, DI="", DB="Scopus")
    c = rec(TI=LONG, PY=2024, DI="")
    m, _ = bio.merge([frame([a, b, c])])
    assert len(m) == 2  # 2020-2021 se unen; 2024 es otro documento


def test_complementary_fields_and_provenance():
    w = rec(TI=LONG, PY=2020, DI="10.1/x", LA="English", AB="", TC=3, DE="rust", C1="Univ, Ecuador.", UT="w")
    s = rec(TI=LONG, PY=2021, DI="10.1/x", LA="", AB="abstract text " * 10, TC=9, DB="Scopus", UT="s")
    m, st, links = bio.merge([frame([w]), frame([s])], return_links=True)
    r = m.iloc[0]
    assert r["LA"] == "English" and r["AB"].startswith("abstract") and r["TC"] == 9
    assert r["N_RECORDS"] == 2 and "Year" in r["CONFLICTS"]
    assert r["FIELDS_FROM_OTHER"]  # algún campo procede del otro registro
    assert len(links) == 2 and links["KEPT_AS_BASE"].sum() == 1
    assert st["groups_with_conflicts"] == 1
