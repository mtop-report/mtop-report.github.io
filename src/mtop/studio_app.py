"""MTOP data studio: the registry at every stage of the cleaning, with every value the cleaning
changed or flagged marked in place, and corrections that are recorded instead of overwriting the
export. Local, FNO only; start it with `python -m mtop studio` (see src/mtop/studio.py)."""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mtop import paths
from mtop import studio_data as D
from mtop.clean import load_raw
from mtop.config import stage_ids
from mtop.corrections import (add_correction, audit, check_password, corrections_file, load_audit,
                              load_corrections, norm_value, password_is_set, revert_correction)

pd.set_option("styler.render.max_elements", 10_000_000)
st.set_page_config(page_title="MTOP data studio", page_icon=":material/table_view:", layout="wide")

SRC = Path(os.environ.get("MTOP_STUDIO_INPUT", str(paths.DEFAULT_RAW_FILE)))
IDLE_LOCK_S = 30 * 60

T = {
    "title": ("MTOP data studio", "MTOP datové studio"),
    "local": ("Local tool at FNO. The data stay on this computer; nothing here is published.",
              "Místní nástroj ve FNO. Data zůstávají na tomto počítači; nic z toho se nezveřejňuje."),
    "password": ("Studio password", "Heslo studia"), "unlock": ("Unlock", "Odemknout"),
    "wrong": ("Wrong password.", "Špatné heslo."),
    "no_password": ("No studio password is set. Run: python -m mtop studio --set-password",
                    "Heslo studia není nastaveno. Spusťte: python -m mtop studio --set-password"),
    "locked_idle": ("Locked after 30 minutes without activity.", "Zamčeno po 30 minutách nečinnosti."),
    "lang": ("Language", "Jazyk"), "author": ("Your name (for the audit trail)", "Vaše jméno (pro záznam změn)"),
    "author_needed": ("Enter your name in the sidebar before changing anything.", "Před jakoukoli změnou zadejte vlevo své jméno."),
    "file": ("Registry export", "Export registru"), "lock": ("Lock", "Zamknout"),
    "tab_tables": ("Tables", "Tabulky"), "tab_review": ("To review", "Ke kontrole"), "tab_corr": ("Corrections", "Opravy"),
    "tab_help": ("How it works", "Jak to funguje"),
    "version": ("Version of the data", "Verze dat"), "rows": ("Rows", "Řádky"),
    "rows_all": ("all", "všechny"), "rows_marked": ("with any mark", "s jakoukoli značkou"),
    "search": ("Search (patient ID or any value)", "Hledat (ID pacienta nebo jakoukoli hodnotu)"),
    "columns": ("Columns", "Sloupce"), "only_marked_cols": ("Only columns with marks", "Jen sloupce se značkami"),
    "shown": ("{n} of {N} rows", "{n} z {N} řádků"), "select_hint": ("Select a row to see and correct the record.",
                                                                     "Vyberte řádek, chcete-li záznam zobrazit a opravit."),
    "patient": ("Patient", "Pacient"), "in_versions": ("This record in each version", "Tento záznam v jednotlivých verzích"),
    "issues_of": ("What the cleaning found for this record", "Co čištění u tohoto záznamu našlo"),
    "none_found": ("Nothing found for this record.", "U tohoto záznamu se nic nenašlo."),
    "recorded": ("Recorded in the export", "Zapsáno v exportu"), "rule": ("Rule", "Pravidlo"),
    "suggested": ("Suggested for the source", "Návrh pro zdroj"), "status": ("Status", "Stav"),
    "new_value": ("New value for the export (empty = clear the cell)", "Nová hodnota pro export (prázdné = vymazat)"),
    "reason": ("Reason", "Důvod"), "save": ("Save correction", "Uložit opravu"),
    "apply_sug": ("Apply suggestion", "Použít návrh"), "confirm": ("The value is correct", "Hodnota je správná"),
    "which_col": ("Registry column", "Sloupec registru"), "saved": ("Saved. The tables now include it.", "Uloženo. Tabulky ji už zahrnují."),
    "edit_any": ("Correct another value of this record", "Opravit jinou hodnotu tohoto záznamu"),
    "corr_of": ("Corrections of this record", "Opravy tohoto záznamu"), "revert": ("Revert", "Vrátit"),
    "filter_status": ("Status", "Stav"), "filter_kind": ("Kind", "Druh"), "filter_var": ("Variable", "Proměnná"),
    "open": ("open", "otevřené"), "corrected": ("corrected", "opravené"), "confirmed": ("checked, correct", "zkontrolované, správné"),
    "n_issues": ("{n} findings", "{n} nálezů"), "any": ("any", "jakékoli"), "note": ("Note", "Poznámka"),
    "corr_title": ("All corrections", "Všechny opravy"), "no_corr": ("No corrections yet.", "Zatím žádné opravy."),
    "new_short": ("New value", "Nová hodnota"), "by": ("By", "Kdo"), "when": ("When", "Kdy"), "result": ("In the current export", "V aktuálním exportu"),
    "dl_excel": ("Download the corrected export (Excel)", "Stáhnout opravený export (Excel)"),
    "prep_excel": ("Prepare the corrected export (Excel)", "Připravit opravený export (Excel)"),
    "dl_csv": ("Download the corrections (CSV)", "Stáhnout opravy (CSV)"),
    "audit": ("Audit trail", "Záznam změn"), "rerun": ("Re-run the cleaning and save", "Znovu spustit čištění a uložit"),
    "rerun_done": ("Cleaning re-run: {n} records, {k} findings. clean.parquet and data_quality.xlsx are updated. "
                   "The models and the public app change only after `python -m mtop all` and a commit.",
                   "Čištění proběhlo: {n} záznamů, {k} nálezů. clean.parquet a data_quality.xlsx jsou aktualizovány. "
                   "Modely a veřejná aplikace se změní až po `python -m mtop all` a commitu."),
    "legend": ("Marks", "Značky"), "summary": ("Corrections: {a} active · {c} not matching the export", "Opravy: {a} aktivních · {c} neodpovídá exportu"),
    "no_rows": ("No rows match.", "Žádné řádky neodpovídají."),
    "excluded": ("not analysed: {r}", "neanalyzován: {r}"), "in_cohort": ("analysed", "analyzován"),
    "not_in_stage": ("not in this stage cohort", "není v kohortě této fáze"),
}
CSTATUS = {"active": ("active", "platná"), "reverted": ("reverted", "vrácená"), "superseded": ("replaced", "nahrazená")}
RESULT = {"applied": ("applied", "použita"), "already in the source": ("already in the source", "už ve zdroji"),
          "conflict": ("does not match the export any more", "už neodpovídá exportu"),
          "patient not found": ("patient not in the export", "pacient není v exportu"),
          "unknown column": ("unknown column", "neznámý sloupec"), "confirmed": ("checked, correct", "zkontrolováno, správné")}
CSS = {"missing": "background-color:#fde2e1;color:#7a1712", "flag": "background-color:#fff1bf;color:#6b4e00",
       "repaired": "background-color:#dceafe;color:#123f78", "kept": "background-color:#ece6fb;color:#3d2a78",
       "corrected": "background-color:#d6f1df;color:#11532c", "confirmed": "background-color:#e9edf2;color:#2c3a4b"}


def lang() -> str:
    return st.session_state.get("lang") or "en"


def t(key: str, **kw) -> str:
    s = T[key][0 if lang() == "en" else 1]
    return s.format(**kw) if kw else s


def L2(pair) -> str:
    return pair[0] if lang() == "en" else pair[1]


# ----------------------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def _raw(path: str, mtime: float) -> pd.DataFrame:
    return load_raw(path)


# Built once per export and corrections file and shared, not copied: the views below never modify it.
@st.cache_resource(show_spinner="…", max_entries=2)
def _build(path: str, mtime: float, corr_mtime: float) -> dict:
    return D.build(_raw(path, mtime), load_corrections())


def _stamp() -> tuple:
    cf = corrections_file()
    return str(SRC), SRC.stat().st_mtime, cf.stat().st_mtime if cf.exists() else 0.0


def data() -> dict:
    return _build(*_stamp())


@st.cache_resource(max_entries=24)
def _view(version: str, lng: str, path: str, mtime: float, corr_mtime: float):
    dat = _build(path, mtime, corr_mtime)
    tab = D.table(dat, version)
    mk, summ = D.marks(dat, version, tab, lng)
    low = tab.map(norm_value).apply(lambda col: col.str.lower()).agg(" ".join, axis=1)
    return tab, mk, summ, low


def display(df: pd.DataFrame) -> pd.DataFrame:
    """Values as recorded, in a form the table can show (dates as text, whole numbers without .0)."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.map(norm_value)
        elif pd.api.types.is_float_dtype(s):
            v = s.dropna()
            out[c] = s.astype("Int64") if len(v) and (v == v.round()).all() and v.abs().max() < 1e15 else s
        elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = s
        else:
            out[c] = s.map(norm_value)
    return out


def labels(cols, version: str) -> list[str]:
    seen, out = {}, []
    for c in cols:
        lab = D.column_label(c, lang(), version) if c != "_issues" else L2(D.EXTRA_LABELS["_issues"])
        if lab in seen:
            lab = f"{lab} ({c})"
        seen[lab] = 1
        out.append(lab)
    return out


def author() -> str:
    return (st.session_state.get("author") or "").strip()


def need_author() -> bool:
    if not author():
        st.warning(t("author_needed"))
        return True
    return False


# ----------------------------------------------------------------------------------------- pieces
def legend() -> None:
    chips = " ".join(f'<span style="{CSS[m]};padding:2px 8px;border-radius:4px;margin-right:6px;font-size:0.85em">'
                     f"{L2(D.MARK_LABELS[m])}</span>" for m in D.MARK_ORDER)
    st.markdown(f"**{t('legend')}:** {chips}", unsafe_allow_html=True)


def record_trace(dat: dict, pid: int, issue_col: str | None) -> pd.DataFrame:
    """Where a record stands in each version, with the values the finding concerns."""
    raw_cols = D.ISSUE_RAW.get(issue_col, []) if issue_col else []
    clean_cols = D.ISSUE_CLEAN.get(issue_col, [issue_col]) if issue_col else []
    rows = []
    r = dat["raw"][pd.to_numeric(dat["raw"]["Patient ID"], errors="coerce") == pid]
    rows.append({"version": D.version_label("raw", lang()),
                 "values": " · ".join(f"{D.column_label(c, lang(), 'raw')} = {norm_value(r.iloc[0][c]) or '∅'}" for c in raw_cols)
                 if len(r) else "–"})
    c = dat["clean"][dat["clean"]["patient_id"] == pid]
    cv = [x for x in clean_cols if x in dat["clean"].columns]
    rows.append({"version": D.version_label("clean", lang()),
                 "values": " · ".join(f"{D.column_label(x, lang())} = {norm_value(c.iloc[0][x]) or '∅'}" for x in cv) if len(c) else "–"})
    inco = dat["cohort"]["patient_id"].eq(pid).any()
    why = c.iloc[0]["excluded"] if len(c) else ""
    rows.append({"version": D.version_label("cohort", lang()),
                 "values": t("in_cohort") if inco else t("excluded", r=D.excluded_text(why, lang()) or "–")})
    for s in stage_ids():
        m = dat[s][dat[s]["patient_id"] == pid]
        if not len(m):
            rows.append({"version": D.version_label(s, lang()), "values": t("not_in_stage")})
            continue
        mv = [x for x in cv if x in m.columns]
        rows.append({"version": D.version_label(s, lang()),
                     "values": " · ".join(f"{D.column_label(x, lang())} = {norm_value(m.iloc[0][x]) or '∅'}" for x in mv) or "✓"})
    return pd.DataFrame(rows)


def issue_actions(dat: dict, r: dict, key: str) -> None:
    """Correct, apply the suggestion, or confirm one finding."""
    pid = int(r["patient_id"])
    raw = dat["raw"]
    row = raw[pd.to_numeric(raw["Patient ID"], errors="coerce") == pid]
    cols = r["raw_columns"] or []
    note = _opt(r.get("note"))
    st.markdown(f"**{t('patient')} {pid}** · {L2(D.MARK_LABELS[r['kind']])} · {D.issue_text(r['action'], lang())}"
                + (f" · {D.note_text(note, lang())}" if note else ""))
    st.dataframe(record_trace(dat, pid, r["column"]).rename(columns={"version": t("version"), "values": ""}),
                 hide_index=True, width="stretch")
    if not cols or not len(row):
        return
    col = cols[0] if len(cols) == 1 else st.radio(t("which_col"), cols, horizontal=True, key=f"{key}-col",
                                                  format_func=lambda c: D.column_label(c, lang(), "raw"))
    cur = norm_value(row.iloc[0][col])
    st.caption(f"{t('recorded')}: **{cur or '∅'}**")
    sug = _opt(r.get("suggested"))
    new = st.text_input(t("new_value"), value=sug or "", key=f"{key}-new")
    reason = st.text_input(t("reason"), value=D.issue_text(r["action"], lang()), key=f"{key}-why")
    b1, b2, b3 = st.columns(3)
    if sug and b1.button(f"{t('apply_sug')}: {sug}", key=f"{key}-sug", type="primary", disabled=not author()):
        _save(pid, col, cur, sug, reason, "correction", r["action"])
    if b2.button(t("save"), key=f"{key}-save", disabled=not author()):
        _save(pid, col, cur, new, reason, "correction", r["action"])
    if b3.button(t("confirm"), key=f"{key}-ok", disabled=not author()):
        _save(pid, col, cur, cur, reason, "confirmed", r["action"])
    if not author():
        st.caption(t("author_needed"))


def _opt(v):
    """None for empty / NaN, else the text."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def _save(pid, col, old, new, reason, kind, issue="") -> None:
    try:
        add_correction(pid, col, old, new, reason, author(), kind=kind, issue=issue)
    except ValueError as e:
        st.error(str(e))
        return
    st.toast(t("saved"))
    time.sleep(0.3)
    st.rerun()


def edit_any(dat: dict, pid: int, key: str) -> None:
    raw = dat["raw"]
    row = raw[pd.to_numeric(raw["Patient ID"], errors="coerce") == pid]
    if not len(row):
        return
    cols = [c for c in raw.columns if c != "Patient ID"]
    col = st.selectbox(t("which_col"), cols, key=f"{key}-anycol", format_func=lambda c: D.column_label(c, lang(), "raw"))
    cur = norm_value(row.iloc[0][col])
    st.caption(f"{t('recorded')}: **{cur or '∅'}**")
    new = st.text_input(t("new_value"), value=cur, key=f"{key}-anynew-{col}")
    reason = st.text_input(t("reason"), key=f"{key}-anywhy-{col}")
    if st.button(t("save"), key=f"{key}-anysave", disabled=not author() or new == cur):
        _save(pid, col, cur, new, reason, "correction")


def corrections_of(pid: int, key: str) -> None:
    corr = load_corrections()
    mine = corr[(corr["patient_id"] == str(pid)) & (corr["status"] == "active")]
    if not len(mine):
        return
    st.markdown(f"**{t('corr_of')}**")
    for _, c in mine.iterrows():
        a, b = st.columns([5, 1])
        kind = t("confirmed") if c["kind"] == "confirmed" else f"{c['old_value'] or '∅'} → {c['new_value'] or '∅'}"
        a.write(f"{D.column_label(c['column'], lang(), 'raw')}: {kind} · {c['author']} · {c['created']} · {c['reason']}")
        if b.button(t("revert"), key=f"{key}-rv-{c['id']}", disabled=not author()):
            revert_correction(c["id"], author())
            st.rerun()


# ----------------------------------------------------------------------------------------- views
def tables_view(dat: dict) -> None:
    vers = D.versions()
    v = st.selectbox(t("version"), vers, format_func=lambda x: D.version_label(x, lang()), key="ver")
    tab, mk, summ, low = _view(v, lang(), *_stamp())
    present = [m for m in D.MARK_ORDER if (mk == m).any().any()]
    rowf = st.segmented_control(t("rows"), ["all", "marked"] + present, key=f"rowf-{v}", default="all", required=True,
                                format_func=lambda m: t("rows_all") if m == "all" else t("rows_marked") if m == "marked"
                                else L2(D.MARK_LABELS[m]))
    q = st.text_input(t("search"), key="q").strip().lower()
    keep = pd.Series(True, index=tab.index)
    if rowf and rowf != "all":
        want = set(D.MARK_ORDER) if rowf == "marked" else {rowf}
        keep &= mk.isin(want).any(axis=1)
    if q:
        keep &= low.str.contains(q, regex=False)
    only_marked = st.toggle(t("only_marked_cols"), key="onlycols")
    pid_col = "Patient ID" if v == "raw" else "patient_id"
    cols = list(tab.columns)
    if only_marked:
        sub = mk[keep]
        cols = [pid_col] + [c for c in cols if c != pid_col and (sub[c] != "").any()]
    shown, mks = tab.loc[keep, cols], mk.loc[keep, cols]
    shown = display(shown)
    shown["_issues"] = summ[keep]
    mks = mks.assign(_issues="")
    names = labels(shown.columns, v)
    shown.columns, mks.columns = names, names
    st.caption(t("shown", n=f"{int(keep.sum()):,}", N=f"{len(tab):,}") + " · " + t("select_hint"))
    legend()
    if not len(shown):
        st.info(t("no_rows"))
        return
    styler = shown.style.apply(lambda _: mks.map(lambda m: CSS.get(m, "")), axis=None)
    ev = st.dataframe(styler, hide_index=True, height=520, width="stretch", on_select="rerun",
                      selection_mode="single-row", key=f"tbl-{v}", placeholder="–")
    rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
    if rows:
        pid = int(pd.to_numeric(tab.loc[keep, pid_col].iloc[rows[0]], errors="coerce"))
        patient_panel(dat, pid, key=f"p-{v}")


def patient_panel(dat: dict, pid: int, key: str) -> None:
    st.divider()
    st.subheader(f"{t('patient')} {pid}")
    iss = dat["issues"][dat["issues"]["patient_id"] == pid]
    st.markdown(f"**{t('issues_of')}**")
    if not len(iss):
        st.caption(t("none_found"))
    for k, r in enumerate(iss.to_dict("records")):
        with st.expander(f"{D.column_label(r['column'], lang())}: {D.issue_text(r['action'], lang())} · {L2(STATUS[r['status']])}",
                         expanded=len(iss) == 1):
            issue_actions(dat, r, key=f"{key}-i{k}")
    with st.expander(t("edit_any")):
        edit_any(dat, pid, key)
    corrections_of(pid, key)


STATUS = {"open": T["open"], "corrected": T["corrected"], "confirmed": T["confirmed"]}


def review_view(dat: dict) -> None:
    iss = dat["issues"].copy()
    c1, c2, c3 = st.columns(3)
    stt = c1.multiselect(t("filter_status"), list(STATUS), default=["open"], key="rs", format_func=lambda s: L2(STATUS[s]),
                         placeholder=t("any"))
    knd = c2.multiselect(t("filter_kind"), [m for m in D.MARK_ORDER if m in set(iss["kind"])], default=[], key="rk",
                         format_func=lambda m: L2(D.MARK_LABELS[m]), placeholder=t("any"))
    var = c3.multiselect(t("filter_var"), sorted(iss["column"].unique()), default=[], key="rv",
                         format_func=lambda c: D.column_label(c, lang()), placeholder=t("any"))
    if stt:
        iss = iss[iss["status"].isin(stt)]
    if knd:
        iss = iss[iss["kind"].isin(knd)]
    if var:
        iss = iss[iss["column"].isin(var)]
    iss = iss.reset_index(drop=True)
    st.caption(t("n_issues", n=f"{len(iss):,}"))
    if not len(iss):
        st.info(t("no_rows"))
        return
    show = pd.DataFrame({
        t("patient"): iss["patient_id"].astype(int),
        t("filter_var"): iss["column"].map(lambda c: D.column_label(c, lang())),
        t("recorded"): iss["raw_value"].map(norm_value),
        t("rule"): iss["action"].map(lambda a: D.issue_text(a, lang())),
        t("suggested"): iss["suggested"].fillna(""),
        t("note"): iss["note"].map(lambda n: D.note_text(n, lang())),
        t("status"): iss["status"].map(lambda s: L2(STATUS[s])),
    })
    styler = show.style.apply(lambda _: pd.DataFrame(
        {c: [CSS.get(k, "") if c == t("rule") else "" for k in iss["kind"]] for c in show.columns}, index=show.index), axis=None)
    ev = st.dataframe(styler, hide_index=True, height=420, width="stretch", on_select="rerun", selection_mode="single-row",
                      key="rev", placeholder="–")
    rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
    if rows:
        r = iss.iloc[rows[0]].to_dict()
        st.divider()
        issue_actions(dat, r, key=f"rev-{rows[0]}")
        corrections_of(int(r["patient_id"]), key=f"rev-{rows[0]}")


def corrections_view(dat: dict) -> None:
    corr = load_corrections()
    rep = dat["report"]
    res = {r["id"]: r["result"] for r in rep.to_dict("records")} if len(rep) else {}
    st.subheader(t("corr_title"))
    if not len(corr):
        st.info(t("no_corr"))
    else:
        corr = corr.iloc[::-1].reset_index(drop=True)                       # newest first
        show = pd.DataFrame({
            t("patient"): corr["patient_id"], t("which_col"): corr["column"].map(lambda c: D.column_label(c, lang(), "raw")),
            t("recorded"): corr["old_value"], t("new_short"): corr["new_value"].where(corr["kind"] != "confirmed", "✓"),
            t("reason"): corr["reason"], t("by"): corr["author"], t("when"): corr["created"],
            t("status"): corr["status"].map(lambda s: L2(CSTATUS.get(s, (s, s)))),
            t("result"): corr["id"].map(lambda i: L2(RESULT.get(res.get(i, ""), ("", "")))),
        })
        ev = st.dataframe(show, hide_index=True, width="stretch", height=min(360, 40 + 35 * len(show)), on_select="rerun",
                          selection_mode="single-row", key="corr", placeholder="–")
        rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
        if rows and corr.iloc[rows[0]]["status"] == "active":
            if st.button(t("revert"), key="corr-revert", disabled=not author()):
                revert_correction(corr.iloc[rows[0]]["id"], author())
                st.rerun()
    b1, b2 = st.columns(2)
    stamp = _stamp()
    if st.session_state.get("xl_stamp") != stamp:
        st.session_state.pop("xl", None)
    if "xl" not in st.session_state:
        if b1.button(t("prep_excel"), key="prep-xl"):
            xl = io.BytesIO()                      # built on request: writing the workbook takes a moment
            with pd.ExcelWriter(xl) as xw:
                dat["raw"].to_excel(xw, sheet_name="registry", index=False)
                load_corrections().to_excel(xw, sheet_name="corrections", index=False)
                dat["issues"].drop(columns=["raw_columns"]).to_excel(xw, sheet_name="findings", index=False)
            st.session_state["xl"], st.session_state["xl_stamp"] = xl.getvalue(), stamp
            st.rerun()
    else:
        b1.download_button(t("dl_excel"), st.session_state["xl"], file_name=f"{SRC.stem}_corrected_{time.strftime('%Y-%m-%d')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
                           on_click=lambda: audit("export", author() or "?", reason="corrected export"))
    b2.download_button(t("dl_csv"), load_corrections().to_csv(index=False).encode("utf-8"), file_name="corrections.csv",
                       mime="text/csv")
    with st.expander(t("audit")):
        st.dataframe(load_audit(), hide_index=True, width="stretch")


def help_view() -> None:
    if lang() == "en":
        st.markdown("""
**Versions.** *Registry export* is the Excel file as recorded, with the active corrections applied. *Cleaned registry*
is every record after the cleaning rules. *Analysed patients* are the adults with a 90-day mRS, with their temporal split
and the stage cohorts they belong to. *Model inputs* show each stage cohort with exactly the values the models use.

**Marks.** Red: the cleaning set the value to missing (implausible or unreadable). Amber: kept but listed for review
(two fields disagree). Blue: repaired automatically (a one-year typo in the onset date). Purple: record kept in the
registry but not analysed. Green: changed by a correction. Grey: checked, the value is right.

**Correcting.** Select a row (Tables) or a finding (To review). Apply the suggestion, type the right value, or confirm
that the recorded value is correct. Each entry goes to `data/studio/corrections.csv` with your name, the time and the
reason; nothing in the Excel file is overwritten. A correction applies only while the export still holds the value it
was made against, so a later export that already contains the fix, or a different value, is reported instead of being
overwritten. Reverting keeps the entry in the file as reverted, and every action is in the audit trail.

**Afterwards.** *Re-run the cleaning and save* writes the cleaned data and the data-quality workbook. The models and
the public app change only after `python -m mtop all` and a commit; the public app never shows patient rows. Send the
corrected export (Corrections tab) to the registry team so the source is fixed too.
""")
    else:
        st.markdown("""
**Verze.** *Export registru* je soubor Excel tak, jak je zapsán, s použitými aktivními opravami. *Vyčištěný registr* jsou
všechny záznamy po pravidlech čištění. *Analyzovaní pacienti* jsou dospělí se známým mRS za 90 dní, s časovým rozdělením
a kohortami fází, do kterých patří. *Vstupy modelu* ukazují kohortu každé fáze přesně s hodnotami, které modely používají.

**Značky.** Červená: čištění nastavilo hodnotu jako chybějící (nevěrohodná nebo nečitelná). Žlutá: ponecháno, ale ke
kontrole (dvě pole si odporují). Modrá: opraveno automaticky (překlep o rok v datu vzniku). Fialová: záznam ponechán
v registru, ale neanalyzován. Zelená: změněno opravou. Šedá: zkontrolováno, hodnota je správná.

**Opravy.** Vyberte řádek (Tabulky) nebo nález (Ke kontrole). Použijte návrh, zadejte správnou hodnotu, nebo potvrďte,
že zapsaná hodnota je správná. Každý záznam jde do `data/studio/corrections.csv` s vaším jménem, časem a důvodem; v souboru
Excel se nic nepřepisuje. Oprava se použije, jen dokud export obsahuje hodnotu, proti níž byla zadána, takže pozdější
export, který opravu už obsahuje nebo má jinou hodnotu, je ohlášen a nepřepíše se. Vrácení ponechá záznam v souboru
jako vrácený a každý krok je v záznamu změn.

**Potom.** *Znovu spustit čištění a uložit* zapíše vyčištěná data a sešit kvality dat. Modely a veřejná aplikace se změní
až po `python -m mtop all` a commitu; veřejná aplikace nikdy neukazuje řádky pacientů. Opravený export (karta Opravy)
pošlete týmu registru, aby byl opraven i zdroj.
""")


# ----------------------------------------------------------------------------------------- app
def main() -> None:
    st.sidebar.segmented_control(T["lang"][0] + " / " + T["lang"][1], ["en", "cs"], key="lang", default="en", required=True,
                                 format_func=lambda x: "English" if x == "en" else "Čeština")
    now = time.time()
    if st.session_state.get("auth") and now - st.session_state.get("seen", now) > IDLE_LOCK_S:
        st.session_state.auth = False
        st.session_state.idle = True
    if not st.session_state.get("auth"):
        st.title(t("title"))
        st.caption(t("local"))
        if st.session_state.pop("idle", False):
            st.info(t("locked_idle"))
        if not password_is_set():
            st.error(t("no_password"))
            st.stop()
        with st.form("login"):
            pw = st.text_input(t("password"), type="password")
            ok = st.form_submit_button(t("unlock"), type="primary")
        if ok:
            if check_password(pw):
                st.session_state.auth = True
                st.session_state.seen = now
                st.rerun()
            time.sleep(1.0)
            st.error(t("wrong"))
        st.stop()
    st.session_state.seen = now
    st.sidebar.text_input(t("author"), key="author")
    st.sidebar.caption(f"{t('file')}: {SRC.name}")
    dat = data()
    rep = dat["report"]
    active = int((load_corrections()["status"] == "active").sum())
    bad = int(rep["result"].isin(["conflict", "patient not found", "unknown column"]).sum()) if len(rep) else 0
    st.sidebar.caption(t("summary", a=active, c=bad))
    if st.sidebar.button(t("rerun"), disabled=not author()):
        from mtop.cli import pipeline_clean

        paths.ensure_dirs()
        out = pipeline_clean(SRC)
        audit("rerun cleaning", author())
        st.sidebar.success(t("rerun_done", n=f"{len(out['clean']):,}", k=f"{len(out['issues']):,}"))
    if st.sidebar.button(t("lock")):
        st.session_state.auth = False
        st.rerun()
    st.title(t("title"))
    st.caption(t("local"))
    a, b, c, d = st.tabs([t("tab_tables"), t("tab_review"), t("tab_corr"), t("tab_help")])
    with a:
        tables_view(dat)
    with b:
        review_view(dat)
    with c:
        corrections_view(dat)
    with d:
        help_view()


main()
