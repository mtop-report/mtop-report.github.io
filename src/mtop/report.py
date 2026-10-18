"""Manuscript/poster outputs (local, in reports/, not published): validation tables for both
temporal splits and every learner, the model choice, nested cross-validation, Table 1 and figures.
Everything is read from the training outputs; nothing is refitted here."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import learners as Lr  # noqa: E402
from .cohort import flow  # noqa: E402
from .config import features, settings, stage_ids, stage_spec, targets  # noqa: E402
from .explorer import _table1  # noqa: E402
from .paths import PRIVATE, PUBLIC, REPORTS  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
INK, MUTED, GRID = "#111820", "#66727c", "#dce2e6"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3cbd1")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _label(stage):
    return stage_spec(stage)["title"]["en"]


def _feat_label(key):
    f = next((f for f in features() if f.key == key), None)
    return f.spec["label"]["en"] if f else key


def _ci(c, d=3):
    return f"{c[0]:.{d}f}–{c[1]:.{d}f}" if c else ""


def _model_table(tv: dict, chosen: str) -> list[str]:
    L = ["| Model | AUROC (95% CI) | ΔAUROC vs reference (95% CI) | Brier | Cal. intercept (95% CI) | Cal. slope (95% CI) | O/E |",
         "|---|---|---|---|---|---|---|"]
    for k, v in tv["models"].items():
        dl = v["ci"].get("delta_auroc_vs_ref")
        name = f"**{k}** (chosen)" if k == chosen else k
        delta = "reference" if k == Lr.REFERENCE else (f"{dl['mean']:+.3f} ({_ci(dl['ci'])})" if dl else "–")
        L.append(f"| {name} | {v['auroc']:.3f} ({_ci(v['ci']['auroc'])}) | {delta} | "
                 f"{v['brier']:.3f} | {v['citl']:+.2f} ({_ci(v['ci'].get('citl'), 2)}) | {v['slope']:.2f} ({_ci(v['ci'].get('slope'), 2)}) | {v['oe']:.2f} |")
    return L


def build_report(clean: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    figdir = REPORTS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cfg = settings()
    min_cell = int(cfg["privacy"]["min_cell"])
    results = {}
    for label in ("primary", "sensitivity"):
        f = PRIVATE / f"train_{label}.json"
        if f.exists():
            results[label] = json.loads(f.read_text())
    dep_f = PRIVATE / "deployment.json"
    dep = json.loads(dep_f.read_text()) if dep_f.exists() else {"selection": {}}
    sel = dep.get("selection", {})
    nested_f = PRIVATE / "nested_cv.json"
    nested = json.loads(nested_f.read_text()) if nested_f.exists() else None
    base, steps = flow(clean)
    md = ["# MTOP validation report", "", f"Cohort: {len(base)} patients (" + ", ".join(f"{s['step']} {s['n']}" for s in steps) + ").", ""]
    for label, r in results.items():
        md += [f"## {label.capitalize()} split: validation from {r['test_from_year']}", ""]
        for st in stage_ids():
            for tg in targets():
                tv = r["stages"][st]["targets"][tg]
                chosen = ((sel.get(st) or {}).get(tg) or {}).get("chosen", Lr.REFERENCE)
                md += [f"### {_label(st)} · {tg}", "",
                       f"Development n = {tv['n_dev']} (events {tv['events_dev']}), validation n = {tv['n_test']} (events {tv['events_test']}).", ""]
                md += _model_table(tv, chosen) + [""]
    if nested:
        md += ["## Nested cross-validation (all years)", "",
               f"{nested['folds']} outer folds × {nested['repeats']} repeats, {nested['trials']} inner trials.", ""]
        for st, per in nested["stages"].items():
            for tg, rr in per.items():
                md += [f"### {_label(st)} · {tg} (n = {rr['n']}, events {rr['events']})", "",
                       "| Model | AUROC mean ± SD | ΔAUROC vs reference | Folds better | Mean |CITL| |", "|---|---|---|---|---|"]
                for k, m in rr["models"].items():
                    md.append(f"| {k} | {m['auroc']['mean']:.3f} ± {m['auroc']['sd']:.3f} | {m['delta_mean']:+.3f} | "
                              f"{m['folds_better'] if m['folds_better'] is not None else '–'} | {m['abs_citl_mean']:.3f} |")
                md.append("")
    if sel:
        md += ["## Model choice (pre-specified rule)", "", "| Stage | Outcome | Chosen | Candidates qualifying |", "|---|---|---|---|"]
        for st in stage_ids():
            for tg in targets():
                ss = sel[st][tg]
                q = [k for k, v in ss["candidates"].items() if v["qualifies"]]
                md.append(f"| {_label(st)} | {tg} | {ss['chosen']} | {', '.join(q) or 'none'} |")
        md.append("")
    t1 = _table1(base, min_cell)
    md += ["## Table 1", "", "| Characteristic | All | Development | Validation | SMD | mRS 0–2 | mRS 3–6 | SMD (outcome) |",
           "|---|---|---|---|---|---|---|---|"]
    for row in t1["rows"]:
        if "section" in row:
            md.append(f"| **{row['section']['en']}** | | | | | | | |")
            continue

        def cell(g):
            v = row[g]
            if row["kind"] == "median":
                return "–" if v["median"] is None else f"{v['median']:g} ({v['iqr'][0]:g}–{v['iqr'][1]:g})"
            return f"<{min_cell}" if v.get("small") else ("–" if v.get("pct") is None else f"{v['pct']:.1f}%")
        smd = lambda k: "–" if row.get(k) is None else f"{row[k]:+.2f}"  # noqa: E731
        md.append(f"| {row['label']['en']} | {cell('all')} | {cell('dev')} | {cell('test')} | {smd('smd')} | {cell('good')} | {cell('poor')} | {smd('smd_outcome')} |")
    (REPORTS / "validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # figure: AUROC across stages for every learner (primary split)
    if "primary" in results:
        r = results["primary"]
        stg = stage_ids()
        for tg in targets():
            fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
            _style(ax)
            for i, m in enumerate(Lr.ORDER):
                if m not in r["stages"][stg[0]]["targets"][tg]["models"]:
                    continue
                ys = [r["stages"][s]["targets"][tg]["models"][m]["auroc"] for s in stg]
                lo = [r["stages"][s]["targets"][tg]["models"][m]["ci"]["auroc"][0] for s in stg]
                hi = [r["stages"][s]["targets"][tg]["models"][m]["ci"]["auroc"][1] for s in stg]
                x = np.arange(len(stg)) + (i - 2) * 0.06
                ax.errorbar(x, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt="o-", color=COLORS[i % 6], lw=1.6, ms=4,
                            capsize=2, label=m)
            ax.set_xticks(range(len(stg)), [_label(s) for s in stg], fontsize=8)
            ax.set_ylabel("AUROC (95% CI)")
            ax.set_ylim(0.5, 1.0)
            ax.set_title(f"Temporal validation from {r['test_from_year']} · {tg}", fontsize=10, loc="left")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
            fig.tight_layout()
            fig.savefig(figdir / f"fig_auroc_stages_{tg}.png")
            plt.close(fig)
    adaptive_figures(figdir)
    print(f"report: {REPORTS}")


def adaptive_figures(figdir) -> None:
    """Drift monitor and update-strategy comparison (admission stage, mRS 0-2)."""
    f = PUBLIC / "adaptive.json"
    if not f.exists():
        return
    a = json.loads(f.read_text(encoding="utf-8"))
    r = a["results"]["admission"]["good_outcome"]
    if a.get("learners"):        # one result per learner: show the reference model
        r = r.get(Lr.REFERENCE) or r[a["learners"][0]]
    labels = {s["key"]: s["label"]["en"] for s in a["strategies"]}
    mon = r["monitor"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    _style(ax)
    yrs = [m["year"] for m in mon]
    ax.fill_between(yrs, [m["lo95"] * 100 for m in mon], [m["hi95"] * 100 for m in mon], color=GRID, alpha=0.8, lw=0,
                    label="Expected, 95% control limits")
    ax.plot(yrs, [m["exp"] * 100 for m in mon], color=MUTED, lw=1.5)
    ax.plot(yrs, [m["obs"] * 100 for m in mon], "o-", color=BLUE, lw=2, ms=5, label="Observed")
    for m in mon:
        if m["flag"] != "ok":
            ax.annotate("!!" if m["flag"] == "alarm" else "!", (m["year"], m["obs"] * 100), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#b8312f" if m["flag"] == "alarm" else "#6b4a00")
    ax.set_ylim(min(m["obs"] for m in mon) * 100 - 7, max(m["hi95"] for m in mon) * 100 + 2)
    ax.text(0.99, 0.02, "! outside 95% limits   !! outside 3σ", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_ylabel("mRS 0–2 at 90 days (%)")
    ax.set_title(f"Drift monitor: model frozen in {a['settings']['base_end']} (admission stage)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figdir / "fig_drift_monitor.png")
    plt.close(fig)

    order = ["refit_recal", "frozen", "refit", "recal_int", "recal_log", "sliding"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)
    yy = [y["year"] for y in r["years"]]
    for i, k in enumerate(order):
        lw = 2.4 if k == "refit_recal" else 1.4
        axes[0].plot(yy, [y["by"][k]["citl"] for y in r["years"]], "o-", color=COLORS[i], lw=lw, ms=3.5, label=labels[k])
    _style(axes[0])
    axes[0].axhline(0, color="#c3cbd1", lw=1)
    axes[0].set_title("Calibration-in-the-large by test year", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    p = r["pooled"]
    ax = axes[1]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3cbd1")
    ax.spines["bottom"].set_color("#c3cbd1")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ys = np.arange(len(order))[::-1]
    vals = [p[k]["ici"] for k in order]
    ax.barh(ys, vals, color=COLORS[:len(order)], height=0.55)
    ax.set_yticks(ys, [labels[k] for k in order], fontsize=7.5, color=INK)
    for yv, k, v in zip(ys, order, vals):
        ax.text(v + max(vals) * 0.02, yv, f"{v:.3f} · AUROC {p[k]['auroc']:.3f}", va="center", fontsize=7, color=INK)
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel("Integrated calibration index (lower is better)")
    ax.set_title(f"Pooled {a['settings']['first_test']}–{a['settings']['last_test']}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(figdir / "fig_update_strategies.png")
    plt.close(fig)
