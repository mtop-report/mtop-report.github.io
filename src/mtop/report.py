"""Manuscript/poster outputs (local, in reports/): validation tables, Table 1, figures,
subgroup performance. Also writes the aggregate subgroup table for the public site."""
from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .cohort import flow, stage_cohort  # noqa: E402
from .config import features, settings, stage_ids, stage_spec, targets  # noqa: E402
from .explorer import _table1  # noqa: E402
from .features import column_to_feature, design_matrix  # noqa: E402
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


def subgroup_performance(clean: pd.DataFrame, min_n=30, min_events=10) -> list[dict]:
    pred_file = PRIVATE / "temporal_predictions.parquet"
    if not pred_file.exists():
        return []
    preds = pd.read_parquet(pred_file)
    preds = preds[preds["model"] == "LightGBM"].merge(
        clean[["patient_id", "sex_male", "age", "occlusion", "ivt", "transfer"]], on="patient_id", how="left")
    groups = {
        "sex": {"Women": lambda d: d["sex_male"] == 0, "Men": lambda d: d["sex_male"] == 1},
        "age": {"<70": lambda d: d["age"] < 70, "≥70": lambda d: d["age"] >= 70, "≥80": lambda d: d["age"] >= 80},
        "circulation": {"Anterior": lambda d: ~d["occlusion"].isin(["basilar", "other_posterior"]) & d["occlusion"].notna(),
                        "Posterior": lambda d: d["occlusion"].isin(["basilar", "other_posterior"])},
        "ivt": {"IVT": lambda d: d["ivt"] == 1, "No IVT": lambda d: d["ivt"] == 0},
        "route": {"Direct": lambda d: d["transfer"] == 0, "Drip-and-ship": lambda d: d["transfer"] == 1},
    }
    out = []
    for (stage, target), d in preds.groupby(["stage", "target"]):
        for dim, gs in groups.items():
            for name, fn in gs.items():
                g = d[fn(d).fillna(False).astype(bool)]
                ev = int(g["y"].sum())
                if len(g) < min_n or ev < min_events or len(g) - ev < min_events:
                    out.append({"stage": stage, "target": target, "dim": dim, "group": name, "suppressed": True})
                    continue
                out.append({"stage": stage, "target": target, "dim": dim, "group": name, "n": int(len(g)),
                            "events": ev, "auroc": float(roc_auc_score(g["y"], g["p"])),
                            "observed": float(g["y"].mean()), "predicted": float(g["p"].mean())})
    return out


def build_report(clean: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    figdir = REPORTS / "figures"
    figdir.mkdir(exist_ok=True)
    with open(PRIVATE / "train_results.json") as fh:
        res = json.load(fh)
    cfg = settings()
    base, steps = flow(clean)

    # ---------------- Table 1
    t1 = _table1(base, int(cfg["privacy"]["min_cell"]))
    rows = []
    for r in t1["rows"]:
        row = {"Variable": r["label"]["en"]}
        for k in ("all", "dev", "test", "good", "poor"):
            v = r[k]
            row[k] = (f"{v['median']:g} ({v['iqr'][0]:g}–{v['iqr'][1]:g})" if r["kind"] == "median" and v["median"] is not None
                      else (f"{v['pct']:.1f}%" if r["kind"] == "pct" and v["pct"] is not None else "–"))
        rows.append(row)
    tab1 = pd.DataFrame(rows).rename(columns={"all": f"All (n={t1['n']['all']})", "dev": f"Development ≤2020 (n={t1['n']['dev']})",
                                              "test": f"Temporal validation 2021+ (n={t1['n']['test']})",
                                              "good": f"mRS 0–2 (n={t1['n']['good']})", "poor": f"mRS 3–6 (n={t1['n']['poor']})"})
    tab1.to_excel(REPORTS / "table1.xlsx", index=False)

    # ---------------- validation summary (markdown)
    md = ["# MTOP validation summary", "",
          "Temporal validation: development cohort onset ≤ 2020, validation cohort 2021–2025.", "",
          "## Cohort flow", ""]
    md += [f"- {s['step']}: {s['n']}" for s in steps]
    md += ["", "| Stage | n dev | n test |", "|---|---|---|"]
    for st in stage_ids():
        sv = res["stages"][st]
        md.append(f"| {_label(st)} | {sv['n_dev']} | {sv['n_test']} |")
    for tg, tcfg in targets().items():
        md += ["", f"## {tcfg['label']['en']}", ""]
        for st in stage_ids():
            tv = res["stages"][st]["targets"][tg]
            md += [f"### {_label(st)}", "",
                   "| Model | AUROC (95% CI) | Brier | Cal. intercept | Cal. slope | ΔAUROC vs LightGBM (95% CI) |",
                   "|---|---|---|---|---|---|"]
            for k, v in tv["temporal"].items():
                ci = v["ci"]["auroc"]
                d = v["ci"].get("delta_auroc_vs_ref")
                dtxt = f"{d['mean']:+.3f} ({d['ci'][0]:+.3f} to {d['ci'][1]:+.3f})" if d else "–"
                md.append(f"| {k} | {v['auroc']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | {v['brier']:.3f} | {v['citl']:+.2f} | {v['slope']:.2f} | {dtxt} |")
            it = tv["internal"]
            dep = tv["deployed"]
            md += ["", f"Internal (development, repeated CV): AUROC {it['auroc']:.3f}; fold mean {it['fold_auroc_mean']:.3f} ± {it['fold_auroc_sd']:.3f}.",
                   f"Deployed model: {dep['rounds']} trees; recent-era intercept shift {dep['intercept_recent']:+.2f}; "
                   f"OOF AUROC all years {dep['oof_all']['auroc']:.3f}.", ""]
            rc = tv.get("recalibration")
            if rc:
                md.append(f"Recalibration check: intercept re-estimated on {rc['recal_years']} (shift {rc['intercept_shift']:+.2f}); "
                          f"in {rc['eval_years']} calibration intercept {rc['before']['citl']:+.2f} → {rc['after']['citl']:+.2f}.")
                md.append("")
            md.append(f"Hyperparameters: `{json.dumps(tv['params'])}`, {tv['rounds_dev']} rounds.")
            md.append("")
    sub = subgroup_performance(clean)
    if sub:
        md += ["## Subgroup discrimination (LightGBM, temporal validation)", "",
               "| Outcome | Stage | Subgroup | n | events | AUROC | observed | predicted |", "|---|---|---|---|---|---|---|---|"]
        for r in sub:
            if r.get("suppressed"):
                continue
            md.append(f"| {r['target']} | {r['stage']} | {r['dim']}: {r['group']} | {r['n']} | {r['events']} | {r['auroc']:.3f} | "
                      f"{100 * r['observed']:.1f}% | {100 * r['predicted']:.1f}% |")
        with open(PUBLIC / "subgroups.json", "w", encoding="utf-8") as fh:
            json.dump(sub, fh, ensure_ascii=False, separators=(",", ":"))
    (REPORTS / "validation_summary.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- figures
    tg = "good_outcome"
    stg = stage_ids()
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    _style(ax)
    for i, m in enumerate(["LightGBM", "Logistic regression", "XGBoost", "MLP"]):
        ys = [res["stages"][s]["targets"][tg]["temporal"][m]["auroc"] for s in stg]
        lo = [res["stages"][s]["targets"][tg]["temporal"][m]["ci"]["auroc"][0] for s in stg]
        hi = [res["stages"][s]["targets"][tg]["temporal"][m]["ci"]["auroc"][1] for s in stg]
        x = np.arange(len(stg)) + (i - 1.5) * 0.07
        ax.errorbar(x, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="o-", color=COLORS[i], lw=2, ms=5,
                    capsize=0, elinewidth=1.2, label=m)
    ax.set_xticks(range(len(stg)), [_label(s) for s in stg])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (2021–2025)", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Functional independence (mRS 0–2): discrimination by stage", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(figdir / "fig_auroc_by_stage.png")
    plt.close(fig)

    for st in stg:
        tv = res["stages"][st]["targets"][tg]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), dpi=200)
        ax = axes[0]
        _style(ax)
        for i, (k, pts) in enumerate(tv["curves"]["roc"].items()):
            ax.plot([p["fpr"] for p in pts], [p["tpr"] for p in pts], color=COLORS[i % 6], lw=2.2 if k == "LightGBM" else 1.4,
                    label=f"{k} ({tv['temporal'][k]['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#c3cbd1", lw=1)
        ax.set_xlabel("1 − specificity")
        ax.set_ylabel("Sensitivity")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.set_title(f"ROC · {_label(st)}", fontsize=10, loc="left")
        ax = axes[1]
        _style(ax)
        cal = tv["curves"]["calibration"]
        mx = max(max(b["pred"], b["obs"]) for b in cal) + 0.05
        ax.plot([0, mx], [0, mx], color="#c3cbd1", lw=1)
        ax.plot([b["pred"] for b in cal], [b["obs"] for b in cal], "o-", color=BLUE, lw=2, ms=5)
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_xlabel("Predicted (development model)")
        ax.set_ylabel("Observed 2021–2025")
        t = tv["temporal"]["LightGBM"]
        ax.set_title(f"Calibration · intercept {t['citl']:+.2f}, slope {t['slope']:.2f}", fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(figdir / f"fig_roc_calibration_{st}.png")
        plt.close(fig)

    # SHAP summary (deployed model, all patients) — local figure only
    for st in stg:
        booster = lgb.Booster(model_file=str(PRIVATE / f"lgbm_{st}_{tg}.txt"))
        d = stage_cohort(base, st)
        X = design_matrix(d, st)
        contrib = booster.predict(X, pred_contrib=True)[:, :-1]
        c2f = column_to_feature(st)
        phi = pd.DataFrame(contrib, columns=X.columns).T.groupby(lambda c: c2f[c]).sum().T
        order = phi.abs().mean().sort_values(ascending=True).index[-12:]
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(order) + 1.2), dpi=200)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        rng = np.random.default_rng(0)
        for i, k in enumerate(order):
            vals = phi[k].to_numpy()
            if k in X.columns:
                fv = X[k].to_numpy()
                ok = ~np.isnan(fv)
                rank = np.full(len(fv), 0.5)
                if ok.sum() > 1:
                    rank[ok] = pd.Series(fv[ok]).rank(pct=True).to_numpy()
            else:
                rank = np.full(len(vals), 0.5)
            ax.scatter(vals, i + rng.normal(0, 0.08, len(vals)), c=rank, cmap="coolwarm", s=4, alpha=0.6, linewidths=0)
        ax.axvline(0, color="#c3cbd1", lw=1)
        ax.set_yticks(range(len(order)), [_feat_label(k) for k in order])
        ax.set_xlabel("SHAP value (log-odds of mRS 0–2)")
        ax.set_title(f"SHAP summary · {_label(st)} (colour: low → high value)", fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(figdir / f"fig_shap_{st}.png")
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
