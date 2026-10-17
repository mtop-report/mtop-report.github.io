"""Demonstration of the federated network on FNO's own registry.

FNO's patients are split into four virtual centres by treatment era (A 2005-2014, B 2015-2017,
C 2018-2020, D 2021-2025). Each centre's data are written in the MTOP upload template and read
back by its own site node, exactly as a partner hospital would do. Inside every centre, 5-fold
cross-validation keeps a fifth of the patients aside; each centre scores its own held-out patients
(as it would at home) for three options:

  alone     a model trained on the centre's own training patients only;
  network   the shared model trained across all four nodes by the federated protocol;
  adapted   the shared model plus a few boosting steps on the centre's own patients (done locally).

Further analyses: centre size (a centre given only 50/100/200 of its patients), growth of the
network (every group of 1-4 centres), later stages, leave-one-centre-out validation (a network
of three tested at the fourth), proof of equivalence with pooled training, and a run over HTTP.

Outputs: artifacts/public/federated.json (aggregates only; every group >= min_cell patients and
events) and reports/federated.md. Row-level files stay in data/processed/federated/.
"""
from __future__ import annotations

import itertools
import json
import logging
import secrets
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .. import metrics as M
from ..cohort import flow
from ..config import settings
from ..paths import PROCESSED, PUBLIC, REPORTS, ensure_dirs
from . import PROTOCOL
from .bins import fed_settings
from .client import SiteClient
from .coordinator import (Coordinator, LocalTransport, PooledNetwork, federated_eval, federated_gbdt, federated_logreg,
                          fit_gbdt, fit_logreg)
from .gbdt import GBDTParams, leaf_sizes, raw_score, sigmoid, tree_output
from .logreg import predict_spline, spline_design, spline_spec
from .schema import read_site_csv, write_site_csv
from .server import Hub, serve, token_hash
from .site import SiteNode

log = logging.getLogger("mtop.fed")

CENTRES = [("A", 2005, 2014), ("B", 2015, 2017), ("C", 2018, 2020), ("D", 2021, 2025)]
STAGE = "admission"
OUTCOMES = ("good_outcome", "death90")
MORE_STAGES = (("procedure", "good_outcome"), ("h24", "good_outcome"), ("d7", "good_outcome"))
ADAPT_TREES = 10
SIZES = (50, 100, 200)
MODELS = ("alone", "network", "adapted")
FED_DIR = PROCESSED / "federated"


def _r(x, d=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), d)


def _cv(k: int, folds: int, seed: int) -> tuple[dict, dict]:
    tr = {"kind": "cv", "folds": folds, "fold": k, "part": "train", "seed": seed}
    return tr, dict(tr, part="test")


def make_sites(base: pd.DataFrame) -> tuple[list[SiteNode], dict]:
    FED_DIR.mkdir(parents=True, exist_ok=True)
    for f in FED_DIR.glob("audit_*.jsonl"):
        f.unlink()
    nodes, reports = [], {}
    for cid, a, z in CENTRES:
        d = base[(base["year"] >= a) & (base["year"] <= z)].copy()
        d["year"] = d["year"].astype(float)
        path = FED_DIR / f"centre_{cid}.csv"
        write_site_csv(d, path)
        typed, rep = read_site_csv(path)
        reports[cid] = rep
        nodes.append(SiteNode(cid, typed, label=f"Centre {cid}", audit_path=FED_DIR / f"audit_{cid}.jsonl"))
    return nodes, reports


def staged(model: dict, B: np.ndarray, checks: list[int]) -> dict[int, np.ndarray]:
    F = np.full(len(B), float(model["init"]))
    out = {0: F.copy()} if 0 in checks else {}
    for i, tree in enumerate(model["trees"], 1):
        F = F + tree_output(tree, B)
        if i in checks:
            out[i] = F.copy()
    return out


def auc(y, p) -> float:
    return float(M.roc_auc_score(y, p))


def _metrics(y, p, ci=None) -> dict:
    s = M.summary(y, p)
    out = {k: _r(s[k]) for k in ("auroc", "brier", "citl", "slope", "mean_pred")}
    if ci is not None:
        out["auroc_ci"] = [_r(v) for v in ci["auroc"]]
    return out


def _cat(xs) -> np.ndarray:
    return np.concatenate(xs)


# --------------------------------------------------------------------------- local vs network
def compare(sites, co, stage: str, target: str, p: GBDTParams, folds: int, seed: int, *, n_boot: int = 0,
            checks: list[int] | None = None, growth: bool = False) -> tuple[dict, tuple]:
    ids = [s.site_id for s in sites]
    adapt_p = GBDTParams.default(trees=ADAPT_TREES)
    keep = {s.site_id: {"y": [], "alone": [], "network": [], "adapted": [], "cn": {}, "ca": {}, "ntrain": []} for s in sites}
    grow: dict = {}
    fold0 = None
    secs = []
    for k in range(folds):
        tr, te = _cv(k, folds, seed)
        t0 = time.time()
        net, _ = federated_gbdt(co, ids, stage, target, tr, p)
        secs.append(time.time() - t0)
        if k == 0:
            fold0 = (net, te)
        for s in sites:
            d, B, y = s.cohort(stage, target)
            r_te = s.rows(stage, target, te)
            Bt = B[r_te]
            alone = fit_gbdt(PooledNetwork.from_sites([s], stage, target, tr), stage, target, p)
            adapted = fit_gbdt(PooledNetwork.from_sites([s], stage, target, tr), stage, target, adapt_p, start=net)
            K = keep[s.site_id]
            K["y"].append(y[r_te])
            K["ntrain"].append(len(s.rows(stage, target, tr)))
            K["alone"].append(raw_score(alone, Bt))
            K["network"].append(raw_score(net, Bt))
            K["adapted"].append(raw_score(adapted, Bt))
            if checks:
                for t, F in staged(net, Bt, checks).items():
                    K["cn"].setdefault(t, []).append(F)
                for t, F in staged(alone, Bt, checks).items():
                    K["ca"].setdefault(t, []).append(F)
        if growth:
            for size in range(2, len(sites)):
                for T in itertools.combinations(range(len(sites)), size):
                    m = fit_gbdt(PooledNetwork.from_sites([sites[i] for i in T], stage, target, tr), stage, target, p)
                    for i in T:
                        s = sites[i]
                        d, B, y = s.cohort(stage, target)
                        grow.setdefault((T, s.site_id), []).append(raw_score(m, B[s.rows(stage, target, te)]))
        log.info("%s/%s fold %d (%.1f s federated)", stage, target, k + 1, secs[-1])

    res: dict = {"stage": stage, "target": target, "per_centre": []}
    if checks:
        res["curves"] = {"trees": [t for t in checks if t > 0]}
    if growth:
        res["growth"] = {"centres": list(range(1, len(sites) + 1))}
    pooled = {m: [] for m in ("y",) + MODELS}
    for i, s in enumerate(sites):
        K = keep[s.site_id]
        y = _cat(K["y"])
        P = {m: sigmoid(_cat(K[m])) for m in MODELS}
        entry = {"id": s.site_id, "n": int(len(y)), "events": int(y.sum()), "n_train": int(np.mean(K["ntrain"]))}
        if n_boot:
            ci = M.bootstrap_ci(y, {"alone": P["alone"], "network": P["network"], "adapted": P["adapted"]},
                                n_boot=n_boot, seed=seed + i, ref="alone")
            for m in MODELS:
                entry[m] = _metrics(y, P[m], ci[m])
            for m in ("network", "adapted"):
                dl = ci[m]["delta_auroc_vs_ref"]               # alone - model
                entry[f"gain_{m}"] = {"mean": _r(-dl["mean"]), "ci": [_r(-dl["ci"][1]), _r(-dl["ci"][0])],
                                      "p": _r(dl["p_two_sided"], 3)}
        else:
            for m in MODELS:
                entry[m] = {"auroc": _r(auc(y, P[m]))}
        res["per_centre"].append(entry)
        pooled["y"].append(y)
        for m in MODELS:
            pooled[m].append(P[m])
        if checks:
            res["curves"][s.site_id] = {
                "network": [_r(auc(y, sigmoid(_cat(K["cn"][t])))) for t in checks if t > 0],
                "alone": [_r(auc(y, sigmoid(_cat(K["ca"][t])))) for t in checks if t > 0]}
        if growth:
            row = [auc(y, P["alone"])]
            for size in range(2, len(sites)):
                row.append(float(np.mean([auc(y, sigmoid(_cat(v))) for (T, sid), v in grow.items()
                                          if sid == s.site_id and len(T) == size])))
            row.append(auc(y, P["network"]))
            res["growth"][s.site_id] = [_r(v) for v in row]
    y = _cat(pooled["y"])
    res["pooled"] = {m: _metrics(y, _cat(pooled[m])) for m in MODELS}
    res["mean"] = {m: _r(np.mean([e[m]["auroc"] for e in res["per_centre"]])) for m in MODELS}
    if checks:
        res["curves"]["mean"] = {m: [_r(np.mean([res["curves"][s.site_id][m][j] for s in sites]))
                                     for j in range(len(res["curves"]["trees"]))] for m in ("network", "alone")}
    if growth:
        res["growth"]["mean"] = [_r(np.mean([res["growth"][s.site_id][j] for s in sites])) for j in range(len(sites))]
    res["seconds_per_federated_training"] = _r(np.mean(secs), 2)
    return res, fold0


def size_effect(sites, stage: str, target: str, p: GBDTParams, folds: int, seed: int, reps: int, full: dict) -> dict:
    """A centre that contributes only m of its training patients: alone vs joined with the others."""
    acc = {(m, r, s.site_id, kind): [] for m in SIZES for r in range(reps) for s in sites for kind in ("alone", "joined")}
    ys = {s.site_id: [] for s in sites}
    for k in range(folds):
        tr, te = _cv(k, folds, seed)
        for hi, h in enumerate(sites):
            d, B, y = h.cohort(stage, target)
            r_tr, r_te = h.rows(stage, target, tr), h.rows(stage, target, te)
            ys[h.site_id].append(y[r_te])
            others = []
            for s in sites:
                if s is not h:
                    ds, Bs, yy = s.cohort(stage, target)
                    rr = s.rows(stage, target, tr)
                    others.append((ds.iloc[rr], Bs[rr], yy[rr]))
            for m in SIZES:
                for r in range(reps):
                    rng = np.random.default_rng([seed, k, m, r, hi])
                    pos, neg = r_tr[y[r_tr] == 1], r_tr[y[r_tr] == 0]
                    n1 = int(round(m * len(pos) / len(r_tr)))
                    sub = np.concatenate([rng.choice(pos, n1, replace=False), rng.choice(neg, m - n1, replace=False)])
                    part = (d.iloc[sub], B[sub], y[sub])
                    alone = fit_gbdt(PooledNetwork([part], stage, target), stage, target, p)
                    joined = fit_gbdt(PooledNetwork(others + [part], stage, target), stage, target, p)
                    acc[(m, r, h.site_id, "alone")].append(raw_score(alone, B[r_te]))
                    acc[(m, r, h.site_id, "joined")].append(raw_score(joined, B[r_te]))
        log.info("size effect %s fold %d", target, k + 1)
    out = {"sizes": list(SIZES), "alone": [], "network": [], "per_centre": {}}
    for s in sites:
        y = _cat(ys[s.site_id])
        out["per_centre"][s.site_id] = {
            kind: [_r(np.mean([auc(y, sigmoid(_cat(acc[(m, r, s.site_id, src)]))) for r in range(reps)])) for m in SIZES]
            for kind, src in (("alone", "alone"), ("network", "joined"))}
    for kind in ("alone", "network"):
        out[kind] = [_r(np.mean([out["per_centre"][s.site_id][kind][j] for s in sites])) for j in range(len(SIZES))]
    out["full"] = {"n_train": int(np.mean([e["n_train"] for e in full["per_centre"]])),
                   "alone": full["mean"]["alone"], "network": full["mean"]["network"]}
    return out


def leave_one_out(sites, co, target: str, p: GBDTParams, n_boot: int, seed: int) -> list[dict]:
    rows = []
    for i, h in enumerate(sites):
        others = [s.site_id for s in sites if s is not h]
        model, _ = federated_gbdt(co, others, STAGE, target, None, p)
        d, B, y = h.cohort(STAGE, target)
        pr = sigmoid(raw_score(model, B))
        ci = M.bootstrap_ci(y, {"m": pr}, n_boot=n_boot, seed=seed + 10 + i)
        rows.append({"id": h.site_id, "trained_on": others, "n": int(len(y)), "events": int(y.sum()),
                     "observed": _r(y.mean()), **_metrics(y, pr, ci["m"])})
    return rows


# --------------------------------------------------------------------------- proof of equivalence
def equivalence(sites, co, p: GBDTParams, fold0) -> tuple[dict, dict]:
    ids = [s.site_id for s in sites]
    out = {}
    t0 = time.time()
    fed, job = federated_gbdt(co, ids, STAGE, OUTCOMES[0], None, p)
    t_fed = time.time() - t0
    pool = PooledNetwork.from_sites(sites, STAGE, OUTCOMES[0])
    ref = fit_gbdt(pool, STAGE, OUTCOMES[0], p)
    pf, pr = sigmoid(raw_score(fed, pool.B)), sigmoid(raw_score(ref, pool.B))
    lv = leaf_sizes(fed)
    out["gbdt"] = {"identical": fed["trees"] == ref["trees"] and fed["init"] == ref["init"], "trees": len(fed["trees"]),
                   "leaves": int(sum(1 for t in fed["trees"] for nd in t["nodes"] if "col" not in nd)),
                   "min_leaf": int(min(lv)), "max_abs_risk_diff": float(np.max(np.abs(pf - pr))),
                   "rounds": job.round, "seconds": _r(t_fed, 2)}
    lr_fed, ljob = federated_logreg(co, ids, STAGE, OUTCOMES[0])
    lr_ref = fit_logreg(pool, spline_spec(STAGE))
    X, _ = spline_design(pool.d, spline_spec(STAGE))
    Z = (X - np.asarray(lr_fed["mean"])) / np.asarray(lr_fed["sd"])
    sk = LogisticRegression(C=1.0 / lr_fed["lambda"], max_iter=10000, tol=1e-10).fit(Z, pool.y)
    p_fed, p_ref = predict_spline(lr_fed, pool.d), predict_spline(lr_ref, pool.d)
    out["logreg"] = {"coefficients": len(lr_fed["beta"]), "iterations": lr_fed["iterations"], "rounds": ljob.round,
                     "max_abs_coef_diff": float(np.max(np.abs(np.asarray(lr_fed["beta"]) - np.asarray(lr_ref["beta"])))),
                     "max_abs_risk_diff": float(np.max(np.abs(p_fed - p_ref))),
                     "max_abs_risk_diff_sklearn": float(np.max(np.abs(p_fed - sk.predict_proba(Z)[:, 1])))}
    model0, te0 = fold0
    res, _ = federated_eval(co, ids, model0, te0)
    yy, pp = [], []
    for s in sites:
        d, B, y = s.cohort(STAGE, OUTCOMES[0])
        r = s.rows(STAGE, OUTCOMES[0], te0)
        yy.append(y[r])
        pp.append(sigmoid(raw_score(model0, B[r])))
    exact = M.summary(_cat(yy), _cat(pp))
    out["eval"] = {"n": res["n"], **{f"{k}_{src}": _r(v[k], 5) for k in ("auroc", "slope", "citl", "brier")
                                     for src, v in (("federated", res), ("exact", exact))}}
    return out, {"gbdt": fed, "fold0": res}


def http_run(sites, p: GBDTParams, reference: dict, fold0) -> dict:
    """Coordinator and four site nodes talking over real HTTP sockets on this machine."""
    tokens = {s.site_id: secrets.token_urlsafe(24) for s in sites}
    hub = Hub({sid: token_hash(t) for sid, t in tokens.items()}, name="MTOP demo coordinator")
    httpd, _ = serve(hub, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    clients = [SiteClient(s, url, tokens[s.site_id], wait=10) for s in sites]
    for c in clients:
        c.start()
    try:
        ids = hub.wait_for(len(sites), timeout=60)
        co = Coordinator(hub)
        hub.status.update(phase="training")

        def on_tree(t, tree, model):
            hub.status["job"] = {"kind": "gbdt", "tree": t + 1, "trees": p.trees}
            hub.status["curve"] = list(model["train_loss"])

        t0 = time.time()
        model, job = federated_gbdt(co, ids, STAGE, OUTCOMES[0], None, p, on_tree)
        t_train = time.time() - t0
        model0, te0 = fold0
        t1 = time.time()
        res, ejob = federated_eval(co, ids, model0, te0)
        t_eval = time.time() - t1
        stats = {sid: dict(v) for sid, v in hub.stats.items()}
        lat = [r["seconds"] for r in job.rounds]
        kinds = {}
        for r in job.rounds:
            kinds[r["op"]] = kinds.get(r["op"], 0) + 1
        return {
            "transport": "HTTP with long polling, coordinator and four nodes on one machine",
            "centres": len(ids), "rounds": job.round, "round_kinds": kinds, "eval_rounds": ejob.round,
            "values_per_round": int(np.median([r["values"] for r in job.rounds])),
            "seconds_train": _r(t_train, 2), "seconds_eval": _r(t_eval, 2),
            "median_round_ms": _r(1000 * float(np.median(lat)), 1),
            "kb_up_per_centre": _r(np.mean([v["up"] for v in stats.values()]) / 1000, 1),
            "kb_down_per_centre": _r(np.mean([v["down"] for v in stats.values()]) / 1000, 1),
            "identical_to_pooled": model["trees"] == reference["gbdt"]["trees"],
            "eval_identical": abs(res["auroc"] - reference["fold0"]["auroc"]) < 1e-12,
            "sample": co.sample,
        }
    finally:
        for c in clients:
            c.stop.set()
        httpd.shutdown()


# --------------------------------------------------------------------------- main
def run(http: bool = True, quick: bool = False) -> dict:
    ensure_dirs()
    cfg = settings()
    seed = int(cfg["seed"])
    k = int(cfg["privacy"]["min_cell"])
    p = GBDTParams.default(**({"trees": 30} if quick else {}))
    folds = 3 if quick else 5
    n_boot = 200 if quick else int(cfg["bootstrap"]["n"])
    reps = 1 if quick else 2
    checks = list(range(0, p.trees + 1, 5))
    clean = pd.read_parquet(PROCESSED / "clean.parquet")
    base, _ = flow(clean)
    sites, reports = make_sites(base)
    co = Coordinator(LocalTransport(sites), log=log.info)
    t0 = time.time()

    outcomes, fold0 = {}, None
    for target in OUTCOMES:
        res, f0 = compare(sites, co, STAGE, target, p, folds, seed, n_boot=n_boot, checks=checks, growth=True)
        fold0 = fold0 or f0
        res["size"] = size_effect(sites, STAGE, target, p, folds, seed, reps, res)
        res["loso"] = leave_one_out(sites, co, target, p, n_boot, seed)
        outcomes[target] = res
    stages = [{"stage": STAGE, "target": t, "mean": outcomes[t]["mean"],
               "per_centre": [{"id": e["id"], "n": e["n"], "events": e["events"], **{m: e[m]["auroc"] for m in MODELS}}
                              for e in outcomes[t]["per_centre"]]} for t in OUTCOMES]
    for st, t in MORE_STAGES:
        res, _ = compare(sites, co, st, t, p, folds, seed)
        stages.append({"stage": st, "target": t, "mean": res["mean"],
                       "per_centre": [{"id": e["id"], "n": e["n"], "events": e["events"], **{m: e[m]["auroc"] for m in MODELS}}
                                      for e in res["per_centre"]]})
    eq, ref = equivalence(sites, co, p, fold0)
    run_stats = http_run(sites, p, ref, fold0) if http else None

    centres = []
    for (cid, a, z), s in zip(CENTRES, sites):
        c = {"id": cid, "era": f"{a}–{z}", "unrecognised_values": int(sum(reports[cid]["unrecognised_values"].values()))}
        for t in OUTCOMES:
            y = s.cohort(STAGE, t)[2]
            c[t] = {"n": int(len(y)), "events": int(y.sum()), "rate": _r(y.mean(), 3)}
        centres.append(c)
    net = fed_settings()["network"]
    out = {
        "generated": time.strftime("%Y-%m-%d"), "protocol": PROTOCOL, "stage": STAGE, "quick": quick, "folds": folds,
        "adapt_trees": ADAPT_TREES,
        "recipe": {k2: v for k2, v in asdict(p).items() if k2 in ("trees", "learning_rate", "max_depth", "min_child", "reg_lambda")},
        "settings": {"min_sites": int(net["min_sites"]), "min_site_patients": int(net["min_site_patients"]),
                     "min_leaf": int(cfg["privacy"]["min_leaf"]), "fixed_point_bits": int(net["fixed_point_bits"]),
                     "bands_max": int(max(len(e) for e in fed_settings()["bands"].values()) + 1)},
        "crypto": {"key_agreement": "X25519, fresh keys per job, signed with Ed25519 identity keys",
                   "kdf": "HKDF-SHA256", "masks": "ChaCha20 key stream", "arithmetic": "integers modulo 2^64"},
        "centres": centres, "outcomes": outcomes, "stages": stages, "equivalence": eq, "run": run_stats,
        "seconds_total": _r(time.time() - t0, 1),
    }
    _check_public(out, k)
    (PUBLIC / "federated.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    _report(out)
    return out


def _check_public(out: dict, k: int) -> None:
    groups = [c[t] for c in out["centres"] for t in OUTCOMES]
    for res in out["outcomes"].values():
        groups += res["per_centre"] + res["loso"]
    for st in out["stages"]:
        groups += st["per_centre"]
    for g in groups:
        assert g["n"] >= k and g["events"] >= k and g["n"] - g["events"] >= k, "small group in federated.json"


def _report(o: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    L = [f"# Federated demo ({o['generated']})", "",
         f"Four virtual centres from the FNO registry by era; {o['stage']} models; {o['folds']}-fold CV inside each centre; "
         f"'adapted' = network model + {o['adapt_trees']} local trees.", ""]
    for t, res in o["outcomes"].items():
        L += [f"## {t}", "", "| Centre | n | Alone | Network | Adapted | Gain network (95% CI) | Gain adapted (95% CI) | CITL network | CITL adapted |",
              "|---|---|---|---|---|---|---|---|---|"]
        for e in res["per_centre"]:
            gn, ga = e["gain_network"], e["gain_adapted"]
            L.append(f"| {e['id']} | {e['n']} | {e['alone']['auroc']} | {e['network']['auroc']} | {e['adapted']['auroc']} | "
                     f"{gn['mean']} ({gn['ci'][0]} to {gn['ci'][1]}) | {ga['mean']} ({ga['ci'][0]} to {ga['ci'][1]}) | "
                     f"{e['network']['citl']} | {e['adapted']['citl']} |")
        L += ["", f"Mean: {res['mean']}", f"Growth (1-4 centres): {res['growth']['mean']}",
              f"Size {res['size']['sizes']}: alone {res['size']['alone']}, network {res['size']['network']}", "",
              "| Held-out centre | n | AUROC | CITL | Slope |", "|---|---|---|---|---|"]
        for r in res["loso"]:
            L.append(f"| {r['id']} | {r['n']} | {r['auroc']} ({r['auroc_ci'][0]}–{r['auroc_ci'][1]}) | {r['citl']} | {r['slope']} |")
        L.append("")
    L += ["## Stages", "", "| Stage | Outcome | Alone | Network | Adapted |", "|---|---|---|---|---|"]
    for s in o["stages"]:
        L.append(f"| {s['stage']} | {s['target']} | {s['mean']['alone']} | {s['mean']['network']} | {s['mean']['adapted']} |")
    e = o["equivalence"]
    L += ["", f"Equivalence: trees identical to pooled training: {e['gbdt']['identical']}; spline LR max |Δβ| "
          f"{e['logreg']['max_abs_coef_diff']:.2e}; federated AUROC {e['eval']['auroc_federated']} vs exact {e['eval']['auroc_exact']}."]
    if o["run"]:
        r = o["run"]
        L.append(f"HTTP run: {r['rounds']} rounds in {r['seconds_train']} s, {r['kb_up_per_centre']} kB up per centre, "
                 f"identical: {r['identical_to_pooled']}.")
    (REPORTS / "federated.md").write_text("\n".join(L) + "\n", encoding="utf-8")
