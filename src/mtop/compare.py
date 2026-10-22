"""The learners added by amendment 1 (docs/amendment_1.md), validated without re-tuning the five
main learners of train.py.

Candidates (kind "learner", eligible for the model choice and published; extra_learners.py):
  EBM, LR + interactions, LR + multiple imputation, Stacking.
Comparison only (kind "comparison", reported, never chosen or published):
  EBM (defaults)            interpret's defaults throughout, as in the 2026 report
  CatBoost                  default settings, as in the 2026 report
  Probability averaging     mean of the stacking's three base models (reference, XGBoost, EBM)
  FT-Transformer            rtdl FT-Transformer, default settings; epochs by early stopping on a fifth
                            of the development patients, then refitted on all of them
  FT-Transformer, tuned     the same with 12 Optuna trials (width, depth, dropout, learning rate, decay)
  TabPFN v2, TabICL v2      only from local weight files in data/models (never an online service)
The MLP of train.py stays comparison only.

Everything is computed as in train.validate_split: same patients, design matrix and temporal split;
AUROC, Brier, calibration with bootstrap CIs and the paired AUROC difference against the reference
(same seed, so the same bootstrap draws), DeLong's test against the reference, ROC, calibration bins,
decision curves, fixed thresholds, subgroups and the updating analysis. Every published or bedside
score of scores.STAGE_SCORES is recomputed too (and now also gets the updating analysis), and every
existing learner gets its DeLong p-value from the saved predictions.

Nested CV (admission, mRS 0-2): the 15 outer folds of nested.py; inside each outer training part the
reference and XGBoost are tuned again (the nested budget) and passed to the stacking and the multiple
imputation; "FT-Transformer, tuned" is left out (cost).

Chunks run in parallel (python -m mtop compare --split ... --stage ... --learner ...; --nested 0-4);
each writes artifacts/private/compare/*.json and *_predictions.parquet; `--merge` folds them into
train_<split>.json, predictions_<split>.parquet and nested_cv.json.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import learners as Lr
from . import metrics as M
from .cohort import split_temporal, splits, stage_cohort
from .config import settings, stage_ids, targets
from .features import design_matrix, monotone_vector
from .scores import STAGE_SCORES, benchmark_predictions
from .tuning import jsonable, threads

warnings.filterwarnings("ignore")

CANDIDATES = list(Lr.NEW)
COMPARISON = ["EBM (defaults)", "CatBoost", "Probability averaging", "FT-Transformer", "FT-Transformer, tuned",
              "TabPFN v2", "TabICL v2"]
NO_NESTED = {"FT-Transformer, tuned"}
WEIGHTS = {"TabPFN v2": "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
           "TabICL v2": "tabicl-classifier-v2-20260212.ckpt"}
TORCH = {"FT-Transformer", "FT-Transformer, tuned", "TabPFN v2", "TabICL v2"}


def ref_hash(ids, p_ref) -> str:
    """Fingerprint of the reference's saved test predictions (in patient-ID order): a chunk computed on
    other data or another split no longer matches the training results and is not merged."""
    import hashlib

    order = np.argsort(np.asarray(ids).astype(str), kind="stable")
    return hashlib.sha256(np.round(np.asarray(p_ref, float)[order], 6).tobytes()).hexdigest()[:16]


def out_dir(out_private: Path) -> Path:
    d = Path(out_private) / "compare"
    d.mkdir(parents=True, exist_ok=True)
    return d


def weights_dir() -> Path:
    from .paths import ROOT

    return ROOT / "data" / "models"


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def available(names=None) -> list[str]:
    """Requested learners that can run here (FT-Transformer needs torch; the foundation models need
    their weight files too)."""
    names = list(names or (CANDIDATES + COMPARISON))
    ok = []
    for n in names:
        if n in WEIGHTS and not (weights_dir() / WEIGHTS[n]).exists():
            continue
        if n in TORCH and not _has("torch"):
            continue
        ok.append(n)
    return ok


# ------------------------------------------------------------------- comparison-only learners
def fit_catboost(X, y, seed: int):
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(random_seed=seed, verbose=False, thread_count=threads(), allow_writing_files=False)
    m.fit(X, np.asarray(y, int))
    return m


FT_MAX_EPOCHS, FT_PATIENCE, FT_BATCH = 200, 16, 64


def _ft_arrays(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    sc = StandardScaler()
    A = sc.fit_transform(imp.fit_transform(X_tr))
    B = sc.transform(imp.transform(X_te))
    return np.nan_to_num(A).astype(np.float32), np.nan_to_num(B).astype(np.float32)


def _ft_default() -> dict:
    from rtdl_revisiting_models import FTTransformer

    arch = {k: v for k, v in FTTransformer.get_default_kwargs().items() if not k.startswith("_")}
    return {"arch": arch, "lr": 1e-4, "weight_decay": 1e-5}


def _ft_predict(model, A) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        z = torch.cat([model(torch.from_numpy(A[i:i + 512]), None)[:, 0] for i in range(0, len(A), 512)])
    return torch.sigmoid(z).numpy().astype(float)


def _ft_train(A, y, cfg: dict, seed: int, epochs: int | None = None, val=None, max_epochs: int = FT_MAX_EPOCHS,
              patience: int = FT_PATIENCE):
    """Train for `epochs`, or with early stopping on `val` = (A_val, y_val): (model, best epoch, val loss)."""
    import torch
    from rtdl_revisiting_models import FTTransformer

    torch.set_num_threads(threads())
    torch.manual_seed(seed)
    model = FTTransformer(n_cont_features=A.shape[1], cat_cardinalities=[], d_out=1, **cfg["arch"])
    groups = model.make_parameter_groups() if hasattr(model, "make_parameter_groups") else model.parameters()
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(A), torch.from_numpy(np.asarray(y, np.float32))
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    for ep in range(1, (epochs or max_epochs) + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), FT_BATCH):
            b = perm[i:i + FT_BATCH]
            opt.zero_grad()
            lossf(model(Xt[b], None)[:, 0], yt[b]).backward()
            opt.step()
        if val is not None:
            p = np.clip(_ft_predict(model, val[0]), 1e-7, 1 - 1e-7)
            vl = float(-np.mean(val[1] * np.log(p) + (1 - val[1]) * np.log(1 - p)))
            if vl < best - 1e-5:
                best, best_ep, wait = vl, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_ep if val is not None else epochs), (best if val is not None else None)


def fit_ft(X_tr: pd.DataFrame, y_tr, X_te: pd.DataFrame, seed: int, tuned: bool = False, trials: int = 12):
    from sklearn.model_selection import train_test_split

    y_tr = np.asarray(y_tr, float)
    i_fit, i_val = train_test_split(np.arange(len(y_tr)), test_size=0.2, stratify=y_tr, random_state=seed)
    A_fit, A_val = _ft_arrays(X_tr.iloc[i_fit], X_tr.iloc[i_val])
    cfg = _ft_default()
    if tuned:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            c = {"arch": {"n_blocks": trial.suggest_int("n_blocks", 1, 4),
                          "d_block": trial.suggest_categorical("d_block", [64, 128, 192]),
                          "attention_n_heads": 8,
                          "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
                          "ffn_d_hidden": None,
                          "ffn_d_hidden_multiplier": trial.suggest_categorical("ffn_multiplier", [2 / 3, 4 / 3, 2.0]),
                          "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
                          "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.2)},
                 "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
                 "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)}
            _, ep, vl = _ft_train(A_fit, y_tr[i_fit], c, seed, val=(A_val, y_tr[i_val]), max_epochs=150, patience=12)
            trial.set_user_attr("cfg", c)
            trial.set_user_attr("epochs", ep)
            return vl

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        cfg, epochs, val_loss = study.best_trial.user_attrs["cfg"], study.best_trial.user_attrs["epochs"], study.best_value
    else:
        _, epochs, val_loss = _ft_train(A_fit, y_tr[i_fit], cfg, seed, val=(A_val, y_tr[i_val]))
    A_all, A_te = _ft_arrays(X_tr, X_te)
    model, _, _ = _ft_train(A_all, y_tr, cfg, seed, epochs=max(1, int(epochs)))
    info = {**{k: v for k, v in cfg["arch"].items() if v is not None}, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
            "epochs": int(epochs), "val_logloss": float(val_loss), "trials": trials if tuned else 0}
    # full precision (not jsonable's 6 decimals), so the model can be rebuilt exactly (shap_all.FTModel)
    exact = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (np.integer,)) else v)
             for k, v in info.items()}
    return _ft_predict(model, A_te), exact


def fit_tabpfn(X_tr, y_tr, X_te, seed: int):
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(model_path=str(weights_dir() / WEIGHTS["TabPFN v2"]), device="cpu", random_state=seed,
                           ignore_pretraining_limits=True)
    clf.fit(X_tr.to_numpy(float), np.asarray(y_tr, int))
    return clf.predict_proba(X_te.to_numpy(float))[:, 1], {"weights": WEIGHTS["TabPFN v2"]}


def fit_tabicl(X_tr, y_tr, X_te, seed: int):
    from tabicl import TabICLClassifier

    clf = TabICLClassifier(model_path=str(weights_dir() / WEIGHTS["TabICL v2"]), checkpoint_version=WEIGHTS["TabICL v2"],
                           allow_auto_download=False, device="cpu", random_state=seed)
    clf.fit(X_tr.to_numpy(float), np.asarray(y_tr, int))
    return clf.predict_proba(X_te.to_numpy(float))[:, 1], {"weights": WEIGHTS["TabICL v2"]}


# ------------------------------------------------------------------------------- one cohort
def predictions(X_tr: pd.DataFrame, y_tr, X_te: pd.DataFrame, base_settings: dict, seed: int, trials: int, mono,
                min_leaf: int, names: list[str], log=print) -> tuple[dict, dict, dict]:
    """Test probabilities, settings and fitted candidates of every requested learner. `base_settings`:
    the reference's and XGBoost's tuned settings on these training patients."""
    from . import extra_learners as XL

    y_tr = np.asarray(y_tr, float)
    preds, info, fitted = {}, {}, {}

    def timed(label, fn):
        t = time.time()
        r = fn()
        log(f"      {label} ({time.time() - t:.0f}s)")
        return r

    for name in [n for n in CANDIDATES if n in names or (n == "Stacking" and "Probability averaging" in names)]:
        f = timed(name, lambda: Lr.tune(name, X_tr, y_tr, seed, trials, mono, min_leaf=min_leaf, base_settings=base_settings))
        fitted[name] = f
        if name in names:
            preds[name], info[name] = f.proba(X_te), f.settings
    if "Probability averaging" in names:
        bases = fitted["Stacking"].model.bases
        preds["Probability averaging"] = np.mean([b.proba(X_te) for b in bases], axis=0)
        info["Probability averaging"] = {"base": fitted["Stacking"].model.names}
    if "EBM (defaults)" in names:
        e = timed("EBM (defaults)", lambda: XL.fit_ebm(X_tr, y_tr, seed, threads=threads(), publishable=False))
        preds["EBM (defaults)"], info["EBM (defaults)"] = e.predict_proba(X_te)[:, 1], e.settings
    if "CatBoost" in names:
        cb = timed("CatBoost", lambda: fit_catboost(X_tr, y_tr, seed))
        preds["CatBoost"], info["CatBoost"] = cb.predict_proba(X_te)[:, 1], {"settings": "CatBoost defaults",
                                                                              "trees": int(cb.tree_count_)}
    for name, fn in (("FT-Transformer", lambda: fit_ft(X_tr, y_tr, X_te, seed)),
                     ("FT-Transformer, tuned", lambda: fit_ft(X_tr, y_tr, X_te, seed, tuned=True)),
                     ("TabPFN v2", lambda: fit_tabpfn(X_tr, y_tr, X_te, seed)),
                     ("TabICL v2", lambda: fit_tabicl(X_tr, y_tr, X_te, seed))):
        if name in names:
            preds[name], info[name] = timed(name, fn)
    return preds, info, fitted


# ----------------------------------------------------------------------------- temporal split
def run_temporal(base: pd.DataFrame, out_private: Path, label: str, stages=None, names=None, log=print,
                 tag: str = "main") -> None:
    """`tag` names this run's chunk files, so runs with different learners can go in parallel."""
    from .train import subgroup_table, threshold_table, updating

    cfg = settings()
    seed, n_boot, trials = int(cfg["seed"]), int(cfg["bootstrap"]["n"]), int(cfg["tuning"]["trials"])
    min_cell, min_leaf = int(cfg["privacy"]["min_cell"]), int(cfg["privacy"]["min_leaf"])
    names = available(names)
    test_from = splits()[label]
    train = json.loads((Path(out_private) / f"train_{label}.json").read_text())
    saved = pd.read_parquet(Path(out_private) / f"predictions_{label}.parquet")
    for stage in stages or stage_ids():
        d = stage_cohort(base, stage)
        dev, test = split_temporal(d, test_from)
        X_all = design_matrix(d, stage)
        X_dev, X_test = X_all.loc[dev.index], X_all.loc[test.index]
        for target in targets():
            t0 = time.time()
            y_dev, y_test = dev[target].to_numpy(float), test[target].to_numpy(float)
            tres = train["stages"][stage]["targets"][target]
            sv = saved[(saved["stage"] == stage) & (saved["target"] == target)]
            old = {m: g.set_index("patient_id").loc[test["patient_id"].to_numpy(), "p"].to_numpy(float)
                   for m, g in sv.groupby("model") if tres["models"].get(m, {}).get("kind") == "learner"}
            p_ref = old[Lr.REFERENCE]
            log(f"  [{label} {stage}/{target}] n_dev={len(dev)} n_test={len(test)}: {', '.join(names)}")
            preds, info, _ = predictions(X_dev, y_dev, X_test, tres["settings"], seed, trials,
                                         monotone_vector(stage, target), min_leaf, names, log)
            scores = benchmark_predictions(dev, test, stage, target, seed)
            allp = {**preds, **scores}
            # same seed and patients as train.validate_split: the same bootstrap draws
            ci = M.bootstrap_ci(y_test, {Lr.REFERENCE: p_ref, **allp}, n_boot=n_boot, seed=seed, ref=Lr.REFERENCE)
            kinds = {k: ("learner" if k in CANDIDATES else "comparison") for k in preds}
            models = {}
            for k, v in allp.items():
                models[k] = {**M.summary(y_test, v), "ci": ci[k], "oe": M.oe_ratio(y_test, v), "kind": kinds.get(k, "score"),
                             "delong_p": M.delong_p(y_test, v, p_ref)}
            nb = 8 if len(y_test) < 250 else 10
            curves = {"roc": {k: M.roc_points(y_test, v) for k, v in allp.items()},
                      "calibration": {k: M.calibration_bins(y_test, v, n_bins=nb, min_n=min_cell) for k, v in allp.items()},
                      "dca": {k: [{"t": r["t"], "nb": r["model"]} for r in M.decision_curve(y_test, v)] for k, v in allp.items()}}
            per_model = {"thresholds": {k: threshold_table(y_test, v, min_cell) for k, v in preds.items()},
                         "subgroups": {k: subgroup_table(test, y_test, v, min_cell) for k, v in preds.items()},
                         "updating": {k: updating(y_test, v, test["year"].to_numpy(), test_from) for k, v in allp.items()}}
            delong_old = {k: M.delong_p(y_test, v, p_ref) for k, v in old.items() if k != Lr.REFERENCE}
            jf = out_dir(out_private) / f"{label}_{stage}_{target}__{tag}.json"
            pf = out_dir(out_private) / f"{label}_{stage}_{target}__{tag}_predictions.parquet"
            if jf.exists():           # a run with some learners adds to an earlier run with others
                prev = json.loads(jf.read_text())
                keep = [k for k, v in prev["models"].items() if v["kind"] != "score" and k not in preds]
                models = {**{k: prev["models"][k] for k in keep}, **models}
                info = {**{k: v for k, v in prev["settings"].items() if k in keep}, **info}
                for part in ("roc", "calibration", "dca"):
                    curves[part] = {**{k: prev["curves"][part][k] for k in keep if k in prev["curves"][part]}, **curves[part]}
                for part in ("thresholds", "subgroups", "updating"):
                    per_model[part] = {**{k: prev[part][k] for k in keep if k in prev[part]}, **per_model[part]}
            out = {"split": label, "stage": stage, "target": target, "models": models, "settings": info, "curves": curves,
                   **per_model, "scores": list(scores), "delong_existing": delong_old, "seconds": round(time.time() - t0),
                   "ref_hash": ref_hash(test["patient_id"].to_numpy(), p_ref)}
            jf.write_text(json.dumps(out, indent=1))
            rows = pd.DataFrame([{"patient_id": pid, "split": label, "stage": stage, "target": target, "model": k, "y": yy, "p": pp}
                                 for k, v in allp.items() for pid, yy, pp in zip(test["patient_id"], y_test, v)])
            if pf.exists():
                prev_rows = pd.read_parquet(pf)
                rows = pd.concat([prev_rows[~prev_rows["model"].isin(list(allp))], rows], ignore_index=True)
            rows.to_parquet(pf, index=False)
            msg = ", ".join(f"{k} {models[k]['auroc']:.3f}" for k in preds)
            log(f"  [{label} {stage}/{target}] reference {tres['models'][Lr.REFERENCE]['auroc']:.3f}; {msg} "
                f"({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------------ nested CV
def run_nested(base: pd.DataFrame, out_private: Path, folds_k: list[int], names=None, log=print,
               tag: str = "main") -> None:
    from sklearn.model_selection import RepeatedStratifiedKFold

    from .splines import fit_spline_logreg
    from .tuning import tune_xgb

    cfg = settings()
    nc = cfg["nested_cv"]
    seed, trials = int(cfg["seed"]), int(nc["trials"])
    min_leaf = int(cfg["privacy"]["min_leaf"])
    names = [n for n in available(names) if n not in NO_NESTED]
    stage, target = "admission", "good_outcome"
    d = stage_cohort(base, stage)
    X = design_matrix(d, stage)
    y = d[target].to_numpy(float)
    mono = monotone_vector(stage, target)
    rskf = RepeatedStratifiedKFold(n_splits=int(nc["folds"]), n_repeats=int(nc["repeats"]), random_state=seed)
    for k, (tr, va) in enumerate(rskf.split(X, y)):
        if k not in folds_k:
            continue
        t0 = time.time()
        s = seed + 100 + k
        Xtr, ytr, Xva = X.iloc[tr], y[tr], X.iloc[va]
        spl = fit_spline_logreg(Xtr, ytr, s)
        xgb = tune_xgb(Xtr, ytr, s, trials)
        base_settings = {Lr.REFERENCE: spl.settings, "XGBoost": xgb.settings}
        preds, _, _ = predictions(Xtr, ytr, Xva, base_settings, s, trials, mono, min_leaf, names, log)
        res = {"fold": k, "n_test": int(len(va)),
               "reference_check": float(M.summary(y[va], spl.predict_proba(Xva)[:, 1])["auroc"]),
               "models": {nm: {m: M.summary(y[va], p)[m] for m in ("auroc", "brier", "citl", "slope")} for nm, p in preds.items()}}
        ff = out_dir(out_private) / f"nested_fold{k:02d}__{tag}.json"
        if ff.exists():
            prev = json.loads(ff.read_text())
            res["models"] = {**{nm: v for nm, v in prev["models"].items() if nm not in preds}, **res["models"]}
        ff.write_text(json.dumps(res, indent=1))
        log(f"  [nested fold {k + 1}] reference {res['reference_check']:.3f}; "
            + ", ".join(f"{nm} {v['auroc']:.3f}" for nm, v in res["models"].items()) + f" ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------- merge
def merge(out_private: Path, log=print) -> None:
    """Fold the chunks into train_<split>.json (candidates as kind "learner" with their settings,
    comparison-only learners as kind "comparison"; the recomputed scores replace the old ones),
    predictions_<split>.parquet and nested_cv.json."""
    from .nested import _summ

    od = out_dir(out_private)
    for label in splits():
        f = Path(out_private) / f"train_{label}.json"
        if not f.exists():
            continue
        train = json.loads(f.read_text())
        pf = Path(out_private) / f"predictions_{label}.parquet"
        saved = pd.read_parquet(pf)
        n, new_rows, new_models = 0, [], set()
        groups: dict = {}
        for part in sorted(od.glob(f"{label}_*__*.json")):          # one file per stage/outcome and run tag
            c = json.loads(part.read_text())
            tr = train["stages"][c["stage"]]["targets"][c["target"]]
            test_ids = saved[(saved["stage"] == c["stage"]) & (saved["target"] == c["target"]) & (saved["model"] == Lr.REFERENCE)]
            if c.get("ref_hash") and len(test_ids) and c["ref_hash"] != ref_hash(test_ids["patient_id"].to_numpy(), test_ids["p"].to_numpy(float)):
                log(f"merge: {part.name} was computed on other training results, skipped (run it again)")
                continue
            if Lr.REFERENCE not in tr["models"]:
                continue
            groups.setdefault((c["stage"], c["target"]), []).append((part, c))
        for (st, tg), parts in sorted(groups.items()):
            tres = train["stages"][st]["targets"][tg]
            main = {k: v for k, v in tres["models"].items() if v.get("kind") == "learner" and k not in CANDIDATES}
            added, settings_new = {}, {}
            per = {"roc": {}, "calibration": {}, "dca": {}, "thresholds": {}, "subgroups": {}, "updating": {}}
            parts = sorted(parts, key=lambda pc: pc[0].name.endswith("__main.json"))   # the main run's scores last
            for part, c in parts:
                learned = {k for k, v in c["models"].items() if v["kind"] in ("learner", "comparison")}
                added.update({k: c["models"][k] for k in learned})
                settings_new.update(c["settings"])
                for kind in ("roc", "calibration", "dca"):
                    per[kind].update({k: v for k, v in c["curves"][kind].items() if k in learned})
                per["thresholds"].update(c["thresholds"])
                per["subgroups"].update(c["subgroups"])
                per["updating"].update({k: v for k, v in c["updating"].items() if k in learned})
                for k, p in c.get("delong_existing", {}).items():
                    if k in main:
                        main[k]["delong_p"] = p
                ppart = part.with_name(part.stem + "_predictions.parquet")
                if ppart.exists():
                    new_rows.append(pd.read_parquet(ppart))
            c = parts[-1][1]                                          # scores: identical in every run
            scores = {k: c["models"][k] for k in STAGE_SCORES[st] if k in c["models"]}
            if Lr.REFERENCE in main:
                main[Lr.REFERENCE]["delong_p"] = None
            tres["models"] = {**main, **added, **scores}
            for kind in ("roc", "calibration", "dca"):
                tres["curves"][kind] = {**{k: v for k, v in tres["curves"][kind].items() if k in main}, **per[kind],
                                        **{k: c["curves"][kind][k] for k in scores}}
            for part_name in ("thresholds", "subgroups"):
                tres[part_name] = {**{k: v for k, v in tres[part_name].items() if k in main}, **per[part_name]}
            tres["updating"] = {**{k: v for k, v in tres["updating"].items() if k in main}, **per["updating"],
                                **{k: c["updating"][k] for k in scores if k in c["updating"]}}
            tres["settings"] = {**{k: v for k, v in tres["settings"].items() if k not in CANDIDATES},
                                **{k: v for k, v in settings_new.items() if k in CANDIDATES}}
            tres["comparison_settings"] = {k: v for k, v in settings_new.items() if k not in CANDIDATES}
            new_models |= set(added) | set(scores)
            n += 1
        if new_rows:
            new = pd.concat(new_rows, ignore_index=True).drop_duplicates(["stage", "target", "model", "patient_id"], keep="last")
            key = ["stage", "target", "model"]
            drop = saved.merge(new[key].drop_duplicates(), on=key, how="left", indicator=True)["_merge"] == "both"
            saved = pd.concat([saved[~drop.to_numpy()], new], ignore_index=True)
            saved.to_parquet(pf, index=False)
        f.write_text(json.dumps(train, indent=1))
        log(f"merge: {label}: {n} stage/outcome results ({len(new_models)} models added or refreshed)")
    nf = Path(out_private) / "nested_cv.json"
    by_fold: dict = {}
    for p in sorted(od.glob("nested_fold*__*.json")):              # one file per fold and run tag
        fo = json.loads(p.read_text())
        if fo["fold"] in by_fold:
            by_fold[fo["fold"]]["models"].update(fo["models"])
        else:
            by_fold[fo["fold"]] = fo
    folds = [by_fold[k] for k in sorted(by_fold)]
    if nf.exists() and folds:
        nested = json.loads(nf.read_text())
        st = nested["stages"]["admission"]["good_outcome"]
        ref = np.asarray(st["models"][Lr.REFERENCE]["fold_auroc"], float)
        total = int(nested["folds"]) * int(nested["repeats"])
        names = sorted({nm for fo in folds for nm in fo["models"]})
        for nm in names:
            got = sorted((fo["fold"], fo["models"][nm]) for fo in folds if nm in fo["models"])
            if len(got) != total:
                log(f"merge: nested {nm}: {len(got)}/{total} folds, left out")
                continue
            per = {m: [g[1][m] for g in got] for m in ("auroc", "brier", "citl", "slope")}
            a = np.asarray(per["auroc"], float)
            delta = a - ref
            st["models"][nm] = {"auroc": _summ(a), "brier": _summ(per["brier"]), "citl": _summ(per["citl"]),
                                "slope": _summ(per["slope"]), "abs_citl_mean": float(np.mean(np.abs(per["citl"]))),
                                "delta_mean": float(delta.mean()), "delta_lo": float(np.percentile(delta, 2.5)),
                                "delta_hi": float(np.percentile(delta, 97.5)), "share_better": float((delta > 0).mean()),
                                "folds_better": int((delta > 0).sum()), "fold_auroc": [round(float(x), 4) for x in a],
                                "kind": "learner" if nm in CANDIDATES else "comparison"}
        checks = [abs(fo["reference_check"] - ref[fo["fold"]]) for fo in folds]
        log(f"merge: nested CV, {len(names)} learners; reference refit vs saved fold AUROC: max difference "
            f"{max(checks):.4f}")
        nested["admission"] = nested["stages"]["admission"]
        nf.write_text(json.dumps(nested, indent=1))
