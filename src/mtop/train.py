"""Model development and validation.

For every stage x target:
  1. Tune LightGBM (monotone constraints, min leaf size) on the development cohort
     (onset <= 2020) with Optuna, minimising 5-fold cross-validated log-loss.
  2. Internal validation: repeated stratified CV on the development cohort.
  3. Temporal validation on 2021+ patients for LightGBM and comparators
     (logistic regression, XGBoost, MLP) and published scores; bootstrap 95% CIs.
  4. Recalibration analysis: does re-estimating the intercept on 2021-22 fix calibration
     in 2023+?
  5. Deployment model: refit on all years, then intercept-recalibrate to the recent era
     using cross-validated (out-of-fold) predictions.
"""
from __future__ import annotations

import json
import logging
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .cohort import split_temporal, stage_cohort
from .config import settings, stage_ids, targets
from .features import column_to_feature, design_matrix, monotone_vector
from .scores import benchmark_scores

log = logging.getLogger("mtop.train")
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_PARAMS = dict(objective="binary", verbosity=-1, deterministic=True, force_row_wise=True,
                   num_threads=2, bagging_freq=1, monotone_constraints_method="advanced")


MIN_LEAF = 20


def _params(p: dict, mono: list[int], seed: int) -> dict:
    out = dict(BASE_PARAMS)
    out.update(p)
    out["seed"] = seed
    out["monotone_constraints"] = mono
    # LightGBM checks min_child_samples on hessian-estimated counts, so real leaves can be
    # smaller. For log-loss each patient adds at most 0.25 to a leaf's hessian sum, so a
    # hessian floor of MIN_LEAF / 4 guarantees >= MIN_LEAF patients in every leaf.
    out["min_sum_hessian_in_leaf"] = MIN_LEAF / 4.0
    return out


def tune_lgbm(X, y, mono, seed, trials, max_rounds, es, min_leaf):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))

    def objective(trial):
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 15),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", min_leaf, 80),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        }
        ds = lgb.Dataset(X, y, params={"feature_pre_filter": False})
        cv = lgb.cv(_params(p, mono, seed), ds, num_boost_round=max_rounds, folds=folds,
                    callbacks=[lgb.early_stopping(es, verbose=False)])
        key = "valid binary_logloss-mean"
        trial.set_user_attr("rounds", len(cv[key]))
        return cv[key][-1]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    return best, int(study.best_trial.user_attrs["rounds"]), float(study.best_value)


def cv_rounds(X, y, params, seed, max_rounds, es):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    cv = lgb.cv(params, lgb.Dataset(X, y, params={"feature_pre_filter": False}), num_boost_round=max_rounds, folds=folds,
                callbacks=[lgb.early_stopping(es, verbose=False)])
    return len(cv["valid binary_logloss-mean"])


def fit_lgbm(X, y, params, rounds):
    return lgb.train(params, lgb.Dataset(X, y, params={"feature_pre_filter": False}), num_boost_round=rounds)


def oof_lgbm(X, y, params, rounds, seed, folds=5, repeats=1):
    """Out-of-fold predicted probabilities (averaged over repeats) and fold AUROCs."""
    pred = np.zeros(len(y))
    cnt = np.zeros(len(y))
    aucs = []
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    for tr, va in rskf.split(X, y):
        m = fit_lgbm(X.iloc[tr], y[tr], params, rounds)
        p = m.predict(X.iloc[va])
        pred[va] += p
        cnt[va] += 1
        aucs.append(M.roc_auc_score(y[va], p))
    return pred / cnt, aucs


# ------------------------------------------------------------------------------- comparators
def fit_logreg(X, y, seed):
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegressionCV(Cs=np.logspace(-3, 1, 12), cv=5, scoring="neg_log_loss", l1_ratios=(0.0,),
                             use_legacy_attributes=False, max_iter=5000, random_state=seed),
    )
    return model.fit(X, y)


def fit_mlp(X, y, seed):
    """MLP comparator: small grid search, then a 5-seed average of the best setting (stabler)."""
    from sklearn.base import clone

    base = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        MLPClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.2,
                      n_iter_no_change=25, random_state=seed),
    )
    grid = {"mlpclassifier__hidden_layer_sizes": [(16,), (32, 16)],
            "mlpclassifier__alpha": [1e-2, 1e-1, 1.0],
            "mlpclassifier__learning_rate_init": [1e-3]}
    gs = GridSearchCV(base, grid, scoring="neg_log_loss", n_jobs=2,
                      cv=StratifiedKFold(5, shuffle=True, random_state=seed)).fit(X, y)
    members = [clone(base).set_params(**gs.best_params_, mlpclassifier__random_state=seed + k).fit(X, y)
               for k in range(5)]

    class _Ensemble:
        def predict_proba(self, Xn):
            return np.mean([m.predict_proba(Xn) for m in members], axis=0)

    return _Ensemble()


def fit_xgb(X, y, seed):
    import xgboost as xgb

    d = xgb.DMatrix(X, label=y, missing=np.nan)
    best = None
    for md in (2, 3, 4):
        for mcw in (1, 5, 10):
            p = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.03,
                 "max_depth": md, "min_child_weight": mcw, "subsample": 0.8,
                 "colsample_bytree": 0.8, "lambda": 1.0, "seed": seed, "nthread": 2}
            cv = xgb.cv(p, d, num_boost_round=3000, nfold=5, stratified=True, seed=seed,
                        early_stopping_rounds=100, verbose_eval=False)
            score = cv["test-logloss-mean"].iloc[-1]
            if best is None or score < best[0]:
                best = (score, p, len(cv))
    booster = xgb.train(best[1], d, num_boost_round=best[2])

    class _Wrap:
        def __init__(self, b):
            self.b = b

        def predict_proba(self, Xn):
            p = self.b.predict(xgb.DMatrix(Xn, missing=np.nan))
            return np.column_stack([1 - p, p])

    return _Wrap(booster)


def _score_to_prob(train_score, y_train, test_score):
    """Map a benchmark score to probabilities with a logistic fit on the development cohort."""
    s_tr = np.asarray(train_score, float)
    keep = ~np.isnan(s_tr)
    a, b = M._fit_logistic(s_tr[keep], y_train[keep])
    s_te = np.asarray(test_score, float)
    fill = np.nanmedian(s_tr)
    s_te = np.where(np.isnan(s_te), fill, s_te)
    return M.expit(a + b * s_te)


# ------------------------------------------------------------------------------- explanations
def shap_summary(model, X: pd.DataFrame, stage: str, min_cell: int):
    """Global SHAP importance per predictor and binned dependence (aggregates only)."""
    contrib = model.predict(X, pred_contrib=True)
    phi = pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)
    c2f = column_to_feature(stage)
    grouped = phi.T.groupby(lambda c: c2f[c]).sum().T
    importance = grouped.abs().mean().sort_values(ascending=False)
    dependence = {}
    for key in importance.index:
        if key not in X.columns:
            continue  # categoricals: summarised per category below
        x = X[key]
        ok = x.notna()
        if x[ok].nunique() <= 8:
            groups = x[ok].astype(float)
        else:
            try:
                groups = pd.qcut(x[ok], q=10, duplicates="drop")
            except ValueError:
                continue
        rows = []
        for g, idx in groups.groupby(groups, observed=True).groups.items():
            if len(idx) < min_cell:
                continue
            xs = x.loc[idx]
            rows.append({"x": float(xs.median()), "shap": float(grouped.loc[idx, key].mean()), "n": int(len(idx))})
        dependence[key] = rows
    for f in {c2f[c] for c in X.columns if "__" in c}:
        cols = [c for c in X.columns if c2f[c] == f]
        rows = []
        for c in cols:
            idx = X.index[X[c] == 1]
            if len(idx) >= min_cell:
                rows.append({"category": c.split("__", 1)[1], "shap": float(grouped.loc[idx, f].mean()),
                             "n": int(len(idx))})
        dependence[f] = rows
    return {"importance": {k: float(v) for k, v in importance.items()}, "dependence": dependence,
            "base_value": float(contrib[:, -1].mean())}


# ------------------------------------------------------------------------------- main routine
def run(base: pd.DataFrame, out_private, quick: bool = False) -> dict:
    cfg = settings()
    seed = int(cfg["seed"])
    tcfg = cfg["tuning"]
    trials = 8 if quick else int(tcfg["trials"])
    n_boot = 200 if quick else int(cfg["bootstrap"]["n"])
    min_leaf = int(cfg["privacy"]["min_leaf"])
    global MIN_LEAF
    MIN_LEAF = min_leaf
    min_cell = int(cfg["privacy"]["min_cell"])
    recent_from = int(cfg["split"]["recent_era_from_year"])
    test_from = int(cfg["split"]["temporal_test_from_year"])

    results = {"stages": {}, "meta": {"seed": seed, "test_from_year": test_from,
                                       "recent_era_from_year": recent_from, "quick": quick}}
    models = {}
    pred_rows = []
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        dev, test = split_temporal(d)
        X_all = design_matrix(d, stage)
        X_dev, X_test = X_all.loc[dev.index], X_all.loc[test.index]
        results["stages"][stage] = {"n_all": int(len(d)), "n_dev": int(len(dev)), "n_test": int(len(test)),
                                    "columns": list(X_all.columns), "targets": {}}
        for target in targets():
            t0 = time.time()
            y_all = d[target].to_numpy(float)
            y_dev, y_test = dev[target].to_numpy(float), test[target].to_numpy(float)
            mono = monotone_vector(stage, target)

            best, rounds, cv_ll = tune_lgbm(X_dev, y_dev, mono, seed, trials, tcfg["max_rounds"],
                                            tcfg["early_stopping"], min_leaf)
            params = _params(best, mono, seed)

            # internal validation on the development cohort
            oof_dev, fold_aucs = oof_lgbm(X_dev, y_dev, params, rounds, seed,
                                          folds=cfg["cv"]["folds"], repeats=1 if quick else cfg["cv"]["repeats"])
            internal = M.summary(y_dev, oof_dev)
            internal["fold_auroc_mean"] = float(np.mean(fold_aucs))
            internal["fold_auroc_sd"] = float(np.std(fold_aucs))

            # temporal validation
            m_dev = fit_lgbm(X_dev, y_dev, params, rounds)
            preds = {"LightGBM": m_dev.predict(X_test)}
            for name, fit in (("Logistic regression", fit_logreg), ("XGBoost", fit_xgb), ("MLP", fit_mlp)):
                mdl = fit(X_dev, y_dev, seed)
                preds[name] = mdl.predict_proba(X_test)[:, 1]
            bench_dev = benchmark_scores(dev, stage, target)
            bench_test = benchmark_scores(test, stage, target)
            for name in bench_dev:
                preds[name] = _score_to_prob(bench_dev[name], y_dev, bench_test[name])
            temporal = {k: M.summary(y_test, v) for k, v in preds.items()}
            ci = M.bootstrap_ci(y_test, preds, n_boot=n_boot, seed=seed, ref="LightGBM")
            for k in temporal:
                temporal[k]["ci"] = ci[k]

            curves = {
                "roc": {k: M.roc_points(y_test, v) for k, v in preds.items()},
                "calibration": M.calibration_bins(y_test, preds["LightGBM"], n_bins=8 if len(y_test) < 250 else 10,
                                                  min_n=min_cell),
                "dca": M.decision_curve(y_test, preds["LightGBM"]),
            }

            # recalibration analysis inside the temporal window
            recal = None
            early = (test["year"] < test_from + 2).to_numpy()
            late = ~early
            if early.sum() >= 40 and late.sum() >= 40 and len(np.unique(y_test[early])) == 2:
                p_early, p_late = preds["LightGBM"][early], preds["LightGBM"][late]
                a = M._fit_logistic(np.zeros(early.sum()), y_test[early], offset=M.logit(p_early), slope=False)[0]
                p_late_recal = M.expit(M.logit(p_late) + a)
                recal = {"recal_years": f"{test_from}-{test_from + 1}", "eval_years": f"{test_from + 2}+",
                         "intercept_shift": float(a), "n_recal": int(early.sum()), "n_eval": int(late.sum()),
                         "before": M.summary(y_test[late], p_late), "after": M.summary(y_test[late], p_late_recal)}

            # deployment model: all years, then intercept recalibration to the recent era
            rounds_all = cv_rounds(X_all, y_all, params, seed, tcfg["max_rounds"], tcfg["early_stopping"])
            m_all = fit_lgbm(X_all, y_all, params, rounds_all)
            oof_all, _ = oof_lgbm(X_all, y_all, params, rounds_all, seed + 1, folds=5, repeats=1)
            recent = (d["year"] >= recent_from).to_numpy()
            shift = float(M._fit_logistic(np.zeros(recent.sum()), y_all[recent],
                                          offset=M.logit(oof_all[recent]), slope=False)[0])
            p_recent = M.expit(M.logit(oof_all[recent]) + shift)
            deployed = {
                "reality_bins": M.calibration_bins(y_all[recent], p_recent,
                                                   n_bins=int(max(4, min(10, recent.sum() // 25))), min_n=min_cell),
                "n_recent": int(recent.sum()),
                "events_recent": int(y_all[recent].sum()),
                "rounds": int(rounds_all),
                "intercept_recent": shift,
                "oof_all": M.summary(y_all, oof_all),
                "oof_recent_uncalibrated": M.summary(y_all[recent], oof_all[recent]),
                "oof_recent_recalibrated": M.summary(y_all[recent], M.expit(M.logit(oof_all[recent]) + shift)),
                "prevalence_recent": float(y_all[recent].mean()),
                "prevalence_all": float(y_all.mean()),
            }
            explain = shap_summary(m_all, X_all, stage, min_cell)

            results["stages"][stage]["targets"][target] = {
                "params": best, "rounds_dev": rounds, "tuning_cv_logloss": cv_ll,
                "internal": internal, "temporal": temporal, "curves": curves,
                "recalibration": recal, "deployed": deployed, "explain": explain,
                "seconds": round(time.time() - t0, 1),
            }
            models[(stage, target)] = {"booster": m_all, "columns": list(X_all.columns),
                                       "intercept_recent": shift}
            m_all.save_model(str(out_private / f"lgbm_{stage}_{target}.txt"))
            for k, v in preds.items():
                pred_rows.append(pd.DataFrame({"patient_id": test["patient_id"].to_numpy(), "stage": stage,
                                               "target": target, "model": k, "y": y_test, "p": v}))
            log.info("%s/%s: temporal AUROC %.3f (%.0fs)", stage, target,
                     temporal["LightGBM"]["auroc"], time.time() - t0)
            print(f"[{stage}/{target}] n_dev={len(dev)} n_test={len(test)} rounds={rounds} "
                  f"internal AUROC={internal['auroc']:.3f} temporal AUROC={temporal['LightGBM']['auroc']:.3f} "
                  f"citl={temporal['LightGBM']['citl']:.2f} slope={temporal['LightGBM']['slope']:.2f} "
                  f"recent shift={shift:.2f} ({time.time() - t0:.0f}s)", flush=True)

    pd.concat(pred_rows).to_parquet(out_private / "temporal_predictions.parquet", index=False)
    with open(out_private / "train_results.json", "w") as f:
        json.dump(results, f, indent=1)
    return {"results": results, "models": models}
