"""LightGBM: tuning, fitting and out-of-fold predictions (monotone constraints, >= 20 patients per leaf)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from . import metrics as M
from .tuning import threads

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_PARAMS = dict(objective="binary", verbosity=-1, deterministic=True, force_row_wise=True,
                   bagging_freq=1, monotone_constraints_method="advanced")
MIN_LEAF = 20


def params(p: dict, mono: list[int], seed: int, min_leaf: int = MIN_LEAF) -> dict:
    out = dict(BASE_PARAMS)
    out["num_threads"] = threads()
    out.update({k: v for k, v in p.items() if k not in ("trees", "cv_logloss", "trials")})
    out["seed"] = seed
    out["monotone_constraints"] = mono
    # LightGBM checks min_child_samples on hessian-estimated counts, so real leaves can be
    # smaller. For log-loss each patient adds at most 0.25 to a leaf's hessian sum, so a
    # hessian floor of min_leaf / 4 guarantees >= min_leaf patients in every leaf.
    out["min_sum_hessian_in_leaf"] = min_leaf / 4.0
    return out


def _dataset(X, y):
    return lgb.Dataset(X, y, params={"feature_pre_filter": False})


def tune(X, y, mono, seed, trials, max_rounds=3000, es=100, min_leaf=MIN_LEAF):
    """Optuna TPE over seven settings, scored by 5-fold CV log-loss (the folds every learner uses)."""
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
        cv = lgb.cv(params(p, mono, seed, min_leaf), _dataset(X, y), num_boost_round=max_rounds, folds=folds,
                    callbacks=[lgb.early_stopping(es, verbose=False)])
        key = "valid binary_logloss-mean"
        trial.set_user_attr("rounds", len(cv[key]))
        return cv[key][-1]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params), int(study.best_trial.user_attrs["rounds"]), float(study.best_value)


def cv_rounds(X, y, prm, seed, max_rounds=3000, es=100):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
    cv = lgb.cv(prm, _dataset(X, y), num_boost_round=max_rounds, folds=folds,
                callbacks=[lgb.early_stopping(es, verbose=False)])
    return len(cv["valid binary_logloss-mean"])


def fit(X, y, prm, rounds):
    return lgb.train(prm, _dataset(X, y), num_boost_round=int(rounds))


def oof(X, y, prm, rounds, seed, folds=5, repeats=1):
    """Out-of-fold predicted probabilities (averaged over repeats) and fold AUROCs."""
    y = np.asarray(y, float)
    pred = np.zeros(len(y))
    cnt = np.zeros(len(y))
    aucs = []
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    for tr, va in rskf.split(X, y):
        m = fit(X.iloc[tr], y[tr], prm, rounds)
        p = m.predict(X.iloc[va])
        pred[va] += p
        cnt[va] += 1
        aucs.append(M.roc_auc_score(y[va], p))
    return pred / cnt, aucs
