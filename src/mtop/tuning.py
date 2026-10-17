"""Equal-budget tuning for the comparison learners.

Every learner gets the same search as LightGBM: 50 candidate settings (`tuning.trials` in
config/settings.yaml), each scored by 5-fold stratified cross-validated log-loss on the training
patients, with the same folds (StratifiedKFold(5, shuffle=True, random_state=seed)).

  * LightGBM (train.tune_lgbm), XGBoost and the MLP: Bayesian optimisation (Optuna TPE, seeded)
    over six or seven settings each; boosted trees stop adding trees after 100 rounds without
    improvement.
  * Logistic regression: its only setting, the L2 penalty strength, over 50 values on a log grid
    (lr.fit_logreg); the spline version also chooses 3, 4 or 5 knots (splines.fit_spline_logreg).

Each fitted model carries `.settings` (the chosen values and the cross-validated log-loss), which
train.py stores, export.py publishes and the model card reports.
"""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

MAX_ROUNDS, EARLY_STOP = 3000, 100


def shared_folds(y, seed: int) -> list:
    """The cross-validation folds every learner is tuned on."""
    y = np.asarray(y)
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _study(seed: int):
    return optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))


# ------------------------------------------------------------------------------------- XGBoost
def tune_xgb(X, y, seed: int, trials: int = 50):
    import xgboost as xgb

    y = np.asarray(y, float)
    folds = shared_folds(y, seed)
    d = xgb.DMatrix(X, label=y, missing=np.nan)
    base = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": seed, "nthread": 2}

    def params(get):
        return dict(base,
                    eta=get("learning_rate", 0.01, 0.08, True),
                    max_depth=int(get("max_depth", 2, 5, None)),
                    min_child_weight=get("min_child_weight", 1.0, 20.0, True),
                    subsample=get("subsample", 0.6, 1.0, False),
                    colsample_bytree=get("colsample_bytree", 0.5, 1.0, False),
                    gamma=get("gamma", 1e-3, 5.0, True),
                    **{"lambda": get("lambda", 1e-3, 30.0, True)})

    def objective(trial):
        def get(name, lo, hi, log):
            return trial.suggest_int(name, lo, hi) if log is None else trial.suggest_float(name, lo, hi, log=log)
        cv = xgb.cv(params(get), d, num_boost_round=MAX_ROUNDS, folds=folds, early_stopping_rounds=EARLY_STOP,
                    verbose_eval=False)
        trial.set_user_attr("trees", int(len(cv)))
        return float(cv["test-logloss-mean"].iloc[-1])

    study = _study(seed)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_params
    p = params(lambda name, lo, hi, log: best[name])
    trees = int(study.best_trial.user_attrs["trees"])
    booster = xgb.train(p, d, num_boost_round=trees)

    class XGBModel:
        settings = {"learning_rate": best["learning_rate"], "max_depth": best["max_depth"],
                    "min_child_weight": best["min_child_weight"], "subsample": best["subsample"],
                    "colsample_bytree": best["colsample_bytree"], "lambda": best["lambda"], "gamma": best["gamma"],
                    "trees": trees, "cv_logloss": float(study.best_value), "trials": trials}

        def predict_proba(self, Xn):
            q = booster.predict(xgb.DMatrix(Xn, missing=np.nan))
            return np.column_stack([1 - q, q])

    return XGBModel()


# ----------------------------------------------------------------------------------------- MLP
def _mlp(seed: int, layers: tuple, alpha: float, lr: float, activation: str, batch: int):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=layers, alpha=alpha, learning_rate_init=lr, activation=activation,
                      batch_size=batch, max_iter=1000, early_stopping=True, validation_fraction=0.2,
                      n_iter_no_change=25, random_state=seed),
    )


def _layers(n: int, units: int) -> tuple:
    """A funnel: the first layer has `units` neurons, each further layer half as many (at least 4)."""
    return tuple(max(4, units // (2 ** i)) for i in range(n))


def tune_mlp(X, y, seed: int, trials: int = 50, ensemble: int = 5):
    """MLP (scikit-learn): Optuna over depth, width, weight decay, learning rate, activation and
    batch size; the best setting is then trained from `ensemble` random starts and averaged."""
    y = np.asarray(y, float)
    folds = shared_folds(y, seed)

    def draw(trial) -> dict:
        n = trial.suggest_int("layers", 1, 3)
        units = trial.suggest_int("units", 8, 128, log=True)
        return {"layers": _layers(n, units),
                "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
                "lr": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "batch": trial.suggest_categorical("batch_size", [32, 64, 128, 256])}

    def objective(trial):
        s = cross_val_score(_mlp(seed, **draw(trial)), X, y, cv=folds, scoring="neg_log_loss", n_jobs=2)
        v = -float(np.mean(s))
        return v if np.isfinite(v) else 10.0

    study = _study(seed)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    b = study.best_params
    cfg = {"layers": _layers(b["layers"], b["units"]), "alpha": b["alpha"], "lr": b["learning_rate"],
           "activation": b["activation"], "batch": b["batch_size"]}
    template = _mlp(seed, **cfg)
    members = [clone(template).set_params(mlpclassifier__random_state=seed + k).fit(X, y) for k in range(ensemble)]

    class MLPModel:
        settings = {"hidden_layers": list(cfg["layers"]), "alpha": cfg["alpha"], "learning_rate": cfg["lr"],
                    "activation": cfg["activation"], "batch_size": cfg["batch"], "ensemble": ensemble,
                    "cv_logloss": float(study.best_value), "trials": trials}

        def predict_proba(self, Xn):
            return np.mean([m.predict_proba(Xn) for m in members], axis=0)

    return MLPModel()


# ------------------------------------------------------------------------------ all comparators
LEARNERS = ("Logistic regression", "Logistic regression (splines)", "XGBoost", "MLP")


def comparator_models(X, y, seed: int, trials: int = 50) -> dict:
    """The four comparison learners, each tuned with the same budget on the same folds."""
    from .lr import fit_logreg
    from .splines import fit_spline_logreg

    return {"Logistic regression": fit_logreg(X, y, seed),
            "Logistic regression (splines)": fit_spline_logreg(X, y, seed),
            "XGBoost": tune_xgb(X, y, seed, trials),
            "MLP": tune_mlp(X, y, seed, trials)}


def lgbm_settings(params: dict, trees: int, cv_logloss: float, trials: int) -> dict:
    return {**params, "trees": int(trees), "cv_logloss": float(cv_logloss), "trials": int(trials)}


def jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 6)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out
