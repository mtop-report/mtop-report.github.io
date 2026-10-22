"""SHAP values of every validated model on the same test patients: what each model relies on,
comparable across models.

For each stage and outcome of a temporal split, the development model of every learner (the five of
train.py, the four amendment-1 candidates and the comparison-only learners of compare.py) is refitted
with its saved settings on the development years, checked against its saved test predictions, and
explained on the test patients on the log-odds scale:

  exact    regressions (coefficient x standardised value; spline terms and missing-value indicators
           belong to their predictor, an interaction product is split half and half; for multiple
           imputation, averaged over the imputations), LightGBM, XGBoost and CatBoost (TreeSHAP), EBM
           (its additive terms; a pair term split half and half), stacking (the meta-model's weighted
           sum of its three base models' values)
  sampled  MLP and probability averaging, which have no exact method: interventional Shapley values
           of the predictors by antithetic permutation sampling, with a fixed random sample of
           development patients as the background (the same sample for every model)
  gradient FT-Transformers: expected gradients (SHAP's GradientExplainer for differentiable models:
           the gradient of the log-odds along random paths from background patients to the patient,
           averaged), because permutation sampling of a transformer is too slow on a CPU; the same
           background sample

A predictor's value is the sum over its model columns (one-hot categories, missing-value indicators,
spline terms); in the sampled method a predictor's columns are switched together. Published
(artifacts/public/shap.json): mean |SHAP| per predictor over the test patients and its share of the
model's total. Aggregates only.

    python -m mtop shap [--split primary] [--stage admission] [--learner MLP]   chunks, then assembles
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import learners as Lr
from .cohort import split_temporal, splits, stage_cohort
from .config import settings, stage_ids, targets
from .export import clean_json
from .features import column_to_feature, design_matrix, monotone_vector
from .paths import PUBLIC

N_BACKGROUND, N_PERM, N_GRAD = 40, 10, 64
METHOD = {"exact_linear": "exact: the regression's terms", "exact_tree": "exact: TreeSHAP",
          "exact_ebm": "exact: the EBM's additive terms", "exact_stack": "exact: weighted sum of the base models' values",
          "sampled": f"sampled: {N_PERM} antithetic permutations x {N_BACKGROUND} background patients",
          "gradient": f"expected gradients: {N_GRAD} random paths from {N_BACKGROUND} background patients per patient"}


def out_dir(out_private: Path) -> Path:
    d = Path(out_private) / "shap"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------ the models again
class Model:
    """A refitted development model: probability, and exact per-column contributions when one exists
    (or `gradient`: expected-gradient attributions per model column)."""

    def __init__(self, proba, exact=None, method: str = "sampled", gradient=None):
        self.proba, self.exact, self.method, self.gradient = proba, exact, method, gradient


class FTModel:
    """FT-Transformer from its saved settings: the same preprocessing and training as compare.fit_ft's
    final fit (the epochs chosen there by early stopping)."""

    def __init__(self, X_dev: pd.DataFrame, y_dev, info: dict, seed: int):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        from . import compare as C

        self.C, self.columns = C, list(X_dev.columns)
        self.imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit(X_dev)
        self.sc = StandardScaler().fit(self.imp.transform(X_dev))
        arch = dict(C._ft_default()["arch"])
        arch.update({k: info[k] for k in arch if k in info})
        if "ffn_d_hidden" not in info:
            arch["ffn_d_hidden"] = None
        cfg = {"arch": arch, "lr": float(info["lr"]), "weight_decay": float(info["weight_decay"])}
        self.model, _, _ = C._ft_train(self.prep(X_dev), np.asarray(y_dev, float), cfg, seed, epochs=max(1, int(info["epochs"])))

    def prep(self, X) -> np.ndarray:
        return np.nan_to_num(self.sc.transform(self.imp.transform(X))).astype(np.float32)

    def proba(self, X) -> np.ndarray:
        return self.C._ft_predict(self.model, self.prep(X))

    def expected_gradients(self, X: pd.DataFrame, background: pd.DataFrame, n: int = N_GRAD, seed: int = 0):
        """(attributions per model column, base): the gradient of the log-odds at random points on the
        straight path from a random background patient to the patient, times the difference, averaged
        over `n` draws (Erion et al. 2021). A missing-value indicator's attribution goes to its column."""
        import torch

        self.model.eval()
        A, Bg = torch.from_numpy(self.prep(X)), torch.from_numpy(self.prep(background))
        g = torch.Generator().manual_seed(seed)
        phi = torch.zeros_like(A)
        for _ in range(n):
            b = Bg[torch.randint(len(Bg), (len(A),), generator=g)]
            z = (b + torch.rand(len(A), 1, generator=g) * (A - b)).requires_grad_(True)
            (grad,) = torch.autograd.grad(self.model(z, None)[:, 0].sum(), z)
            phi += (A - b) * grad
        phi = (phi / n).numpy().astype(float)
        with torch.no_grad():
            base = float(self.model(Bg, None)[:, 0].mean())
        p = len(self.columns)
        ind = [] if self.imp.indicator_ is None else [int(i) for i in self.imp.indicator_.features_]
        out = phi[:, :p].copy()
        for j, o in enumerate(ind):
            out[:, o] += phi[:, p + j]
        return out, base


def refit_models(names, tres: dict, X_dev: pd.DataFrame, y_dev, seed: int, mono, min_leaf: int, log=print) -> dict:
    from . import compare as C
    from . import extra_learners as XL
    from .tuning import threads

    settings_ = {**(tres.get("comparison_settings") or {}), **(tres.get("settings") or {})}
    out, stack = {}, None
    for name in names:
        t0 = time.time()
        if name in Lr.KIND:
            f = Lr.refit(name, settings_[name], X_dev, y_dev, seed, mono, min_leaf)
            kind = {"lgbm": "exact_tree", "xgb": "exact_tree", "ebm": "exact_ebm", "stack": "exact_stack",
                    "mlp": "sampled"}.get(f.kind, "exact_linear")
            out[name] = Model(f.proba, None if kind == "sampled" else (lambda X, f=f: Lr.contributions(f, X)), kind)
            if name == "Stacking":
                stack = f
        elif name == "EBM (defaults)":
            e = XL.fit_ebm(X_dev, y_dev, seed, threads=threads(), publishable=False)
            out[name] = Model(lambda X, e=e: e.predict_proba(X)[:, 1],
                              lambda X, e=e: XL.ebm_contributions(e, X, list(X.columns)), "exact_ebm")
        elif name == "CatBoost":
            from catboost import Pool

            cb = C.fit_catboost(X_dev, y_dev, seed)

            def cb_exact(X, cb=cb):
                v = cb.get_feature_importance(Pool(X), type="ShapValues")
                return pd.DataFrame(v[:, :-1], columns=X.columns, index=X.index), float(np.mean(v[:, -1]))
            out[name] = Model(lambda X, cb=cb: cb.predict_proba(X)[:, 1], cb_exact, "exact_tree")
        elif name == "Probability averaging":
            if stack is None:
                stack = Lr.refit("Stacking", settings_["Stacking"], X_dev, y_dev, seed, mono, min_leaf)
            bases = stack.model.bases
            out[name] = Model(lambda X, bases=bases: np.mean([b.proba(X) for b in bases], axis=0))
        elif name in ("FT-Transformer", "FT-Transformer, tuned"):
            ft = FTModel(X_dev, y_dev, settings_[name], seed)
            out[name] = Model(ft.proba, method="gradient", gradient=ft.expected_gradients)
        else:
            continue
        log(f"      {name} refitted ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------------------------- Shapley
def sampled_shap(proba, X: pd.DataFrame, background: pd.DataFrame, groups: list[list[int]], n_perm: int = N_PERM,
                 seed: int = 0) -> tuple[np.ndarray, float]:
    """Interventional Shapley values of column groups on the log-odds scale by antithetic permutation
    sampling: v(S) = mean over the background rows of logit f(x_S, b_rest). Each permutation's
    increments add up exactly to logit f(x) - v(empty set)."""
    rng = np.random.default_rng(seed)
    T, Bg = X.to_numpy(float), background.to_numpy(float)
    n, p = T.shape
    B, G = len(Bg), len(groups)
    cols = list(X.columns)

    def value(A):
        pr = np.clip(np.asarray(proba(pd.DataFrame(A, columns=cols)), float), 1e-9, 1 - 1e-9)
        return np.log(pr / (1 - pr))

    base = float(value(Bg).mean())
    orders = []
    for _ in range(max(1, n_perm // 2)):
        o = rng.permutation(G)
        orders += [o, o[::-1]]
    phi = np.zeros((n, G))
    for order in orders:
        cur = np.broadcast_to(Bg, (n, B, p)).copy()
        prev = np.full(n, base)
        for g in order:
            idx = groups[g]
            cur[:, :, idx] = T[:, None, idx]
            val = value(cur.reshape(n * B, p)).reshape(n, B).mean(axis=1)
            phi[:, g] += val - prev
            prev = val
    return phi / len(orders), base


def predictor_groups(columns: list[str], stage: str) -> tuple[list[str], list[list[int]]]:
    c2f = column_to_feature(stage)
    keys = list(dict.fromkeys(c2f[c] for c in columns))
    return keys, [[j for j, c in enumerate(columns) if c2f[c] == k] for k in keys]


def summarise(phi: np.ndarray, keys: list[str]) -> dict:
    imp = np.abs(phi).mean(axis=0)
    total = float(imp.sum()) or 1.0
    order = np.argsort(-imp)
    return {"importance": {keys[j]: round(float(imp[j]), 5) for j in order},
            "share": {keys[j]: round(100 * float(imp[j]) / total, 2) for j in order}}


# ------------------------------------------------------------------------------------ one split
def run(base: pd.DataFrame, out_private: Path, label: str = "primary", stages=None, names=None, log=print) -> None:
    from sklearn.metrics import roc_auc_score

    cfg = settings()
    seed, min_leaf = int(cfg["seed"]), int(cfg["privacy"]["min_leaf"])
    train = json.loads((Path(out_private) / f"train_{label}.json").read_text())
    saved = pd.read_parquet(Path(out_private) / f"predictions_{label}.parquet")
    test_from = splits()[label]
    for stage in stages or stage_ids():
        d = stage_cohort(base, stage)
        dev, test = split_temporal(d, test_from)
        X_all = design_matrix(d, stage)
        X_dev, X_test = X_all.loc[dev.index], X_all.loc[test.index]
        keys, groups = predictor_groups(list(X_all.columns), stage)
        bg = X_dev.sample(n=min(N_BACKGROUND, len(X_dev)), random_state=seed)
        for target in targets():
            t0 = time.time()
            tres = train["stages"][stage]["targets"][target]
            learned = [k for k, v in tres["models"].items() if v.get("kind") in ("learner", "comparison")]
            todo = [k for k in learned if not names or k in names]
            if not todo:
                continue
            y_dev, y_test = dev[target].to_numpy(float), test[target].to_numpy(float)
            log(f"  [{label} {stage}/{target}] {len(todo)} models, n_test={len(test)}")
            models = refit_models(todo, tres, X_dev, y_dev, seed, monotone_vector(stage, target), min_leaf, log)
            sv = saved[(saved["stage"] == stage) & (saved["target"] == target)]
            f = out_dir(out_private) / f"{label}_{stage}_{target}.json"
            res = json.loads(f.read_text()) if f.exists() else {"split": label, "stage": stage, "target": target, "models": {}}
            for name, m in models.items():
                ts = time.time()
                p = np.asarray(m.proba(X_test), float)
                g = sv[sv["model"] == name].set_index("patient_id")
                check = {"auroc": round(float(roc_auc_score(y_test, p)), 4)}
                if len(g):
                    p_saved = g.loc[test["patient_id"].to_numpy(), "p"].to_numpy(float)
                    check.update({"auroc_saved": round(float(roc_auc_score(y_test, p_saved)), 4),
                                  "max_abs_dp": round(float(np.max(np.abs(p - p_saved))), 5)})
                lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
                if m.exact is not None:
                    c, b0 = m.exact(X_test)
                    phi = np.column_stack([c.iloc[:, idx].sum(axis=1).to_numpy(float) for idx in groups])
                    check["additivity"] = round(float(np.max(np.abs(phi.sum(axis=1) + b0 - lp))), 4)
                elif m.gradient is not None:
                    c, b0 = m.gradient(X_test, bg, N_GRAD, seed)
                    phi = np.column_stack([c[:, idx].sum(axis=1) for idx in groups])
                    # expected gradients add up to log-odds minus the base only on average over draws
                    check["additivity"] = round(float(np.mean(np.abs(phi.sum(axis=1) + b0 - lp))), 4)
                else:
                    phi, b0 = sampled_shap(m.proba, X_test, bg, groups, N_PERM, seed)
                res["models"][name] = {"method": m.method, "n": int(len(X_test)), "base": round(float(b0), 4),
                                       "check": check, **summarise(phi, keys)}
                log(f"      {name}: {m.method}, check {check} ({time.time() - ts:.0f}s)")
            res["test_years"] = [int(test["year"].min()), int(test["year"].max())]
            f.write_text(json.dumps(clean_json(res), indent=1))
            log(f"  [{label} {stage}/{target}] done ({time.time() - t0:.0f}s)")


def assemble(out_private: Path) -> dict:
    """artifacts/public/shap.json from the chunks: per split, stage and outcome, every model's mean |SHAP|."""
    out = {"methods": METHOD, "n_background": N_BACKGROUND, "n_permutations": N_PERM, "n_gradient_paths": N_GRAD, "splits": {}}
    for f in sorted(out_dir(out_private).glob("*.json")):
        c = json.loads(f.read_text())
        sp = out["splits"].setdefault(c["split"], {"test_years": c.get("test_years"), "stages": {}})
        sp["stages"].setdefault(c["stage"], {})[c["target"]] = {
            k: {kk: v[kk] for kk in ("method", "n", "importance", "share", "check")} for k, v in c["models"].items()}
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "shap.json").write_text(json.dumps(clean_json(out), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out
