"""The coordinator: opens jobs at the centres, runs the rounds and adds up the masked answers.

It never receives patient rows and cannot read any single centre's answer; it only learns the
network-wide sums it asked for. The same code drives centres in this process (`LocalTransport`,
used for experiments and tests) and centres connected over HTTP (`server.HubTransport`).
`PooledNetwork` runs the identical algorithm on pooled data; it exists only to prove that the
federated result is the same.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import PROTOCOL
from .bins import Bands, fed_settings
from .gbdt import GBDTParams, Network
from .gbdt import train as gbdt_train
from .logreg import newton, spline_design, spline_spec, unpack_terms
from .schema import template_columns
from .secagg import aggregate, from_fixed, unpack
from .site import compute, eval_bins, mono_for


class JobFailed(RuntimeError):
    pass


def _bits() -> int:
    return int(fed_settings()["network"]["fixed_point_bits"])


class LocalTransport:
    """Centres in this process. Every message goes through JSON both ways, as it would over HTTP."""

    def __init__(self, sites):
        self.sites = {s.site_id: s for s in sites}
        self.stats = {sid: {"down": 0, "up": 0, "messages": 0} for sid in self.sites}

    def ids(self) -> list[str]:
        return list(self.sites)

    def exchange(self, sids, msg: dict) -> dict:
        raw = json.dumps(msg, separators=(",", ":"))
        out = {}
        for sid in sids:
            rep = json.dumps(self.sites[sid].handle(json.loads(raw)), separators=(",", ":"))
            st = self.stats[sid]
            st["down"] += len(raw)
            st["up"] += len(rep)
            st["messages"] += 1
            out[sid] = json.loads(rep)
        return out


class Coordinator:
    def __init__(self, transport, log=None):
        self.t = transport
        self.min_sites = int(fed_settings()["network"]["min_sites"])
        self.bits = _bits()
        self.log = log or (lambda *a, **k: None)
        self.jobs: list[dict] = []
        self.sample: dict | None = None

    def hello(self, sids=None) -> dict:
        return self.t.exchange(sids or self.t.ids(), {"type": "hello"})

    def open(self, kind: str, spec: dict, sites: list[str], budget: int) -> "Job":
        sites = list(sites)
        if len(set(sites)) < self.min_sites:
            raise JobFailed(f"a job needs at least {self.min_sites} centres")
        job_id = f"{kind}-{uuid.uuid4().hex[:10]}"
        offer = {"type": "offer", "job": job_id, "kind": kind, "spec": spec, "participants": sites,
                 "protocol": PROTOCOL, "budget": int(budget)}
        rep = self.t.exchange(sites, offer)
        refused = {s: r.get("error", "declined") for s, r in rep.items() if not r.get("accept")}
        if refused:
            self.t.exchange([s for s in sites if s not in refused], {"type": "abort", "job": job_id,
                                                                     "reason": "another centre declined"})
            raise JobFailed(f"declined by {refused}")
        peers = {s: {k: rep[s][k] for k in ("x25519", "sig", "identity")} for s in sites}
        st = self.t.exchange(sites, {"type": "start", "job": job_id, "peers": peers})
        bad = {s: r.get("error") for s, r in st.items() if not r.get("ok")}
        if bad:
            self.t.exchange(sites, {"type": "abort", "job": job_id, "reason": "key agreement failed"})
            raise JobFailed(f"key agreement failed at {bad}")
        self.log(f"job {job_id}: {kind} with {len(sites)} centres")
        return Job(self, job_id, kind, sites)


class Job(Network):
    def __init__(self, coord: Coordinator, job_id: str, kind: str, sites: list[str]):
        self.c, self.id, self.kind, self.sites = coord, job_id, kind, sites
        self.round = 0
        self.started = time.time()
        self.rounds: list[dict] = []

    def sum(self, op: str, args: dict) -> np.ndarray:
        self.round += 1
        t0 = time.time()
        rep = self.c.t.exchange(self.sites, {"type": "round", "job": self.id, "round": self.round, "op": op, "args": args})
        failed = {s: rep[s].get("error", "no answer") for s in self.sites if "vec" not in rep.get(s, {})}
        if failed:
            self.abort(f"round {self.round} failed")
            raise JobFailed(f"round {self.round} ({op}) failed: {failed}")
        vecs = [unpack(rep[s]["vec"]) for s in self.sites]
        if len({len(v) for v in vecs}) != 1:
            self.abort("answers of different length")
            raise JobFailed("answers of different length")
        total = aggregate(vecs)
        if self.c.sample is None and op == "init":
            self.c.sample = {"op": op, "site": self.sites[0], "masked": [f"{int(v):016x}" for v in vecs[0][:2]],
                             "total": [int(v) for v in total[:2]]}
        self.rounds.append({"op": op, "values": int(len(total)), "seconds": time.time() - t0})
        return total

    def finish(self, result=None) -> None:
        self.c.t.exchange(self.sites, {"type": "finish", "job": self.id, "result": result})
        self.c.jobs.append({"job": self.id, "kind": self.kind, "sites": len(self.sites), "rounds": self.round,
                            "seconds": time.time() - self.started})

    def abort(self, reason: str) -> None:
        try:
            self.c.t.exchange(self.sites, {"type": "abort", "job": self.id, "reason": reason})
        except Exception:  # noqa: BLE001 - best effort
            pass


class PooledNetwork(Network):
    """Reference only: the same computations on pooled rows (what centralising the data would give)."""

    def __init__(self, parts, stage: str, target: str, spec: dict | None = None):
        self.d = pd.concat([p[0] for p in parts], ignore_index=True)
        self.B = np.vstack([p[1] for p in parts])
        self.y = np.concatenate([p[2] for p in parts])
        self.job = {"stage": stage, "target": target, "F": None, "n_trees": 0, "spec": spec or {}}
        self.bits = _bits()

    @classmethod
    def from_sites(cls, sites, stage: str, target: str, subset: dict | None = None, spec: dict | None = None):
        parts = []
        for s in sites:
            d, B, y = s.cohort(stage, target)
            r = s.rows(stage, target, subset)
            parts.append((d.iloc[r], B[r], y[r]))
        return cls(parts, stage, target, spec)

    def sum(self, op: str, args: dict) -> np.ndarray:
        return compute(op, args, self.job, self.d, self.B, self.y, self.bits)


# --------------------------------------------------------------------------- jobs
def gbdt_spec(stage: str, target: str, subset: dict | None, p: GBDTParams) -> dict:
    return {"stage": stage, "target": target, "subset": subset or {"kind": "all"}, "params": asdict(p)}


def fit_gbdt(net: Network, stage: str, target: str, p: GBDTParams, on_tree=None, start: dict | None = None) -> dict:
    model = gbdt_train(net, Bands.for_stage(stage), mono_for(stage, target), p, on_tree, start)
    model.update(stage=stage, target=target)
    return model


def federated_gbdt(coord: Coordinator, sites: list[str], stage: str, target: str, subset: dict | None = None,
                   params: GBDTParams | None = None, on_tree=None) -> tuple[dict, Job]:
    p = params or GBDTParams.default()
    job = coord.open("gbdt", gbdt_spec(stage, target, subset, p), sites, p.rounds_budget())
    model = fit_gbdt(job, stage, target, p, on_tree)
    model["network"] = {"centres": len(sites), "rounds": job.round, "protocol": PROTOCOL}
    job.finish({"model": model})
    return model, job


def fit_logreg(net: Network, design: dict, lam: float | None = None) -> dict:
    lr = fed_settings()["logreg"]
    lam = float(lr["lambda"] if lam is None else lam)
    bits = _bits()
    mom = net.sum("moments", {"design": design})
    n = float(mom[0])
    vals = from_fixed(mom[1:], bits)
    d0 = len(vals) // 2
    mean = vals[:d0] / n
    sd = np.sqrt(np.clip(vals[d0:] / n - mean ** 2, 0.0, None))
    sd[sd < 1e-9] = 1.0

    def step(beta):
        v = from_fixed(net.sum("lr", {"design": design, "mean": mean.tolist(), "sd": sd.tolist(), "beta": beta.tolist()}), bits)
        return unpack_terms(v, d0 + 1)

    beta, trace, it = newton(step, d0 + 1, lam, np.r_[0.0, np.ones(d0)], int(lr["iterations"]), float(lr["tolerance"]))
    empty = pd.DataFrame({c: pd.Series(dtype=float) for c in template_columns()})
    names = spline_design(empty, design)[1]
    return {"kind": "logreg", "design": design, "names": names, "mean": mean.tolist(), "sd": sd.tolist(),
            "beta": beta.tolist(), "lambda": lam, "iterations": it, "penalised_loglik": trace}


def federated_logreg(coord: Coordinator, sites: list[str], stage: str, target: str, subset: dict | None = None,
                     lam: float | None = None) -> tuple[dict, Job]:
    design = spline_spec(stage)
    spec = {"stage": stage, "target": target, "subset": subset or {"kind": "all"}, "design": design}
    job = coord.open("logreg", spec, sites, 1 + int(fed_settings()["logreg"]["iterations"]))
    model = fit_logreg(job, design, lam)
    model.update(stage=stage, target=target)
    job.finish({"model": model})
    return model, job


def evaluate(net: Network, calibration: bool = True) -> dict:
    """Pooled performance from aggregated counts, plus calibration fitted by federated Newton steps."""
    bits = _bits()
    out = eval_metrics(net.sum("eval", {}), bits)
    if calibration:
        def citl(beta):
            return unpack_terms(from_fixed(net.sum("lr", {"design": {"kind": "citl"}, "beta": beta.tolist()}), bits), 1)

        def slope(beta):
            return unpack_terms(from_fixed(net.sum("lr", {"design": {"kind": "recal"}, "beta": beta.tolist()}), bits), 2)

        a, _, _ = newton(citl, 1, 0.0, np.zeros(1), 25, 1e-8)
        ab, _, _ = newton(slope, 2, 0.0, np.zeros(2), 25, 1e-8, beta0=[0.0, 1.0])
        out.update(citl=float(a[0]), slope=float(ab[1]))
    return out


def federated_eval(coord: Coordinator, sites: list[str], model: dict, subset: dict | None = None,
                   calibration: bool = True) -> tuple[dict, Job]:
    spec = {"stage": model["stage"], "target": model["target"], "subset": subset or {"kind": "all"}, "model": model}
    job = coord.open("eval", spec, sites, 51)
    res = evaluate(job, calibration)
    job.finish({"metrics": res})
    return res, job


def eval_metrics(vec: np.ndarray, bits: int, min_group: int | None = None) -> dict:
    nb, lo, hi = eval_bins()
    pos, neg = vec[:nb].astype(float), vec[nb:2 * nb].astype(float)
    psum = from_fixed(vec[2 * nb:3 * nb], bits)
    n, events = int(vec[3 * nb]), int(vec[3 * nb + 1])
    s_p, s_b, s_ll = from_fixed(vec[3 * nb + 2:3 * nb + 5], bits)
    P, N = pos.sum(), neg.sum()
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    auc = float((pos * (below + 0.5 * neg)).sum() / (P * N)) if P and N else float("nan")
    # calibration groups of about a tenth of the patients (never fewer than min_cell)
    from ..config import settings

    k = int(min_group or settings()["privacy"]["min_cell"])
    target = max(k, int(np.ceil(n / 10)))
    groups, acc = [], [0.0, 0.0, 0.0]
    cnt = pos + neg
    for b in range(nb):
        acc[0] += cnt[b]
        acc[1] += pos[b]
        acc[2] += psum[b]
        if acc[0] >= target:
            groups.append(acc)
            acc = [0.0, 0.0, 0.0]
    if acc[0] > 0:
        if groups and acc[0] < k:
            groups[-1] = [a + b for a, b in zip(groups[-1], acc)]
        else:
            groups.append(acc)
    cal = [{"n": int(g[0]), "pred": g[2] / g[0], "obs": g[1] / g[0]} for g in groups if g[0] > 0]
    return {"n": n, "events": events, "auroc": auc, "brier": s_b / n, "logloss": s_ll / n,
            "mean_pred": s_p / n, "observed": events / n, "calibration": cal}
