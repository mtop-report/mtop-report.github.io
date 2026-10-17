"""A centre's node: answers the coordinator's tasks with masked sums over its own patients.

The node holds the centre's registry export (already parsed with `schema.read_site_csv`) and never
sends a row of it. For every job it checks the request against its own rules before taking part:

  - the job comes with the protocol version this node speaks and lists at least `min_sites` centres;
  - the centre has at least `min_site_patients` eligible patients for it;
  - a person (or, for unattended demo runs, the auto-approve setting) approves it;
  - every round asks for an operation the job type allows, in increasing order, within the round
    budget declared when the job was approved;
  - the other centres' keys are signed by their identities (and match pinned fingerprints, if set).

Every message the node sends is written to an append-only audit log (time, job, round, operation,
size and SHA-256 of what left the hospital).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..cohort import stage_cohort
from ..config import stage_ids, targets
from ..features import design_matrix, monotone_vector
from . import PROTOCOL
from .bins import Bands, fed_settings
from .gbdt import GBDTParams, local_histograms, quantised_gradients, raw_score, route, sigmoid, tree_output
from .logreg import logit, newton_terms, predict_spline, spline_design
from .secagg import Identity, JobKeys, b64, pack, to_fixed

OPS = {"gbdt": {"init", "hist"}, "logreg": {"moments", "lr"}, "eval": {"eval", "lr"}}


class Declined(Exception):
    pass


class SiteNode:
    def __init__(self, site_id: str, data: pd.DataFrame, *, label: str | None = None, identity: Identity | None = None,
                 approve=True, audit_path=None, pinned: dict | None = None, min_patients: int | None = None):
        net = fed_settings()["network"]
        self.site_id = site_id
        self.label = label or site_id
        self.identity = identity or Identity()
        self.approve = approve
        self.pinned = pinned
        self.min_patients = int(net["min_site_patients"] if min_patients is None else min_patients)
        self.min_sites = int(net["min_sites"])
        self.bits = int(net["fixed_point_bits"])
        self.data = data[data["age"].notna()].reset_index(drop=True)
        self.audit_path = Path(audit_path) if audit_path else None
        self.audit: list[dict] = []
        self.jobs: dict[str, dict] = {}
        self.pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._cache: dict = {}

    # ------------------------------------------------------------------ local data views
    def cohort(self, stage: str, target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        key = (stage, target)
        if key not in self._cache:
            d = stage_cohort(self.data, stage)
            d = d[d[target].notna()].reset_index(drop=True)
            bands = Bands.for_stage(stage)
            B = bands.transform(design_matrix(d, stage).to_numpy(float))
            self._cache[key] = (d, B, d[target].to_numpy(float))
        return self._cache[key]

    def folds(self, y: np.ndarray, k: int, seed: int) -> np.ndarray:
        """Stratified fold labels, decided locally (the coordinator never sees them)."""
        rng = np.random.default_rng([int(seed), zlib.crc32(self.site_id.encode())])
        f = np.empty(len(y), dtype=np.int64)
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(y == cls)
            idx = idx[rng.permutation(len(idx))]
            f[idx] = np.arange(len(idx)) % k
        return f

    def rows(self, stage: str, target: str, subset: dict | None) -> np.ndarray:
        _, _, y = self.cohort(stage, target)
        idx = np.arange(len(y))
        subset = subset or {"kind": "all"}
        if subset.get("kind", "all") == "all":
            return idx
        if subset["kind"] == "cv":
            f = self.folds(y, int(subset["folds"]), int(subset["seed"]))
            inside = f == int(subset["fold"])
            return idx[~inside] if subset["part"] == "train" else idx[inside]
        raise Declined(f"unknown subset {subset.get('kind')}")

    def describe(self) -> dict:
        out = {"site": self.site_id, "label": self.label, "rows": int(len(self.data)), "fingerprint": self.identity.fingerprint,
               "stages": {}}
        for st in stage_ids():
            out["stages"][st] = {t: int(len(self.cohort(st, t)[2])) for t in targets()}
        return out

    # ------------------------------------------------------------------ audit
    def _log(self, **entry) -> None:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "site": self.site_id, **entry}
        with self._lock:
            self.audit.append(entry)
            if self.audit_path is not None:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.audit_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ protocol
    def handle(self, msg: dict) -> dict:
        kind = msg.get("type")
        try:
            if kind == "hello":
                return {"site": self.site_id, "label": self.label, "identity": b64(self.identity.public),
                        "fingerprint": self.identity.fingerprint, "protocol": PROTOCOL}
            if kind == "offer":
                return self._offer(msg)
            if kind == "start":
                return self._start(msg)
            if kind == "round":
                return self._round(msg)
            if kind in ("finish", "abort"):
                job = self.jobs.pop(msg["job"], None)
                self.pending.pop(msg["job"], None)
                self._log(job=msg["job"], op=kind, bytes=0, note=msg.get("reason", ""))
                if job is not None and kind == "finish":
                    job["result"] = msg.get("result")
                return {"ok": True}
            raise Declined(f"unknown message {kind}")
        except Declined as e:
            self._log(job=msg.get("job"), op=f"refused {kind}", bytes=0, note=str(e))
            return {"accept": False, "error": str(e)}

    def _offer(self, msg: dict) -> dict:
        if msg.get("protocol") != PROTOCOL:
            raise Declined("protocol version differs")
        if msg["kind"] not in OPS:
            raise Declined(f"job type {msg['kind']} is not supported")
        parts = list(msg["participants"])
        if self.site_id not in parts or len(set(parts)) < self.min_sites:
            raise Declined(f"secure aggregation needs at least {self.min_sites} centres")
        spec = msg["spec"]
        stage, target = spec["stage"], spec["target"]
        if stage not in stage_ids() or target not in targets():
            raise Declined("unknown stage or outcome")
        n = len(self.rows(stage, target, spec.get("subset")))
        if n < self.min_patients:
            raise Declined(f"fewer than {self.min_patients} eligible patients")
        budget = int(msg["budget"])
        if msg["kind"] == "gbdt":
            p = GBDTParams(**spec["params"])
            if budget > p.rounds_budget():
                raise Declined("round budget larger than the model needs")
        if budget > 2000:
            raise Declined("round budget too large")
        info = {"job": msg["job"], "kind": msg["kind"], "spec": {k: v for k, v in spec.items() if k != "model"},
                "participants": parts, "budget": budget, "patients": n}
        ok = self.approve(info) if callable(self.approve) else bool(self.approve)
        if not ok:
            raise Declined("not approved at this centre")
        keys = JobKeys(msg["job"], self.site_id, self.identity)
        self.pending[msg["job"]] = {"msg": msg, "keys": keys, "n": n}
        reply = {"accept": True, "identity": b64(self.identity.public), **keys.offer()}
        self._log(job=msg["job"], op="accept", bytes=len(json.dumps(reply)), note=f"{msg['kind']} · {n} patients · budget {budget}")
        return reply

    def _start(self, msg: dict) -> dict:
        pend = self.pending.pop(msg["job"], None)
        if pend is None:
            raise Declined("job was not accepted here")
        offer = pend["msg"]
        if sorted(msg["peers"]) != sorted(offer["participants"]):
            raise Declined("participants changed after approval")
        keys: JobKeys = pend["keys"]
        try:
            keys.agree(msg["peers"], self.pinned)
        except PermissionError as e:
            raise Declined(str(e)) from e
        spec = offer["spec"]
        stage, target = spec["stage"], spec["target"]
        rows = self.rows(stage, target, spec.get("subset"))
        job = {"kind": offer["kind"], "spec": spec, "keys": keys, "budget": int(offer["budget"]), "round": 0,
               "stage": stage, "target": target, "rows": rows, "n_trees": 0, "F": None}
        self.jobs[msg["job"]] = job
        self._log(job=msg["job"], op="start", bytes=0, note=f"{len(msg['peers'])} centres, keys verified")
        return {"ok": True}

    def _round(self, msg: dict) -> dict:
        job = self.jobs.get(msg["job"])
        if job is None:
            raise Declined("no such job running here")
        r = int(msg["round"])
        if r <= job["round"]:
            raise Declined("round already answered")
        if r > job["budget"]:
            raise Declined("round budget used up")
        op = msg["op"]
        if op not in OPS[job["kind"]]:
            raise Declined(f"operation {op} not part of a {job['kind']} job")
        if op == "lr":
            allowed = {"spline"} if job["kind"] == "logreg" else {"recal", "citl"}
            if msg.get("args", {}).get("design", {}).get("kind") not in allowed:
                raise Declined("regression design not part of this job")
        job["round"] = r
        vec = self._compute(job, op, msg.get("args", {}))
        masked = job["keys"].mask(vec, r)
        payload = pack(masked)
        self._log(job=msg["job"], round=r, op=op, bytes=len(payload), values=int(len(vec)),
                  sha256=hashlib.sha256(payload.encode()).hexdigest()[:16])
        return {"vec": payload}

    # ------------------------------------------------------------------ computations
    def _compute(self, job: dict, op: str, args: dict) -> np.ndarray:
        d, B, y = self.cohort(job["stage"], job["target"])
        rows = job["rows"]
        return compute(op, args, job, d.iloc[rows], B[rows], y[rows], self.bits)


def compute(op: str, args: dict, job: dict, d: pd.DataFrame, B: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    """One centre's integer vector for a task. Shared by site nodes and pooled training."""
    if op == "init":
        return np.array([len(y), int(y.sum())], dtype=np.int64)
    if op == "hist":
        if job["F"] is None:
            job["F"] = np.full(len(y), float(args["base"]))
        for tree in args.get("append", []):
            job["F"] = job["F"] + tree_output(tree, B)
            job["n_trees"] += 1
        if "tree_index" in args and int(args["tree_index"]) != job["n_trees"]:
            raise Declined("model out of step with the coordinator")
        gq, hq = quantised_gradients(job["F"], y, bits)
        pos = route(args["tree"], B)
        nbins = Bands.for_stage(job["stage"]).nbins
        vec = local_histograms(B, gq, hq, pos, list(args["nodes"]), list(args["cols"]), nbins)
        if args.get("loss"):
            p = np.clip(sigmoid(job["F"]), 1e-12, 1 - 1e-12)
            ll = -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            vec = np.concatenate([vec, to_fixed([ll], bits)])
        return vec
    if op == "moments":
        X, _ = spline_design(d, args["design"])
        return np.concatenate([[len(y)], to_fixed(np.concatenate([X.sum(0), (X ** 2).sum(0)]), bits)]).astype(np.int64)
    if op == "lr":
        design = args["design"]
        beta = np.asarray(args["beta"], float)
        if design["kind"] == "spline":
            X, _ = spline_design(d, design)
            Z = (X - np.asarray(args["mean"])) / np.asarray(args["sd"])
            X1, off = np.column_stack([np.ones(len(y)), Z]), None
        else:
            z = logit(predict(job["spec"]["model"], d, B))
            if design["kind"] == "recal":
                X1, off = np.column_stack([np.ones(len(y)), z]), None
            elif design["kind"] == "citl":
                X1, off = np.ones((len(y), 1)), z
            else:
                raise Declined("unknown design")
        return to_fixed(newton_terms(X1, y, beta, off), bits)
    if op == "eval":
        return eval_vector(predict(job["spec"]["model"], d, B), y, bits)
    raise Declined(f"unknown operation {op}")


def predict(model: dict, d: pd.DataFrame, B: np.ndarray) -> np.ndarray:
    if model["kind"] == "gbdt":
        return sigmoid(raw_score(model, B))
    if model["kind"] == "logreg":
        return predict_spline(model, d)
    raise Declined("unknown model type")


def eval_bins() -> tuple[int, float, float]:
    ev = fed_settings()["evaluation"]
    lo, hi = ev["logit_range"]
    return int(ev["logit_bins"]), float(lo), float(hi)


def eval_vector(p: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    """Per narrow logit band: events, non-events and the sum of predicted risks; then n, events,
    and the sums of p, (p - y)^2 and the log-loss."""
    nb, lo, hi = eval_bins()
    z = logit(p)
    b = np.clip(np.floor((z - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
    pos = np.bincount(b[y == 1], minlength=nb)
    neg = np.bincount(b[y == 0], minlength=nb)
    psum = np.bincount(b, weights=p, minlength=nb)
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    sums = [float(p.sum()), float(((p - y) ** 2).sum()), -float(np.sum(y * np.log(pc) + (1 - y) * np.log(1 - pc)))]
    return np.concatenate([pos, neg, to_fixed(psum, bits), [len(y), int(y.sum())], to_fixed(sums, bits)]).astype(np.int64)


def mono_for(stage: str, target: str) -> list[int]:
    return monotone_vector(stage, target)
