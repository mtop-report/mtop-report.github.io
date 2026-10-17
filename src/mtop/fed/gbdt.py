"""Gradient-boosted trees learned from network-wide histograms (lossless federated boosting).

The coordinator grows each tree level by level. For the nodes it needs, every centre returns,
per column and per public value band, the sums of gradients G, Hessians H and patient counts N
over its own patients; secure aggregation turns these into network totals. Split search, gains
and leaf values are computed from the totals only, so the result is the same model that pooled
training with this algorithm would give. Per-patient gradients are rounded to 2^-24 before they
are added up, which makes all sums exact integers: federated and pooled training then agree bit
for bit, not just approximately.

Algorithm: binary log-loss, Newton leaves w = -G / (H + lambda) times the learning rate, depth-wise
growth, missing values sent to the better side, histogram subtraction (only the smaller child of
each split is requested), at least `min_child` patients in every child, and monotone constraints
with bound propagation (as in XGBoost) so that e.g. higher age can never raise P(mRS 0-2).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .bins import Bands, fed_settings, min_leaf


@dataclass
class GBDTParams:
    trees: int = 100
    learning_rate: float = 0.05
    max_depth: int = 3
    min_child: int = 25
    reg_lambda: float = 1.0
    colsample: float = 1.0
    min_gain: float = 0.0
    seed: int = 20261016
    bits: int = 24

    @classmethod
    def default(cls, **over) -> "GBDTParams":
        g = fed_settings()["gbdt"]
        p = cls(trees=int(g["trees"]), learning_rate=float(g["learning_rate"]), max_depth=int(g["max_depth"]),
                min_child=int(g["min_child"]), reg_lambda=float(g["lambda"]), colsample=float(g["colsample"]),
                bits=int(fed_settings()["network"]["fixed_point_bits"]))
        for k, v in over.items():
            setattr(p, k, v)
        p.min_child = max(int(p.min_child), min_leaf())
        return p

    def rounds_budget(self) -> int:
        """Upper bound on aggregation rounds a training job may use (sites enforce it)."""
        return 1 + self.trees * self.max_depth


# --------------------------------------------------------------------------- tree evaluation
def route(nodes: list[dict], B: np.ndarray) -> np.ndarray:
    """Node index reached by each row. Children always have larger indices than their parent."""
    pos = np.zeros(len(B), dtype=np.int32)
    for i, nd in enumerate(nodes):
        if "col" not in nd:
            continue
        m = pos == i
        if not m.any():
            continue
        b = B[m, nd["col"]]
        miss = b == nd["miss_bin"]
        left = np.where(miss, bool(nd["miss_left"]), b <= nd["bin"])
        pos[m] = np.where(left, nd["left"], nd["right"])
    return pos


def tree_output(tree: dict, B: np.ndarray) -> np.ndarray:
    vals = np.array([nd.get("value", 0.0) for nd in tree["nodes"]], float)
    return vals[route(tree["nodes"], B)]


def raw_score(model: dict, B: np.ndarray, n_trees: int | None = None) -> np.ndarray:
    F = np.full(len(B), float(model["init"]))
    for tree in model["trees"][: n_trees if n_trees is not None else len(model["trees"])]:
        F += tree_output(tree, B)
    return F


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def quantised_gradients(F: np.ndarray, y: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(F)
    q = float(1 << bits)
    return np.rint((p - y) * q), np.rint(p * (1.0 - p) * q)


def local_histograms(B: np.ndarray, gq: np.ndarray, hq: np.ndarray, pos: np.ndarray, nodes: list[int],
                     cols: list[int], nbins: np.ndarray) -> np.ndarray:
    """[node][col] -> G bands, H bands, N bands, flattened (int64, exact)."""
    nb = nbins[cols]
    width = int(nb.sum())
    off = np.concatenate([[0], np.cumsum(nb)[:-1]])
    out = []
    for node in nodes:
        m = pos == node
        idx = (B[np.ix_(m, cols)].astype(np.int64) + off[None, :]).ravel()
        g = np.repeat(gq[m], len(cols))
        h = np.repeat(hq[m], len(cols))
        G = np.bincount(idx, weights=g, minlength=width)
        H = np.bincount(idx, weights=h, minlength=width)
        N = np.bincount(idx, minlength=width).astype(float)
        for j in range(len(cols)):
            s = slice(off[j], off[j] + nb[j])
            out.extend([G[s], H[s], N[s]])
    return np.rint(np.concatenate(out)).astype(np.int64) if out else np.zeros(0, np.int64)


# --------------------------------------------------------------------------- split search
def _obj(G, H, w, lam):
    return G * w + 0.5 * (H + lam) * w * w


def _weight(G, H, lam, lo, hi):
    return np.clip(-G / (H + lam), lo, hi)


def _blocks(vec: np.ndarray, n_nodes: int, cols: list[int], nbins: np.ndarray) -> list[list[np.ndarray]]:
    """Split a flattened histogram vector into [node][col] -> (3, nbins) arrays."""
    out, i = [], 0
    for _ in range(n_nodes):
        per = []
        for j in cols:
            k = int(nbins[j])
            per.append(vec[i:i + 3 * k].reshape(3, k))
            i += 3 * k
        out.append(per)
    if i != len(vec):
        raise ValueError("histogram vector has the wrong length")
    return out


def best_split(hist: list[np.ndarray], cols: list[int], node: dict, mono: list[int], p: GBDTParams) -> dict | None:
    q = float(1 << p.bits)
    lam, lo, hi = p.reg_lambda, node["lo"], node["hi"]
    Gp, Hp = node["G"] / q, node["H"] / q
    base = _obj(Gp, Hp, float(_weight(Gp, Hp, lam, lo, hi)), lam)
    best, best_gain = None, p.min_gain
    for j, h in zip(cols, hist):
        L = h.shape[1] - 2                              # number of edges
        if L < 1:
            continue
        # exact integer sums; option 0 sends missing values right, option 1 sends them left
        cg, ch, cn = (np.cumsum(h[r][:L + 1]) for r in range(3))
        gl, hl, nl = cg[:L], ch[:L], cn[:L]
        gr, hr, nr = cg[L] - gl, ch[L] - hl, cn[L] - nl
        gm, hm, nm = h[0][L + 1], h[1][L + 1], h[2][L + 1]
        GLi, HLi, NL = np.stack([gl, gl + gm]), np.stack([hl, hl + hm]), np.stack([nl, nl + nm])
        GRi, HRi, NR = np.stack([gr + gm, gr]), np.stack([hr + hm, hr]), np.stack([nr + nm, nr])
        GL, HL, GR, HR = GLi / q, HLi / q, GRi / q, HRi / q
        ok = (NL >= p.min_child) & (NR >= p.min_child)
        if nm == 0:                                     # no missing here: send missing to the larger side
            ok[0] &= nl < nr
            ok[1] &= nl >= nr
        wl, wr = _weight(GL, HL, lam, lo, hi), _weight(GR, HR, lam, lo, hi)
        c = mono[j]
        if c > 0:
            ok &= wl <= wr
        elif c < 0:
            ok &= wl >= wr
        gain = base - _obj(GL, HL, wl, lam) - _obj(GR, HR, wr, lam)
        gain = np.where(ok, gain, -np.inf)
        a = int(np.argmax(gain))
        opt, k = divmod(a, L)
        g = float(gain[opt, k])
        if g > best_gain:
            best_gain = g
            best = {"col": int(j), "bin": int(k), "miss_left": bool(opt == 1), "miss_bin": L + 1, "gain": g,
                    "wl": float(wl[opt, k]), "wr": float(wr[opt, k]),
                    "L": (int(GLi[opt, k]), int(HLi[opt, k]), int(NL[opt, k])),
                    "R": (int(GRi[opt, k]), int(HRi[opt, k]), int(NR[opt, k]))}
    return best


# --------------------------------------------------------------------------- training
class Network:
    """What the trainer needs: the network-wide sum of an integer vector computed by every centre."""

    def sum(self, op: str, args: dict) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _columns_for_tree(t: int, p: GBDTParams, n_cols: int) -> list[int]:
    if p.colsample >= 1.0:
        return list(range(n_cols))
    rng = np.random.default_rng([p.seed, t])
    k = max(1, int(round(p.colsample * n_cols)))
    return sorted(int(c) for c in rng.choice(n_cols, size=k, replace=False))


def _wire(nodes: list[dict]) -> list[dict]:
    keys = ("col", "bin", "miss_left", "miss_bin", "left", "right")
    return [{k: nd[k] for k in keys} if "col" in nd else {} for nd in nodes]


def train(net: Network, bands: Bands, mono: list[int], p: GBDTParams, on_tree=None, start: dict | None = None) -> dict:
    """Grow `p.trees` trees. With `start`, continue boosting on top of an existing model (e.g. a
    centre adapting the network model to its own patients)."""
    q = float(1 << p.bits)
    nbins = bands.nbins
    tot = net.sum("init", {})
    n, events = int(tot[0]), int(tot[1])
    if n < 2 * p.min_child or events == 0 or events == n:
        raise ValueError("not enough patients or no outcome variation in the network")
    init = math.log(events / (n - events)) if start is None else float(start["init"])
    model = {"kind": "gbdt", "columns": list(bands.columns), "edges": [list(e) for e in bands.edges],
             "mono": list(mono), "init": init, "params": asdict(p), "trees": list(start["trees"]) if start else [],
             "n": n, "events": events, "train_loss": []}
    first_new = len(model["trees"])
    for t in range(first_new, first_new + p.trees):
        cols = _columns_for_tree(t, p, len(bands.columns))
        nodes = [{"depth": 0, "lo": -np.inf, "hi": np.inf}]
        append = model["trees"] if t == first_new else model["trees"][-1:]
        first = {"tree_index": t, "base": init, "append": append, "tree": _wire(nodes), "nodes": [0],
                 "cols": cols, "loss": True}
        vec = net.sum("hist", first)
        model["train_loss"].append(float(vec[-1]) / q / n)
        hist = {0: _blocks(vec[:-1], 1, cols, nbins)[0]}
        h0 = hist[0][0]
        nodes[0].update(G=int(h0[0].sum()), H=int(h0[1].sum()), N=int(h0[2].sum()))
        frontier = [0]
        for depth in range(p.max_depth):
            pairs = []
            for nid in frontier:
                nd = nodes[nid]
                if nd["N"] < 2 * p.min_child:
                    continue
                sp = best_split(hist[nid], cols, nd, mono, p)
                if sp is None:
                    continue
                c = mono[sp["col"]]
                mid = 0.5 * (sp["wl"] + sp["wr"])
                lb, rb = (nd["lo"], nd["hi"]), (nd["lo"], nd["hi"])
                if c > 0:
                    lb, rb = (nd["lo"], mid), (mid, nd["hi"])
                elif c < 0:
                    lb, rb = (mid, nd["hi"]), (nd["lo"], mid)
                li, ri = len(nodes), len(nodes) + 1
                nodes.append({"depth": depth + 1, "lo": lb[0], "hi": lb[1], "G": sp["L"][0], "H": sp["L"][1], "N": sp["L"][2]})
                nodes.append({"depth": depth + 1, "lo": rb[0], "hi": rb[1], "G": sp["R"][0], "H": sp["R"][1], "N": sp["R"][2]})
                nd.update(col=sp["col"], bin=sp["bin"], miss_left=sp["miss_left"], miss_bin=sp["miss_bin"],
                          left=li, right=ri, gain=sp["gain"])
                pairs.append((nid, li, ri))
            if not pairs:
                break
            frontier = [c for _, li, ri in pairs for c in (li, ri)]
            if depth + 1 >= p.max_depth:
                break
            req = []
            for par, li, ri in pairs:
                if max(nodes[li]["N"], nodes[ri]["N"]) >= 2 * p.min_child:
                    small, large = (li, ri) if nodes[li]["N"] <= nodes[ri]["N"] else (ri, li)
                    req.append((par, small, large))
            if not req:
                break
            vec = net.sum("hist", {"tree_index": t, "append": [], "tree": _wire(nodes), "nodes": [s for _, s, _ in req],
                                   "cols": cols, "loss": False})
            for (par, small, large), blk in zip(req, _blocks(vec, len(req), cols, nbins)):
                hist[small] = blk
                hist[large] = [a - b for a, b in zip(hist[par], blk)]
        tree = {"nodes": []}
        for nd in nodes:
            if "col" in nd:
                tree["nodes"].append({k: nd[k] for k in ("col", "bin", "miss_left", "miss_bin", "left", "right", "N")}
                                     | {"thr": bands.threshold(nd["col"], nd["bin"])})
            else:
                w = float(_weight(nd["G"] / q, nd["H"] / q, p.reg_lambda, nd["lo"], nd["hi"]))
                tree["nodes"].append({"value": w * p.learning_rate, "N": nd["N"]})
        model["trees"].append(tree)
        if on_tree is not None:
            on_tree(t, tree, model)
    return model


# --------------------------------------------------------------------------- export
def to_bundle(model: dict) -> dict:
    """Same pre-order arrays as the browser model (web/model.js); the start score is folded into
    the first tree, as LightGBM does."""
    trees = []
    for ti, tree in enumerate(model["trees"]):
        arr = {k: [] for k in "ftlrdmvc"}
        nodes = tree["nodes"]

        def visit(i):
            k = len(arr["f"])
            for key in arr:
                arr[key].append(0)
            nd = nodes[i]
            if "col" in nd:
                arr["f"][k], arr["t"][k] = nd["col"], nd["thr"]
                arr["d"][k], arr["m"][k], arr["c"][k] = int(nd["miss_left"]), 2, nd["N"]
                arr["l"][k] = visit(nd["left"])
                arr["r"][k] = visit(nd["right"])
            else:
                arr["f"][k], arr["l"][k], arr["r"][k] = -1, -1, -1
                arr["v"][k] = nd["value"] + (model["init"] if ti == 0 else 0.0)
                arr["c"][k] = nd["N"]
            return k

        visit(0)
        trees.append(arr)
    return {"feature_names": model["columns"], "sigmoid": 1.0, "trees": trees}


def leaf_sizes(model: dict) -> list[int]:
    return [nd["N"] for t in model["trees"] if len(t["nodes"]) > 1 for nd in t["nodes"] if "col" not in nd]
