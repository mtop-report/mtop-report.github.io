"""Federated network: secure aggregation, exactness against pooled training, site rules, HTTP.

Runs on synthetic centres only (no registry data needed)."""
from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cryptography")

from mtop.fed.bins import Bands  # noqa: E402
from mtop.fed.client import SiteClient  # noqa: E402
from mtop.fed.coordinator import (Coordinator, JobFailed, LocalTransport, PooledNetwork, evaluate,  # noqa: E402
                                  federated_eval, federated_gbdt, federated_logreg, fit_gbdt, fit_logreg)
from mtop.fed.gbdt import GBDTParams, leaf_sizes, raw_score, sigmoid, to_bundle  # noqa: E402
from mtop.fed.logreg import predict_spline, spline_design, spline_spec  # noqa: E402
from mtop.fed.schema import read_site_csv, template_columns, write_site_csv  # noqa: E402
from mtop.fed.secagg import Identity, JobKeys, aggregate, b64, fingerprint, unb64  # noqa: E402
from mtop.fed.server import Hub, serve, token_hash  # noqa: E402
from mtop.fed.site import SiteNode  # noqa: E402
from mtop.features import design_matrix  # noqa: E402

STAGE, TARGET = "admission", "good_outcome"


def synthetic_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def maybe(x, frac):
        x = np.asarray(x, float)
        x[rng.random(n) < frac] = np.nan
        return x

    age = rng.integers(30, 95, n).astype(float)
    nihss = rng.integers(2, 30, n).astype(float)
    mrs_pre = rng.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.12, 0.08]).astype(float)
    aspects = maybe(rng.integers(4, 11, n), 0.1)
    df = pd.DataFrame({
        "age": age, "sex_male": rng.integers(0, 2, n).astype(float), "mrs_pre": mrs_pre, "nihss_adm": nihss,
        "aspects": aspects, "occlusion": rng.choice(["ICA", "M1", "M2", "basilar"], n, p=[0.2, 0.5, 0.2, 0.1]),
        "side_left": rng.integers(0, 2, n).astype(float), "onset_to_puncture": maybe(rng.integers(60, 600, n), 0.15),
        "ivt": rng.integers(0, 2, n).astype(float), "transfer": rng.integers(0, 2, n).astype(float),
        "glucose": maybe(np.round(rng.uniform(4, 15, n), 1), 0.1), "sbp": maybe(rng.integers(100, 220, n), 0.05),
        "htn": rng.integers(0, 2, n).astype(float), "dm": rng.integers(0, 2, n).astype(float),
        "arrhythmia": rng.integers(0, 2, n).astype(float), "stroke_prior": rng.integers(0, 2, n).astype(float),
        "ivt_external": rng.integers(0, 2, n).astype(float), "onset_to_ivt": maybe(rng.integers(40, 240, n), 0.3),
        "onset_hour": maybe(rng.integers(0, 24, n), 0.05), "weekend": rng.integers(0, 2, n).astype(float),
        "dbp": maybe(rng.integers(50, 120, n), 0.05), "bmi": maybe(np.round(rng.uniform(18, 40, n), 1), 0.05),
        "hld": rng.integers(0, 2, n).astype(float), "tia_prior": rng.integers(0, 2, n).astype(float),
        "smoking": maybe(rng.integers(0, 2, n), 0.1), "alcohol": maybe(rng.integers(0, 2, n), 0.1),
        "statin": rng.integers(0, 2, n).astype(float), "heart3": rng.choice(["none", "coronary", "other"], n, p=[0.6, 0.3, 0.1]),
        "antithrombotic": rng.choice(["none", "antiplatelet", "anticoagulant"], n, p=[0.55, 0.3, 0.15]),
        "tici": rng.choice([0.0, 1.0, 2.0, 2.5, 3.0, 4.0], n, p=[0.05, 0.03, 0.07, 0.05, 0.4, 0.4]),
        "procedure_min": maybe(rng.integers(10, 150, n), 0.1),
        "technique": rng.choice(["aspiration", "stent_retriever", "solumbra", "combined"], n),
        "stenting": rng.integers(0, 2, n).astype(float), "angioplasty": (rng.random(n) < 0.1).astype(float),
        "ia_lysis": (rng.random(n) < 0.05).astype(float), "onset_to_recan": maybe(rng.integers(120, 700, n), 0.2),
        "anesthesia": rng.choice(["general", "sedation", "local"], n),
        "nihss_24h": np.clip(nihss + rng.integers(-10, 6, n), 0, 42).astype(float), "sich": (rng.random(n) < 0.06).astype(float),
        "ich_other": (rng.random(n) < 0.1).astype(float),
        "nihss_7d": np.clip(nihss + rng.integers(-14, 4, n), 0, 42).astype(float), "edema": (rng.random(n) < 0.1).astype(float),
        "decompression": (rng.random(n) < 0.03).astype(float),
    })
    lp = 1.4 - 0.05 * (age - 70) - 0.13 * (nihss - 15) - 0.6 * mrs_pre + 0.2 * (np.nan_to_num(aspects, nan=8) - 8)
    good = rng.random(n) < sigmoid(lp)
    df["mrs90"] = np.where(good, rng.choice([0, 1, 2], n), rng.choice([3, 4, 5, 6], n)).astype(float)
    return df


@pytest.fixture(scope="module")
def sites(tmp_path_factory):
    d = tmp_path_factory.mktemp("fed")
    out = []
    for i, n in enumerate([220, 180, 260, 150]):
        f = d / f"site_{i}.csv"
        write_site_csv(synthetic_frame(n, 100 + i), f)
        typed, rep = read_site_csv(f)
        assert rep["columns_missing"] == [] and rep["rows"] == n
        out.append(SiteNode("ABCD"[i], typed, audit_path=d / f"audit_{i}.jsonl"))
    return out


SMALL = dict(trees=12, learning_rate=0.1, max_depth=3, min_child=20)


# --------------------------------------------------------------------------- secure aggregation
def _keys(n=4, job="job-1"):
    ids = [Identity() for _ in range(n)]
    keys = [JobKeys(job, f"S{i}", ids[i]) for i in range(n)]
    peers = {f"S{i}": {**keys[i].offer(), "identity": b64(ids[i].public)} for i in range(n)}
    return ids, keys, peers


def test_masks_cancel_and_hide_values():
    ids, keys, peers = _keys()
    for k in keys:
        k.agree(peers)
    rng = np.random.default_rng(1)
    vals = [rng.integers(-10**12, 10**12, 50, dtype=np.int64) for _ in keys]
    masked = [k.mask(v, round_id=7) for k, v in zip(keys, vals)]
    assert np.array_equal(aggregate(masked), np.sum(vals, axis=0))
    for m, v in zip(masked, vals):
        assert not np.any(m.view(np.int64) == v)          # nothing travels in the clear
    # a different round gives different masks for the same values
    again = keys[0].mask(vals[0], round_id=8)
    assert not np.array_equal(again, masked[0])


def test_key_substitution_is_detected():
    ids, keys, peers = _keys()
    evil = JobKeys("job-1", "S1", Identity())                # coordinator swaps in its own key
    bad = dict(peers, S1={**peers["S1"], "x25519": evil.offer()["x25519"]})
    with pytest.raises(PermissionError):
        keys[0].agree(bad)
    pinned = {"S1": "0000:0000:0000:0000"}
    with pytest.raises(PermissionError):
        keys[0].agree(peers, pinned)
    keys[0].agree(peers, {"S1": fingerprint(unb64(peers["S1"]["identity"]))})


# --------------------------------------------------------------------------- bands and trees
def test_band_index_matches_threshold():
    bands = Bands.for_stage("d7")
    rng = np.random.default_rng(3)
    for j, e in enumerate(bands.edges):
        x = np.concatenate([rng.uniform(min(e) - 5, max(e) + 5, 200), np.asarray(e), [np.nan]])
        X = np.full((len(x), len(bands.columns)), np.nan)
        X[:, j] = x
        b = bands.transform(X)[:, j]
        assert b[-1] == bands.missing_bin(j)
        for k, edge in enumerate(e):
            ok = ~np.isnan(x)
            assert np.array_equal((b[ok] <= k), (x[ok] <= edge))


def test_federated_trees_identical_to_pooled(sites):
    co = Coordinator(LocalTransport(sites))
    p = GBDTParams.default(**SMALL)
    fed, job = federated_gbdt(co, [s.site_id for s in sites], STAGE, TARGET, params=p)
    ref = fit_gbdt(PooledNetwork.from_sites(sites, STAGE, TARGET), STAGE, TARGET, p)
    assert fed["trees"] == ref["trees"] and fed["init"] == ref["init"]
    assert job.round <= p.rounds_budget()
    assert min(leaf_sizes(fed)) >= p.min_child


def test_monotone_constraints_hold(sites):
    p = GBDTParams.default(**SMALL)
    model = fit_gbdt(PooledNetwork.from_sites(sites, STAGE, TARGET), STAGE, TARGET, p)
    bands = Bands.for_stage(STAGE)
    d, B, y = sites[0].cohort(STAGE, TARGET)
    X = design_matrix(d.iloc[:40], STAGE).to_numpy(float)
    for col, sign in (("age", -1), ("nihss_adm", -1), ("aspects", 1)):
        j = bands.columns.index(col)
        prev = None
        for v in np.linspace(0, 100, 41):
            X2 = X.copy()
            X2[:, j] = v
            F = raw_score(model, bands.transform(X2))
            if prev is not None:
                assert np.all(sign * (F - prev) >= -1e-12), col
            prev = F


def test_browser_export_gives_same_scores(sites):
    p = GBDTParams.default(**SMALL)
    model = fit_gbdt(PooledNetwork.from_sites(sites, STAGE, TARGET), STAGE, TARGET, p)
    bundle = to_bundle(model)
    d, B, y = sites[1].cohort(STAGE, TARGET)
    X = design_matrix(d, STAGE).to_numpy(float)

    def tree_value(t, x):
        i = 0
        while t["f"][i] >= 0:
            v = x[t["f"][i]]
            i = (t["l"][i] if t["d"][i] else t["r"][i]) if np.isnan(v) else (t["l"][i] if v <= t["t"][i] else t["r"][i])
        return t["v"][i]

    js_like = np.array([sum(tree_value(t, x) for t in bundle["trees"]) for x in X])
    assert np.allclose(js_like, raw_score(model, B), atol=1e-12)


# --------------------------------------------------------------------------- regression and evaluation
def test_federated_logreg_matches_pooled(sites):
    co = Coordinator(LocalTransport(sites))
    fed, _ = federated_logreg(co, [s.site_id for s in sites], STAGE, TARGET)
    pool = PooledNetwork.from_sites(sites, STAGE, TARGET)
    ref = fit_logreg(pool, spline_spec(STAGE))
    assert np.max(np.abs(np.asarray(fed["beta"]) - np.asarray(ref["beta"]))) < 1e-5
    assert np.max(np.abs(predict_spline(fed, pool.d) - predict_spline(ref, pool.d))) < 1e-6
    X, names = spline_design(pool.d, spline_spec(STAGE))
    assert len(names) == X.shape[1] == len(fed["beta"]) - 1


def test_pooled_metrics_from_counts(sites):
    from sklearn.metrics import roc_auc_score

    co = Coordinator(LocalTransport(sites))
    p = GBDTParams.default(**SMALL)
    ids = [s.site_id for s in sites]
    model, _ = federated_gbdt(co, ids, STAGE, TARGET, params=p)
    res, job = federated_eval(co, ids, model)
    pool = PooledNetwork.from_sites(sites, STAGE, TARGET, spec={"model": model})
    exact = roc_auc_score(pool.y, sigmoid(raw_score(model, pool.B)))
    assert abs(res["auroc"] - exact) < 0.005
    ref = evaluate(pool)
    assert abs(ref["auroc"] - res["auroc"]) < 1e-12 and abs(ref["slope"] - res["slope"]) < 1e-5
    assert all(g["n"] >= 10 for g in res["calibration"])


# --------------------------------------------------------------------------- the centres' rules
def test_site_rules(sites, tmp_path):
    co = Coordinator(LocalTransport(sites))
    ids = [s.site_id for s in sites]
    p = GBDTParams.default(**SMALL)
    with pytest.raises(JobFailed):                         # two centres: no secure aggregation
        federated_gbdt(co, ids[:2], STAGE, TARGET, params=p)
    tiny = SiteNode("T", sites[0].data.iloc[:30].reset_index(drop=True))
    co2 = Coordinator(LocalTransport(sites[:2] + [tiny]))
    with pytest.raises(JobFailed, match="fewer than"):
        federated_gbdt(co2, ["A", "B", "T"], STAGE, TARGET, params=p)
    declining = SiteNode("N", sites[3].data, approve=lambda info: False)
    co3 = Coordinator(LocalTransport(sites[:2] + [declining]))
    with pytest.raises(JobFailed, match="not approved"):
        federated_gbdt(co3, ["A", "B", "N"], STAGE, TARGET, params=p)
    # an approved job cannot be used for other operations, replays or more rounds than declared
    job = co.open("gbdt", {"stage": STAGE, "target": TARGET, "subset": {"kind": "all"},
                           "params": __import__("dataclasses").asdict(p)}, ids, 3)
    job.sum("init", {})
    rep = co.t.exchange(ids, {"type": "round", "job": job.id, "round": 1, "op": "init", "args": {}})
    assert all("vec" not in r for r in rep.values())
    rep = co.t.exchange(ids, {"type": "round", "job": job.id, "round": 2, "op": "eval", "args": {}})
    assert all("operation" in r["error"] for r in rep.values())
    rep = co.t.exchange(ids, {"type": "round", "job": job.id, "round": 9, "op": "init", "args": {}})
    assert all("budget" in r["error"] for r in rep.values())
    assert any(a["op"].startswith("refused") for a in sites[0].audit)
    assert all("sha256" in a for a in sites[0].audit if a.get("round"))


# --------------------------------------------------------------------------- over HTTP
def test_http_network_end_to_end(sites):
    tokens = {s.site_id: secrets.token_urlsafe(16) for s in sites}
    hub = Hub({k: token_hash(v) for k, v in tokens.items()})
    httpd, _ = serve(hub, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    clients = [SiteClient(s, url, tokens[s.site_id], wait=2) for s in sites]
    try:
        for c in clients:
            c.start()
        ids = hub.wait_for(len(sites), timeout=20)
        p = GBDTParams.default(trees=4, learning_rate=0.1, max_depth=2, min_child=20)
        model, job = federated_gbdt(Coordinator(hub), ids, STAGE, TARGET, params=p)
        ref = fit_gbdt(PooledNetwork.from_sites(sites, STAGE, TARGET), STAGE, TARGET, p)
        assert model["trees"] == ref["trees"]
        assert all(v["up"] > 0 and v["down"] > 0 for v in hub.stats.values())
        status = json.loads(urllib.request.urlopen(url + "/v1/status").read())
        assert {s["id"] for s in status["sites"]} == set(ids)
        assert "token" not in json.dumps(status).lower()
        req = urllib.request.Request(url + "/v1/next", headers={"X-Site": "A", "Authorization": "Bearer wrong"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 401
    finally:
        for c in clients:
            c.stop.set()
        httpd.shutdown()


def test_jobs_on_a_range_of_treatment_years(tmp_path):
    """Optional `year` column: a network job can use only, e.g., the most recent years."""
    nodes = []
    for i, n in enumerate([200, 180, 220]):
        df = synthetic_frame(n, 300 + i)
        df["year"] = np.random.default_rng(i).integers(2015, 2026, n)
        f = tmp_path / f"y{i}.csv"
        write_site_csv(df, f)
        typed, rep = read_site_csv(f)
        assert "year" in rep["columns_found"] and typed["year"].between(2015, 2025).all()
        nodes.append(SiteNode("XYZ"[i], typed))
    recent = {"kind": "years", "from": 2019, "to": 2025}
    for s in nodes:
        d, B, y = s.cohort(STAGE, TARGET)
        assert len(s.rows(STAGE, TARGET, recent)) == int((d["year"] >= 2019).sum())
    co = Coordinator(LocalTransport(nodes))
    p = GBDTParams.default(trees=3, learning_rate=0.1, max_depth=2, min_child=20)
    fed, _ = federated_gbdt(co, ["X", "Y", "Z"], STAGE, TARGET, recent, params=p)
    ref = fit_gbdt(PooledNetwork.from_sites(nodes, STAGE, TARGET, recent), STAGE, TARGET, p)
    assert fed["trees"] == ref["trees"] and fed["n"] == sum(len(s.rows(STAGE, TARGET, recent)) for s in nodes)
    # a centre whose export has no year cannot take part in a job defined by years
    no_year = SiteNode("W", read_site_csv(_write(tmp_path, synthetic_frame(150, 9)))[0])
    co2 = Coordinator(LocalTransport(nodes[:2] + [no_year]))
    with pytest.raises(JobFailed, match="no treatment year"):
        federated_gbdt(co2, ["X", "Y", "W"], STAGE, TARGET, recent, params=p)


def _write(tmp_path, df):
    f = tmp_path / f"w{len(df)}.csv"
    write_site_csv(df, f)
    return f


def test_template_columns_are_the_shared_dictionary():
    from mtop.extras import template_columns as extras_template

    assert template_columns() == extras_template()


def test_future_columns_are_optional_and_never_model_inputs(tmp_path):
    """CT perfusion, collaterals and GFAP (config/future.yaml) may be in a centre's export or not;
    when present they are range-checked and counted, and no stage uses them."""
    from mtop.fed.schema import FUTURE, OPTIONAL_COLUMNS

    assert {"ctp_core_ml", "ctp_tmax6_ml", "collaterals_tan", "gfap_pg_ml", "gfap_hours"} <= set(FUTURE)
    assert set(FUTURE) <= set(OPTIONAL_COLUMNS) and not set(FUTURE) & set(template_columns())
    df = synthetic_frame(120, 41)
    f = tmp_path / "with_future.csv"
    write_site_csv(df, f)
    plain, rep0 = read_site_csv(f)
    assert rep0["future_recorded"] == {}
    raw = pd.read_csv(f, dtype=str, keep_default_na=False)
    raw["ctp_core_ml"] = ["12.5"] * 100 + ["-3"] + [""] * 19          # one below the range
    raw["Collaterals_Tan"] = ["2"] * 60 + ["5"] + ["1.5"] + [""] * 58   # 5 and 1.5 are not grades
    raw["gfap_assay"] = ["Alinity i"] * 120
    raw.to_csv(f, index=False)
    typed, rep = read_site_csv(f)
    assert rep["future_recorded"]["ctp_core_ml"] == int(typed["mrs90"].iloc[:100].notna().sum())
    assert rep["unrecognised_values"]["ctp_core_ml"] == 1 and rep["unrecognised_values"]["collaterals_tan"] == 2
    assert typed["gfap_assay"].iloc[0] == "Alinity i"
    a, b = SiteNode("A", plain), SiteNode("B", typed)
    for stage in ("admission", "d7"):
        (_, B1, _), (_, B2, _) = a.cohort(stage, TARGET), b.cohort(stage, TARGET)
        assert np.array_equal(B1, B2)                                     # same model inputs
