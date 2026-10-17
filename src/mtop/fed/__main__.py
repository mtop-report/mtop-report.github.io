"""Command line for the MTOP federated network.

  python -m mtop.fed template OUT.csv           empty upload template (the shared data dictionary)
  python -m mtop.fed check DATA.csv             how a registry export maps onto the template (offline)
  python -m mtop.fed enrol A B C ...            make access tokens for centres; writes network.yaml
  python -m mtop.fed coordinator --network network.yaml [--job gbdt|logreg] [--wait N] [--out model.json]
  python -m mtop.fed site --coordinator URL --site-id A --token-file a.token --data DATA.csv [--console 8751]
  python -m mtop.fed demo [--quick] [--no-http] four virtual centres from the FNO registry
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
import time
from pathlib import Path

import yaml


def _template(args):
    from .schema import OPTIONAL_COLUMNS, template_columns

    Path(args.out).write_text(",".join(template_columns() + OPTIONAL_COLUMNS) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


def _check(args):
    from .schema import read_site_csv
    from .site import SiteNode

    df, rep = read_site_csv(args.data)
    node = SiteNode("local", df)
    print(json.dumps({"mapping": rep, "eligible": node.describe()["stages"]}, indent=1))


def _enrol(args):
    out = Path(args.out)
    from .server import token_hash

    net = {"name": args.name, "sites": {}}
    for sid in args.sites:
        tok = secrets.token_urlsafe(32)
        net["sites"][sid] = {"token_sha256": token_hash(tok)}
        Path(f"{sid}.token").write_text(tok + "\n", encoding="utf-8")
        print(f"{sid}: token written to {sid}.token (give it to that centre only)")
    out.write_text(yaml.safe_dump(net, sort_keys=False), encoding="utf-8")
    print(f"wrote {out} (stores token hashes only)")


def _coordinator(args):
    from .coordinator import Coordinator, federated_eval, federated_gbdt, federated_logreg
    from .gbdt import GBDTParams, to_bundle
    from .server import Hub, serve

    net = yaml.safe_load(Path(args.network).read_text(encoding="utf-8"))
    hub = Hub({sid: v["token_sha256"] for sid, v in net["sites"].items()}, name=net.get("name", "MTOP coordinator"))
    httpd, _ = serve(hub, args.host, args.port, args.cert, args.key)
    print(f"coordinator on {'https' if args.cert else 'http'}://{args.host}:{httpd.server_address[1]}/  (dashboard)")
    ids = hub.wait_for(args.wait, timeout=args.timeout)
    print(f"{len(ids)} centres connected: {', '.join(ids)}")
    co = Coordinator(hub, log=print)
    hub.status["phase"] = "training"
    if args.job == "gbdt":
        p = GBDTParams.default()

        def on_tree(t, tree, model):
            hub.status["job"] = {"kind": "gbdt", "tree": t + 1, "trees": p.trees}
            hub.status["curve"] = list(model["train_loss"])

        model, job = federated_gbdt(co, ids, args.stage, args.target, None, p, on_tree)
        model["bundle"] = to_bundle(model)
    else:
        model, job = federated_logreg(co, ids, args.stage, args.target)
    hub.status["phase"] = "evaluating"
    res, _ = federated_eval(co, ids, model) if args.job == "gbdt" else ({}, None)
    model["apparent_performance"] = res
    Path(args.out).write_text(json.dumps(model), encoding="utf-8")
    hub.status["phase"] = "done"
    print(f"model written to {args.out} after {job.round} rounds")
    if args.serve_after:
        print("dashboard stays up; Ctrl-C to stop")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    httpd.shutdown()


def _site(args):
    from .client import Approvals, SiteClient, console
    from .schema import read_site_csv
    from .secagg import Identity
    from .site import SiteNode

    df, rep = read_site_csv(args.data)
    print(f"{rep['rows']} rows read; missing columns: {', '.join(rep['columns_missing']) or 'none'}")
    ident = Identity.load_or_create(Path(args.state) / f"{args.site_id}.identity.pem")
    pinned = yaml.safe_load(Path(args.pins).read_text(encoding="utf-8")) if args.pins else None
    approvals = None if args.auto_approve else Approvals()
    node = SiteNode(args.site_id, df, label=args.label or args.site_id, identity=ident,
                    approve=True if args.auto_approve else approvals, pinned=pinned,
                    audit_path=Path(args.state) / f"{args.site_id}.audit.jsonl")
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    print(f"identity fingerprint {ident.fingerprint}")
    if args.console:
        console(node, approvals, args.coordinator, args.console)
        print(f"console on http://127.0.0.1:{args.console}/ (this machine only)")
    client = SiteClient(node, args.coordinator, token, cafile=args.cafile)
    try:
        client.run()
    except KeyboardInterrupt:
        client.stop.set()


def _demo(args):
    from .demo import run

    out = run(http=not args.no_http, quick=args.quick)
    print(json.dumps({t: r["mean"] for t, r in out["outcomes"].items()}
                     | {"identical": out["equivalence"]["gbdt"]["identical"], "seconds": out["seconds_total"]}, indent=1))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m mtop.fed", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("template")
    s.add_argument("out")
    s.set_defaults(fn=_template)
    s = sub.add_parser("check")
    s.add_argument("data")
    s.set_defaults(fn=_check)
    s = sub.add_parser("enrol")
    s.add_argument("sites", nargs="+")
    s.add_argument("--out", default="network.yaml")
    s.add_argument("--name", default="MTOP network")
    s.set_defaults(fn=_enrol)
    s = sub.add_parser("coordinator")
    s.add_argument("--network", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8750)
    s.add_argument("--cert")
    s.add_argument("--key")
    s.add_argument("--job", choices=["gbdt", "logreg"], default="gbdt")
    s.add_argument("--stage", default="admission")
    s.add_argument("--target", default="good_outcome")
    s.add_argument("--wait", type=int, default=3, help="start when this many centres are online")
    s.add_argument("--timeout", type=float, default=3600)
    s.add_argument("--out", default="network_model.json")
    s.add_argument("--serve-after", action="store_true")
    s.set_defaults(fn=_coordinator)
    s = sub.add_parser("site")
    s.add_argument("--coordinator", required=True)
    s.add_argument("--site-id", required=True)
    s.add_argument("--token-file", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--label")
    s.add_argument("--state", default=".mtop-site", help="folder for the identity key and audit log")
    s.add_argument("--pins", help="YAML: centre id -> identity fingerprint of the other centres")
    s.add_argument("--console", type=int, default=8751, help="local approval console port (0 = off)")
    s.add_argument("--auto-approve", action="store_true", help="approve every job (testing only)")
    s.add_argument("--cafile")
    s.set_defaults(fn=_site)
    s = sub.add_parser("demo")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--no-http", action="store_true")
    s.set_defaults(fn=_demo)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
