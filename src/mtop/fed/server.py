"""The coordinator over HTTP (Python standard library only; put it behind TLS in production).

Centres connect outward: they long-poll `GET /v1/next` for their next task and post answers to
`POST /v1/reply`, so no hospital has to open an inbound port. Every request carries the centre's
id and a bearer token; the coordinator stores only SHA-256 hashes of the tokens.

  GET  /              live dashboard (centres, current job, learning curve)
  GET  /v1/status     the dashboard's data (no patient-level content; masked traffic counts only)
  POST /v1/hello      a centre announces itself (label, identity fingerprint, protocol)
  GET  /v1/next       next task for this centre (long poll, 204 when there is none)
  POST /v1/reply      a centre's answer to a task
"""
from __future__ import annotations

import hashlib
import hmac
import json
import ssl
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import PROTOCOL
from .bins import fed_settings
from .coordinator import JobFailed


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Hub:
    """Coordinator side of the HTTP transport: one task queue per centre, replies matched by task id."""

    def __init__(self, token_hashes: dict[str, str], name: str = "MTOP coordinator"):
        self.token_hashes = dict(token_hashes)
        self.name = name
        self.cv = threading.Condition()
        self.queues: dict[str, deque] = {sid: deque() for sid in token_hashes}
        self.replies: dict[str, dict] = {}
        self.sites: dict[str, dict] = {}
        self.stats = {sid: {"down": 0, "up": 0, "messages": 0} for sid in token_hashes}
        self.status: dict = {"phase": "waiting for centres", "job": None, "curve": [], "jobs": []}
        self.timeout = float(fed_settings()["network"]["round_timeout_s"])

    # -------------------------------------------------------------- transport interface
    def ids(self) -> list[str]:
        now = time.time()
        with self.cv:
            return [sid for sid, s in sorted(self.sites.items()) if now - s["last"] < 90]

    def exchange(self, sids, msg: dict) -> dict:
        tid = uuid.uuid4().hex
        raw = json.dumps(msg, separators=(",", ":"))
        deadline = time.time() + self.timeout
        with self.cv:
            self.replies[tid] = {}
            for sid in sids:
                if sid not in self.queues:
                    raise JobFailed(f"unknown centre {sid}")
                self.queues[sid].append((tid, raw))
                self.stats[sid]["down"] += len(raw)
                self.stats[sid]["messages"] += 1
            self.cv.notify_all()
            while len(self.replies[tid]) < len(sids):
                left = deadline - time.time()
                if left <= 0:
                    missing = sorted(set(sids) - set(self.replies[tid]))
                    self.replies.pop(tid, None)
                    raise JobFailed(f"no answer from {missing}")
                self.cv.wait(left)
            return self.replies.pop(tid)

    # -------------------------------------------------------------- HTTP side
    def auth(self, sid: str | None, token: str | None) -> bool:
        if not sid or not token or sid not in self.token_hashes:
            return False
        return hmac.compare_digest(self.token_hashes[sid], token_hash(token))

    def hello(self, sid: str, info: dict) -> None:
        with self.cv:
            self.sites[sid] = {"label": str(info.get("label", sid))[:60], "fingerprint": str(info.get("fingerprint", ""))[:40],
                               "protocol": info.get("protocol"), "last": time.time(), "since": time.time()}
            self.cv.notify_all()

    def next_task(self, sid: str, wait: float):
        end = time.time() + wait
        with self.cv:
            if sid in self.sites:
                self.sites[sid]["last"] = time.time()
            while not self.queues[sid]:
                left = end - time.time()
                if left <= 0:
                    return None
                self.cv.wait(left)
            if sid in self.sites:
                self.sites[sid]["last"] = time.time()
            return self.queues[sid].popleft()

    def put_reply(self, sid: str, tid: str, reply: dict, size: int) -> None:
        with self.cv:
            if tid in self.replies:
                self.replies[tid][sid] = reply
            self.stats[sid]["up"] += size
            if sid in self.sites:
                self.sites[sid]["last"] = time.time()
            self.cv.notify_all()

    def wait_for(self, n: int, timeout: float = 3600) -> list[str]:
        end = time.time() + timeout
        with self.cv:
            while len([s for s in self.sites.values() if time.time() - s["last"] < 90]) < n:
                left = end - time.time()
                if left <= 0:
                    raise JobFailed(f"fewer than {n} centres connected")
                self.cv.wait(min(left, 1.0))
        return self.ids()

    def snapshot(self) -> dict:
        now = time.time()
        with self.cv:
            sites = [{"id": sid, "label": s["label"], "fingerprint": s["fingerprint"], "online": now - s["last"] < 90,
                      "down": self.stats[sid]["down"], "up": self.stats[sid]["up"], "messages": self.stats[sid]["messages"]}
                     for sid, s in sorted(self.sites.items())]
            return {"name": self.name, "protocol": PROTOCOL, "sites": sites, **self.status}


DASHBOARD = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTOP network</title><style>
:root{--blue:#1565d7;--ink:#0f1a2f;--muted:#5b6b85;--line:#dde4ee;--paper:#f4f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}p{margin:0;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:8px 6px;border-top:1px solid var(--line);font-size:14px}
th{color:var(--muted);font-weight:600;border-top:0}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:#c5cedb}
.on{background:#1baf7a}code{font-size:12px;color:var(--muted)}svg{width:100%;height:160px}
</style></head><body><main><h1>MTOP network coordinator</h1><p id="sub">Waiting…</p>
<div class="grid"><div class="card"><div class="k">Centres online</div><div class="v" id="n">0</div></div>
<div class="card"><div class="k">Phase</div><div class="v" id="phase" style="font-size:18px">–</div></div>
<div class="card"><div class="k">Trees</div><div class="v" id="rounds">–</div></div></div>
<div class="card"><div class="k">Training loss (network total, per tree)</div><svg id="curve" viewBox="0 0 600 160" preserveAspectRatio="none"></svg></div>
<div class="card" style="margin-top:12px"><table><thead><tr><th>Centre</th><th>Identity</th><th>Sent to centre</th><th>Received (masked)</th></tr></thead><tbody id="sites"></tbody></table></div>
<p style="margin-top:14px">The coordinator sees only the sum over all centres. Each centre's own answer is masked and looks like random numbers.</p>
</main><script>
const kb=(b)=>b>1e6?(b/1e6).toFixed(1)+" MB":(b/1e3).toFixed(0)+" kB";
async function tick(){try{const s=await (await fetch("v1/status",{cache:"no-store"})).json();
document.getElementById("sub").textContent=s.name+" · "+s.protocol;
document.getElementById("n").textContent=s.sites.filter(x=>x.online).length;
document.getElementById("phase").textContent=s.phase;document.getElementById("rounds").textContent=s.job?`${s.job.tree} / ${s.job.trees}`:"–";
document.getElementById("sites").innerHTML=s.sites.map(x=>`<tr><td><span class="dot ${x.online?"on":""}"></span>${x.label}</td><td><code>${x.fingerprint}</code></td><td>${kb(x.down)}</td><td>${kb(x.up)}</td></tr>`).join("");
const c=s.curve||[];const svg=document.getElementById("curve");if(c.length>1){const lo=Math.min(...c),hi=Math.max(...c);
const pts=c.map((v,i)=>`${(i/(c.length-1))*600},${150-((v-lo)/((hi-lo)||1))*140}`).join(" ");
svg.innerHTML=`<polyline points="${pts}" fill="none" stroke="#1565d7" stroke-width="2.5" vector-effect="non-scaling-stroke"/>`;}}catch(e){}
setTimeout(tick,1000)}tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    hub: Hub = None  # set by serve()
    server_version = "mtop-fed"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes = b"", ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode())

    def _site(self) -> str | None:
        sid = self.headers.get("X-Site")
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else None
        return sid if self.hub.auth(sid, token) else None

    def _body(self) -> tuple[dict, int]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 64 * 1024 * 1024:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8")), n

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, DASHBOARD.encode(), "text/html; charset=utf-8")
        if url.path == "/v1/status":
            return self._json(200, self.hub.snapshot())
        if url.path == "/v1/next":
            sid = self._site()
            if sid is None:
                return self._json(401, {"error": "unknown centre or wrong token"})
            wait = min(float(parse_qs(url.query).get("wait", ["25"])[0]), 55.0)
            item = self.hub.next_task(sid, wait)
            if item is None:
                return self._send(204)
            tid, raw = item
            return self._send(200, ('{"id":"%s","msg":' % tid + raw + "}").encode())
        return self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        sid = self._site()
        if sid is None:
            return self._json(401, {"error": "unknown centre or wrong token"})
        try:
            body, size = self._body()
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad request"})
        if url.path == "/v1/hello":
            if body.get("protocol") != PROTOCOL:
                return self._json(409, {"error": f"coordinator speaks {PROTOCOL}"})
            self.hub.hello(sid, body)
            return self._json(200, {"ok": True, "protocol": PROTOCOL, "name": self.hub.name})
        if url.path == "/v1/reply":
            self.hub.put_reply(sid, str(body.get("id")), body.get("reply") or {}, size)
            return self._json(200, {"ok": True})
        return self._json(404, {"error": "not found"})


def serve(hub: Hub, host: str = "127.0.0.1", port: int = 8750, cert: str | None = None, key: str | None = None):
    """Start the coordinator's HTTP server in a background thread; returns (server, thread)."""
    handler = type("BoundHandler", (Handler,), {"hub": hub})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    if cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(cert, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    th = threading.Thread(target=httpd.serve_forever, name="mtop-coordinator", daemon=True)
    th.start()
    return httpd, th
