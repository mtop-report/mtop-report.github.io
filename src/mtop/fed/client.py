"""A centre's node over HTTP, plus its local console.

The node only makes outgoing requests to the coordinator (long polling), so the hospital needs no
open port. The console is a small page on 127.0.0.1 inside the hospital where a person sees the
local data summary, approves or declines each job and reads the audit log of everything sent.
"""
from __future__ import annotations

import html
import json
import secrets
import ssl
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import PROTOCOL
from .site import SiteNode


class SiteClient:
    def __init__(self, node: SiteNode, url: str, token: str, cafile: str | None = None, wait: float = 25.0):
        self.node, self.url, self.token, self.wait = node, url.rstrip("/"), token, wait
        self.ctx = ssl.create_default_context(cafile=cafile) if self.url.startswith("https") else None
        self.stop = threading.Event()
        self.connected = False
        self.error: str | None = None

    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 60.0):
        data = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Content-Type": "application/json", "X-Site": self.node.site_id, "Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)

    def hello(self) -> None:
        info = self.node.handle({"type": "hello"})
        code, body = self._req("POST", "/v1/hello", {"label": info["label"], "fingerprint": info["fingerprint"],
                                                     "protocol": PROTOCOL})
        if code != 200:
            raise RuntimeError(f"coordinator refused the connection: {body}")
        self.connected = True

    def run(self) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            try:
                if not self.connected:
                    self.hello()
                code, body = self._req("GET", f"/v1/next?wait={self.wait:g}", timeout=self.wait + 30)
                backoff = 1.0
                if code == 204 or body is None:
                    continue
                reply = self.node.handle(body["msg"])
                self._req("POST", "/v1/reply", {"id": body["id"], "reply": reply})
            except urllib.error.HTTPError as e:
                self.error = f"HTTP {e.code}"
                self.connected = False
                if e.code in (401, 409):
                    self.stop.set()
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                self.error = str(e)
                self.connected = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def start(self) -> threading.Thread:
        th = threading.Thread(target=self.run, name=f"mtop-site-{self.node.site_id}", daemon=True)
        th.start()
        return th


# --------------------------------------------------------------------------- console
class Approvals:
    """Blocks a job offer until a person decides in the console (or the wait runs out: decline)."""

    def __init__(self, wait_s: float = 900):
        self.wait_s = wait_s
        self.cv = threading.Condition()
        self.waiting: dict[str, dict] = {}
        self.decided: dict[str, bool] = {}

    def __call__(self, info: dict) -> bool:
        job = info["job"]
        with self.cv:
            self.waiting[job] = info
            self.cv.notify_all()
            end = time.time() + self.wait_s
            while job not in self.decided and time.time() < end:
                self.cv.wait(max(0.0, end - time.time()))
            self.waiting.pop(job, None)
            return self.decided.pop(job, False)

    def decide(self, job: str, ok: bool) -> None:
        with self.cv:
            if job in self.waiting:
                self.decided[job] = ok
                self.cv.notify_all()


CONSOLE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MTOP site node</title><style>
:root{--blue:#1565d7;--ink:#0f1a2f;--muted:#5b6b85;--line:#dde4ee;--paper:#f4f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 2px}h2{font-size:16px;margin:0 0 8px}
p{margin:0;color:var(--muted)}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-top:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:7px 6px;border-top:1px solid var(--line);font-size:13px;vertical-align:top}
th{color:var(--muted);font-weight:600;border-top:0}button{font:inherit;border-radius:10px;padding:8px 14px;border:1px solid var(--line);background:#fff;cursor:pointer}
button.ok{background:var(--blue);border-color:var(--blue);color:#fff}code{font-size:12px;color:var(--muted)}.job{border:1px solid var(--blue);border-radius:12px;padding:12px;margin-top:8px}
</style></head><body><main><h1>MTOP site node · {{label}}</h1><p>Identity <code>{{fp}}</code> · connected to <code>{{url}}</code></p>
<div class="card"><h2>Local data (stays here)</h2><div id="data"></div></div>
<div class="card"><h2>Requests waiting for approval</h2><div id="pending"><p>None.</p></div></div>
<div class="card"><h2>Audit log: everything this node sent</h2><table><thead><tr><th>Time</th><th>Job</th><th>Round</th><th>What</th><th>Size</th><th>SHA-256</th></tr></thead><tbody id="log"></tbody></table></div>
</main><script>
const T="{{csrf}}";const KIND={gbdt:"Train the shared boosted-tree model",logreg:"Train the shared regression model",eval:"Validate a shared model"};
const STAGE={admission:"admission",procedure:"end of thrombectomy",h24:"24 hours",d7:"day 7"};const TARGET={good_outcome:"mRS 0–2 at 90 days",death90:"death by 90 days"};
const OP={init:"patient and outcome counts",hist:"value-band sums",moments:"means and spreads",lr:"regression sums",eval:"validation counts",accept:"accepted job",start:"keys checked, job started",finish:"job finished",abort:"job stopped"};
const esc=(s)=>String(s??"").replace(/[&<>"]/g,(c)=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function decide(job,ok){await fetch("decide",{method:"POST",headers:{"Content-Type":"application/json","X-Console":T},body:JSON.stringify({job,ok})});tick()}
async function tick(){try{const s=await (await fetch("status",{cache:"no-store",headers:{"X-Console":T}})).json();
document.getElementById("data").innerHTML=`<p>${s.data.rows} patients in the export · eligible for the admission model: ${s.data.stages.admission.good_outcome}</p>`;
document.getElementById("pending").innerHTML=s.pending.length?s.pending.map(j=>`<div class="job"><b>${esc(KIND[j.kind]||j.kind)}</b> · ${esc(STAGE[j.spec.stage]||j.spec.stage)} · ${esc(TARGET[j.spec.target]||j.spec.target)}<br>${j.patients} of your patients · at most ${j.budget} rounds of masked sums · with ${j.participants.length} centres (${j.participants.map(esc).join(", ")})
<div style="margin-top:8px;display:flex;gap:8px"><button class="ok" onclick='decide(${JSON.stringify(j.job)},true)'>Approve</button><button onclick='decide(${JSON.stringify(j.job)},false)'>Decline</button></div></div>`).join(""):"<p>None.</p>";
document.getElementById("log").innerHTML=s.audit.slice(-60).reverse().map(a=>`<tr><td>${esc(a.time.slice(11))}</td><td><code>${esc(a.job)}</code></td><td>${a.round??""}</td><td>${esc(OP[a.op]||a.op)}${a.round?" (masked)":""} ${esc(a.note??"")}</td><td>${a.bytes?a.bytes.toLocaleString()+" B":""}</td><td><code>${esc(a.sha256??"")}</code></td></tr>`).join("");}catch(e){}
setTimeout(tick,1500)}tick();
</script></body></html>"""


def console(node: SiteNode, approvals: Approvals | None, url: str, port: int = 8751):
    """Serve the console on 127.0.0.1 only. Returns the server (already running in a thread)."""
    csrf = secrets.token_urlsafe(16)
    page = CONSOLE
    for key, val in {"label": html.escape(node.label), "fp": node.identity.fingerprint, "url": html.escape(url), "csrf": csrf}.items():
        page = page.replace("{{" + key + "}}", val)
    summary = node.describe()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if path == "/status" and self.headers.get("X-Console") == csrf:
                pend = list(approvals.waiting.values()) if approvals else []
                return self._send(200, json.dumps({"data": summary, "pending": pend, "audit": node.audit[-200:]}).encode())
            return self._send(404, b"{}")

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/decide" or self.headers.get("X-Console") != csrf or approvals is None:
                return self._send(403, b"{}")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            approvals.decide(str(body.get("job")), bool(body.get("ok")))
            return self._send(200, b'{"ok":true}')

    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="mtop-console", daemon=True).start()
    return httpd
