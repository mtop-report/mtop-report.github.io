"""Build the self-contained web app from web/ and the public artifacts.

site/index.html      full HTML document for GitHub Pages (or any static host, or opening locally);
                     installable and usable offline (web app manifest + service worker)
site/artifact.html   the same page without the document skeleton (for hosts that add their own)
"""
from __future__ import annotations

import json
import shutil

from . import __version__
from .config import settings
from .paths import PUBLIC, SITE, WEB

JS_FILES = ["i18n.js", "model.js", "models_extra.js", "metrics.js", "charts.js", "app.js",
            "view_adaptive.js", "view_network.js", "view_roadmap.js", "boot.js"]
ASSETS = ["icon-192.png", "icon-512.png", "icon-maskable-512.png"]

PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#111820">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" type="image/png" href="icon-192.png">
<link rel="apple-touch-icon" href="icon-192.png">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}[hidden]{display:none!important}img{max-width:100%}</style>
"""

SERVICE_WORKER = """/* MTOP offline cache: network first, cached copy when offline. */
const CACHE = "mtop-%s";
const ASSETS = ["./", "./index.html", "./manifest.webmanifest", "./icon-192.png", "./icon-512.png"];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(fetch(req).then((res) => {
    const copy = res.clone();
    caches.open(CACHE).then((c) => c.put(req, copy));
    return res;
  }).catch(() => caches.match(req).then((r) => r || caches.match("./index.html"))));
});
"""


def _safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _load(name, default=None):
    f = PUBLIC / name
    if not f.exists():
        return default
    return json.loads(f.read_text(encoding="utf-8")) if name.endswith(".json") else f.read_text(encoding="utf-8")


def build_site():
    SITE.mkdir(parents=True, exist_ok=True)
    site_cfg = settings().get("site", {})
    data = {
        "bundle": _load("model_bundle.json"),
        "validation": _load("validation.json"),
        "explorer": _load("explorer.json"),
        "extras": _load("extras.json", {}),
        "adaptive": _load("adaptive.json"),
        "subgroups": _load("subgroups.json", []),
        "demo_csv": _load("demo_synthetic.csv", ""),
        "qr_svg": (WEB / "assets" / "qr.svg").read_text(encoding="utf-8") if (WEB / "assets" / "qr.svg").exists() else "",
        "site": site_cfg,
    }
    tpl = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    js = "\n".join((WEB / f).read_text(encoding="utf-8") for f in JS_FILES)
    if "</script" in js.lower():
        raise ValueError("inline JS must not contain a closing script tag")
    body = tpl.replace("/*CSS*/", css).replace("/*JS*/", js).replace("/*DATA*/", _safe_json(data))
    (SITE / "artifact.html").write_text(body, encoding="utf-8")
    title_end = body.index("</title>") + len("</title>")
    head_part, rest = body[:title_end], body[title_end:]
    # move the <meta>/<link>/<style> block into <head>, keep the app markup in <body>
    split = rest.index('<div id="app">')
    full = PAGE_HEAD + head_part + rest[:split] + "</head>\n<body>\n" + rest[split:] + "\n</body>\n</html>\n"
    out = SITE / "index.html"
    out.write_text(full, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    # installable, offline-capable web app
    for a in ASSETS:
        if (WEB / "assets" / a).exists():
            shutil.copyfile(WEB / "assets" / a, SITE / a)
    manifest = {
        "name": site_cfg.get("title", "MTOP Outcome Explorer"), "short_name": "MTOP",
        "description": "Staged, explainable estimate of 90-day outcome after mechanical thrombectomy",
        "start_url": "./", "scope": "./", "display": "standalone",
        "background_color": "#eef1f3", "theme_color": "#111820",
        "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"},
                  {"src": "icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}],
    }
    (SITE / "manifest.webmanifest").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (SITE / "sw.js").write_text(SERVICE_WORKER % f"{__version__}-{data['bundle'].get('fingerprint', '')[:8]}", encoding="utf-8")
    return out
