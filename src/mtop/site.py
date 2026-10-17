"""Build the self-contained web app from web/ and the public artifacts.

site/index.html      full HTML document for GitHub Pages (or any static host, or opening locally);
                     installable and usable offline (web app manifest + service worker)
site/artifact.html   the same page without the document skeleton (for hosts that add their own)
"""
from __future__ import annotations

import base64
import json
import shutil

from . import __version__
from .config import settings
from .paths import PUBLIC, SITE, WEB

JS_FILES = ["i18n.js", "model.js", "models_extra.js", "metrics.js", "charts.js", "app.js",
            "view_adaptive.js", "view_network.js", "view_roadmap.js", "boot.js"]
ASSETS = ["icon-192.png", "icon-512.png", "icon-maskable-512.png", "apple-touch-icon.png", "mark.svg"]

# Instrument Sans (SIL OFL 1.1, web/assets/fonts/OFL.txt), self-hosted and inlined so the page makes
# no third-party requests and works offline. Two subsets: Latin and Latin Extended (Czech).
FONT_FACES = [
    ("instrument-sans-latin.woff2",
     "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,"
     "U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"),
    ("instrument-sans-latin-ext.woff2",
     "U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+0304,U+0308,U+0329,U+1D00-1DBF,"
     "U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20C0,U+2113,U+2C60-2C7F,U+A720-A7FF"),
]
THEME = "#ffffff"
PAPER = "#f4f7fb"

PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="%(theme)s">
<meta name="color-scheme" content="light">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" type="image/svg+xml" href="mark.svg">
<link rel="icon" type="image/png" sizes="192x192" href="icon-192.png">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}[hidden]{display:none!important}img{max-width:100%%}</style>
""" % {"theme": THEME}

SERVICE_WORKER = """/* MTOP offline cache: network first, cached copy when offline. */
const CACHE = "mtop-%s";
const ASSETS = ["./", "./index.html", "./manifest.webmanifest", "./icon-192.png", "./icon-512.png", "./mark.svg"];
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


def font_css() -> str:
    out = []
    for name, ranges in FONT_FACES:
        f = WEB / "assets" / "fonts" / name
        if not f.exists():
            continue
        b64 = base64.b64encode(f.read_bytes()).decode("ascii")
        out.append("@font-face{font-family:\"Instrument Sans\";font-style:normal;font-display:swap;"
                   "font-weight:400 700;font-stretch:75% 100%;"
                   f"src:url(data:font/woff2;base64,{b64}) format(\"woff2\");unicode-range:{ranges}}}")
    return "\n".join(out)


def build_site():
    SITE.mkdir(parents=True, exist_ok=True)
    site_cfg = settings().get("site", {})
    logo = WEB / "assets" / "logo.svg"
    data = {
        "bundle": _load("model_bundle.json"),
        "validation": _load("validation.json"),
        "explorer": _load("explorer.json"),
        "extras": _load("extras.json", {}),
        "adaptive": _load("adaptive.json"),
        "subgroups": _load("subgroups.json", []),
        "demo_csv": _load("demo_synthetic.csv", ""),
        "qr_svg": (WEB / "assets" / "qr.svg").read_text(encoding="utf-8") if (WEB / "assets" / "qr.svg").exists() else "",
        "logo_svg": logo.read_text(encoding="utf-8").strip() if logo.exists() else "",
        "site": site_cfg,
    }
    tpl = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    js = "\n".join((WEB / f).read_text(encoding="utf-8") for f in JS_FILES)
    if "</script" in js.lower():
        raise ValueError("inline JS must not contain a closing script tag")
    body = (tpl.replace("/*FONTS*/", font_css()).replace("/*CSS*/", css)
            .replace("/*JS*/", js).replace("/*DATA*/", _safe_json(data)))
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
        "background_color": PAPER, "theme_color": THEME,
        "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"},
                  {"src": "icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}],
    }
    (SITE / "manifest.webmanifest").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (SITE / "sw.js").write_text(SERVICE_WORKER % f"{__version__}-{data['bundle'].get('fingerprint', '')[:8]}", encoding="utf-8")
    return out
