"""Build the self-contained web app from web/ and the public artifacts.

site/index.html     full HTML document for GitHub Pages (or any static host, or opening locally)
site/artifact.html  the same page without the document skeleton (for hosts that add their own)
"""
from __future__ import annotations

import json

from .config import settings
from .paths import PUBLIC, SITE, WEB

JS_FILES = ["i18n.js", "model.js", "charts.js", "app.js"]

PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}[hidden]{display:none!important}img{max-width:100%}</style>
"""


def _safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_site():
    SITE.mkdir(parents=True, exist_ok=True)
    data = {
        "bundle": json.loads((PUBLIC / "model_bundle.json").read_text(encoding="utf-8")),
        "validation": json.loads((PUBLIC / "validation.json").read_text(encoding="utf-8")),
        "explorer": json.loads((PUBLIC / "explorer.json").read_text(encoding="utf-8")),
        "site": settings().get("site", {}),
        "subgroups": (json.loads((PUBLIC / "subgroups.json").read_text(encoding="utf-8"))
                      if (PUBLIC / "subgroups.json").exists() else []),
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
    return out
