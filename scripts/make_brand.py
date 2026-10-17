"""Generate the MTOP logo and app icons from the bundled Instrument Sans font.

The wordmark is drawn from real glyph outlines (so it does not depend on the font loading),
with the O replaced by the MTOP mark: a ring holding a filled lower half whose upper edge is a
rising curve, like a line on a chart (the outcome we estimate, read from data).

    pip install "fonttools[woff]" playwright
    python scripts/make_brand.py            # writes web/assets/{logo.svg, mark.svg, *.png}

Run it only when the brand changes; the outputs are committed.
"""
from __future__ import annotations

import asyncio
import math
import pathlib

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web" / "assets"
FONT = ASSETS / "fonts" / "instrument-sans-latin.woff2"

INK = "#0f1a2f"
BLUE = "#1565d7"
WGHT, WDTH = 640, 100
CAP, TOP = 720, 730          # cap height and round overshoot in font units
RING_R, RING_W = 370, 112    # ring: outer radius and stroke (font units)
FILL_R = 208                 # inner filled disc, leaving a clear gap inside the ring


CURVE = (0.42, -0.18, 2.6)   # height at the left and right edge (radii, positive = down), bend


def wave(u: float) -> float:
    """Upper edge of the fill across the disc (u from -1 to 1), in radii, positive = down.
    One smooth rise to the right, steep at first and then levelling off, like a curve on a chart
    (a learning or ROC curve) rather than a wave."""
    y0, y1, k = CURVE
    t = (u + 1) / 2
    return y0 + (y1 - y0) * (1 - math.exp(-k * t)) / (1 - math.exp(-k))


def _cross(f, a: float, b: float) -> float:
    """Root of f between a and b (f(a) and f(b) of opposite sign), by bisection."""
    fa = f(a)
    for _ in range(80):
        m = (a + b) / 2
        if (f(m) > 0) == (fa > 0):
            a, fa = m, f(m)
        else:
            b = m
    return (a + b) / 2


def fill_path(cx: float, cy: float, r: float, n: int = 120) -> str:
    """The part of the disc (cx, cy, r) that lies below the curve, as one closed path: along the
    curve between the two points where it meets the circle, then back along the circle's arc, so
    both ends close smoothly (no vertical cut where the curve leaves the disc)."""
    inside = lambda u: abs(wave(u)) < math.sqrt(max(0.0, 1 - u * u))  # noqa: E731
    # left end: the curve enters through the lower arc; right end: it leaves through the arc as well
    lo = _cross(lambda u: wave(u) - math.sqrt(max(0.0, 1 - u * u)) * (1 if wave(u) > 0 else -1), -1.0, 0.0)
    hi = _cross(lambda u: abs(wave(u)) - math.sqrt(max(0.0, 1 - u * u)), 0.0, 1.0) if not inside(1.0) else 1.0
    pts = [(lo + (hi - lo) * i / n, None) for i in range(n + 1)]
    d = "".join(("M" if i == 0 else "L") + "%.2f,%.2f" % (cx + u * r, cy + wave(u) * r) for i, (u, _) in enumerate(pts))
    a0 = math.atan2(wave(hi), hi)               # angle of the right end (SVG y down)
    a1 = math.atan2(wave(lo), lo)               # angle of the left end
    if a1 < a0:
        a1 += 2 * math.pi
    for i in range(1, n + 1):                   # arc clockwise on screen: right end, bottom, left end
        a = a0 + (a1 - a0) * i / n
        d += "L%.2f,%.2f" % (cx + r * math.cos(a), cy + r * math.sin(a))
    return d + "Z"


def glyphs():
    font = instantiateVariableFont(TTFont(FONT), {"wght": WGHT, "wdth": WDTH})
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    out = {}
    for ch in "MTP":
        g = gs[cmap[ord(ch)]]
        bp = BoundsPen(gs)
        g.draw(bp)
        out[ch] = (g, bp.bounds)
    return gs, out


def path_for(gs, glyph, dx):
    pen = SVGPathPen(gs)
    # font units, y up -> SVG, y down; baseline at y = TOP
    glyph.draw(TransformPen(pen, (1, 0, 0, -1, dx, TOP)))
    return pen.getCommands()


def mark(cx, cy, r_out, width, fill_r, color):
    r = r_out - width / 2
    return (f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="none" stroke="{color}" stroke-width="{width:g}"/>'
            f'<path fill="{color}" d="{fill_path(cx, cy, fill_r)}"/>')


def wordmark():
    gs, G = glyphs()
    (gm, bm), (gt, bt), (gp, bp) = G["M"], G["T"], G["P"]
    gap_mt, gap_to, gap_op = 64, 46, 78
    x = 0.0
    parts = []
    parts.append(f'<path fill="{INK}" d="{path_for(gs, gm, x - bm[0])}"/>')
    x += bm[2] - bm[0] + gap_mt
    parts.append(f'<path fill="{INK}" d="{path_for(gs, gt, x - bt[0])}"/>')
    x += bt[2] - bt[0] + gap_to
    cy = TOP - CAP / 2          # optical centre of the caps
    parts.append(mark(x + RING_R, cy, RING_R, RING_W, FILL_R, BLUE))
    x += 2 * RING_R + gap_op
    parts.append(f'<path fill="{INK}" d="{path_for(gs, gp, x - bp[0])}"/>')
    x += bp[2] - bp[0]
    w, h = round(x), TOP + 10
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="MTOP">'
            f'<title>MTOP</title>{"".join(parts)}</svg>\n')


def tile(size=512, shape="rounded"):
    """App icon: white mark on the brand blue.

    rounded   tile with rounded corners (favicon, manifest icons)
    square    full bleed (apple-touch-icon; iOS applies its own mask)
    maskable  full bleed with the mark inside the 80 % safe zone
    """
    c = size / 2
    r_out = size * (0.25 if shape == "maskable" else 0.31)
    w = r_out * (RING_W / RING_R)
    fr = r_out * (FILL_R / RING_R)
    rx = f' rx="{size * 0.225:g}"' if shape == "rounded" else ""
    bg = f'<rect width="{size}" height="{size}"{rx} fill="{BLUE}"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">{bg}'
            f'{mark(c, c, r_out, w, fr, "#ffffff")}</svg>\n')


async def rasterise(svg: str, out: pathlib.Path, px: int):
    from playwright.async_api import async_playwright
    sized = svg.replace("<svg ", '<svg width="%d" height="%d" ' % (px, px), 1)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": px, "height": px}, device_scale_factor=1)
        await pg.set_content('<html><body style="margin:0;background:transparent">'
                             '<div style="width:%dpx;height:%dpx">%s</div></body></html>' % (px, px, sized))
        await pg.locator("div").screenshot(path=str(out), omit_background=True)
        await b.close()


def main():
    (ASSETS / "logo.svg").write_text(wordmark(), encoding="utf-8")
    (ASSETS / "mark.svg").write_text(tile(), encoding="utf-8")
    jobs = [(tile(), "icon-192.png", 192), (tile(), "icon-512.png", 512), (tile(shape="square"), "apple-touch-icon.png", 180),
            (tile(shape="maskable"), "icon-maskable-512.png", 512)]
    for svg, name, px in jobs:
        asyncio.run(rasterise(svg, ASSETS / name, px))
    print("wrote", ", ".join(["logo.svg", "mark.svg"] + [j[1] for j in jobs]))


if __name__ == "__main__":
    main()
