/* MTOP charts: small, dependency-free SVG charts with hover tooltips, keyboard access,
 * and a table view for every figure. Colours come from the CSS tokens in styles.css. */
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function s(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const k in attrs || {}) {
      const v = attrs[k];
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return n;
  }
  function txt(parent, x, y, str, attrs) {
    const t = s("text", Object.assign({ x, y }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }

  function linear(d0, d1, r0, r1) {
    const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
    f.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * (d1 - d0);
    return f;
  }
  function niceStep(span, n) {
    const raw = span / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const err = raw / mag;
    return (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  }
  function ticks(lo, hi, n) {
    const st = niceStep(hi - lo, n);
    const out = [];
    for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9; v += st) { const r = +v.toFixed(10); out.push(r === 0 ? 0 : r); }
    return out;
  }

  // ------------------------------------------------------------------ tooltip
  const tip = {
    el: null,
    ensure() {
      if (!this.el) { this.el = h("div", { class: "tip", role: "status" }); document.body.appendChild(this.el); }
      return this.el;
    },
    show(head, rows, x, y) {
      const el = this.ensure();
      el.replaceChildren();
      if (head) el.appendChild(h("div", { class: "t-head", text: head }));
      for (const r of rows) {
        const k = h("span", { class: "k" });
        if (r.color) k.appendChild(h("i", { style: `background:${r.color}` }));
        k.appendChild(document.createTextNode(r.label || ""));
        el.appendChild(h("div", { class: "t-row" }, k, h("b", { text: r.value })));
      }
      el.style.display = "block";
      const w = el.offsetWidth, hh = el.offsetHeight;
      let left = x + 14, top = y + 14;
      if (left + w > window.innerWidth - 8) left = x - w - 14;
      if (top + hh > window.innerHeight - 8) top = y - hh - 14;
      el.style.left = Math.max(8, left) + "px";
      el.style.top = Math.max(8, top) + "px";
    },
    hide() { if (this.el) this.el.style.display = "none"; },
  };

  // ------------------------------------------------------------------ figure shell
  /**
   * figure(container, {title, sub, legend, table:{columns, rows}, labels:{table, chart}})
   * returns {body, fig}. The table toggle swaps the drawing for an equivalent table.
   */
  function figure(container, o) {
    const fig = h("figure", { class: "chart" });
    const head = h("div", { class: "chart-head" });
    const titles = h("div", null, h("div", { class: "chart-title", text: o.title || "" }));
    if (o.sub) titles.appendChild(h("div", { class: "chart-sub", text: o.sub }));
    head.appendChild(titles);
    const body = h("div", { class: "chart-body" });
    const tableBox = h("div", { class: "table-scroll", hidden: true });
    if (o.table) {
      const btn = h("button", { class: "btn btn-ghost tbl-toggle", type: "button", "aria-pressed": "false", text: o.labels.table });
      btn.addEventListener("click", () => {
        const showTable = btn.getAttribute("aria-pressed") !== "true";
        btn.setAttribute("aria-pressed", String(showTable));
        btn.textContent = showTable ? o.labels.chart : o.labels.table;
        body.hidden = showTable;
        tableBox.hidden = !showTable;
        if (legendEl) legendEl.hidden = showTable;
      });
      head.appendChild(btn);
      tableBox.appendChild(dataTable(o.table.columns, o.table.rows));
    }
    fig.appendChild(head);
    let legendEl = null;
    if (o.legend && o.legend.length) {
      legendEl = h("div", { class: "legend" });
      for (const l of o.legend) {
        const key = h("span", { class: "key" }, h("i", { class: l.type || "", style: l.color ? `background:${l.color}` : null }), l.label);
        legendEl.appendChild(key);
      }
      fig.appendChild(legendEl);
    }
    fig.appendChild(body);
    fig.appendChild(tableBox);
    container.appendChild(fig);
    return { fig, body };
  }

  function dataTable(columns, rows) {
    const t = h("table");
    const thead = h("thead", null, h("tr", null, columns.map((c) => h("th", { scope: "col", text: c }))));
    const tb = h("tbody");
    for (const r of rows) tb.appendChild(h("tr", null, r.map((v, i) => (i === 0 ? h("th", { scope: "row", text: String(v) }) : h("td", { text: v === null || v === undefined ? "–" : String(v) })))));
    t.append(thead, tb);
    return t;
  }

  function responsive(body, draw) {
    let last = 0;
    const run = () => {
      const w = Math.floor(body.clientWidth);
      if (w > 0 && Math.abs(w - last) > 2) { last = w; draw(w); }
    };
    if (typeof ResizeObserver !== "undefined") {
      const ro = new ResizeObserver(() => run());
      ro.observe(body);
    } else {
      window.addEventListener("resize", run);
    }
    requestAnimationFrame(run);
    run();
  }

  // ------------------------------------------------------------------ line chart
  /**
   * lineChart(body, {
   *   height, x:{type:'cat', cats:[labels]} | {type:'lin', domain:[a,b], fmt, ticks},
   *   y:{domain:[a,b], fmt, title, ticks}, series:[{label, color, points:[{x,y,lo,hi,n}], dots, width, dash}],
   *   diagonal:bool, marker:{x,label}, endLabels:bool, tipFmt:(v)=>str, tipHead:(x)=>str })
   */
  function lineChart(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const H = o.height || 220;
      const m = { t: 12, r: o.endLabels ? 96 : 14, b: o.x.title ? 40 : 26, l: o.y.title ? 52 : 40 };
      if (W < 380 && o.endLabels) m.r = 14;
      const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const cat = o.x.type === "cat";
      const n = cat ? o.x.cats.length : 0;
      const xs = cat
        ? (i) => m.l + ((i + 0.5) * iw) / n
        : linear(o.x.domain[0], o.x.domain[1], m.l, m.l + iw);
      const ys = linear(o.y.domain[0], o.y.domain[1], m.t + ih, m.t);
      const g = s("g", null, svg);
      // grid + y ticks
      const yt = o.y.ticks || ticks(o.y.domain[0], o.y.domain[1], ih < 150 ? 3 : 5);
      for (const v of yt) {
        const y = ys(v);
        s("line", { class: "grid", x1: m.l, x2: m.l + iw, y1: y, y2: y }, g);
        txt(g, m.l - 6, y + 4, o.y.fmt ? o.y.fmt(v) : String(v), { "text-anchor": "end" });
      }
      s("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, g);
      // x ticks
      if (cat) {
        const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / 46))));
        const shown = [];
        for (let i = 0; i < n; i += every) shown.push(i);
        if (shown[shown.length - 1] !== n - 1) {
          if (n - 1 - shown[shown.length - 1] < Math.max(1, every * 0.75)) shown.pop();
          shown.push(n - 1);
        }
        for (const i of shown) txt(g, xs(i), m.t + ih + 16, o.x.cats[i], { "text-anchor": "middle" });
      } else {
        const xt = o.x.ticks || ticks(o.x.domain[0], o.x.domain[1], Math.max(2, Math.floor(iw / 70)));
        for (const v of xt) txt(g, xs(v), m.t + ih + 16, o.x.fmt ? o.x.fmt(v) : String(v), { "text-anchor": "middle" });
      }
      if (o.x.title) txt(g, m.l + iw / 2, H - 4, o.x.title, { "text-anchor": "middle", class: "lbl" });
      if (o.y.title) {
        const t = txt(g, 12, m.t + ih / 2, o.y.title, { "text-anchor": "middle", class: "lbl", transform: `rotate(-90 12 ${m.t + ih / 2})` });
        t.setAttribute("dy", "0");
      }
      if (o.diagonal) s("line", { class: "ref", x1: xs(o.x.domain[0]), y1: ys(o.y.domain[0]), x2: xs(o.x.domain[1]), y2: ys(o.y.domain[1]) }, g);
      if (o.zeroLine && o.y.domain[0] < 0) s("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: ys(0), y2: ys(0) }, g);
      if (o.marker) {
        const mx = xs(o.marker.x);
        s("line", { x1: mx, x2: mx, y1: m.t, y2: m.t + ih, style: "stroke: var(--ink-2); stroke-width:1; stroke-dasharray:3 3" }, g);
        if (o.marker.label) txt(g, Math.min(mx + 4, m.l + iw - 4), m.t + 10, o.marker.label, { class: "lbl", "text-anchor": mx + 90 > m.l + iw ? "end" : "start" });
      }
      const X = (p) => (cat ? xs(p.x) : xs(p.x));
      // CI whiskers / bands
      for (const se of o.series) {
        const off = se.offset || 0;
        for (const p of se.points) {
          if (p.lo === undefined || p.hi === undefined || p.y === null) continue;
          const x = X(p) + off;
          s("line", { x1: x, x2: x, y1: ys(p.lo), y2: ys(p.hi), style: `stroke:${se.color}; stroke-width:1.5; opacity:0.55` }, g);
        }
      }
      // soft area under single-measure trends
      for (const se of o.series) {
        if (!se.area) continue;
        const off = se.offset || 0;
        const pts = se.points.filter((p) => p.y !== null && p.y !== undefined && !Number.isNaN(p.y));
        if (pts.length < 2) continue;
        const base = ys(o.y.domain[0]).toFixed(1);
        let d = `M${(X(pts[0]) + off).toFixed(1)},${base}`;
        for (const p of pts) d += `L${(X(p) + off).toFixed(1)},${ys(p.y).toFixed(1)}`;
        d += `L${(X(pts[pts.length - 1]) + off).toFixed(1)},${base}Z`;
        s("path", { d, style: `fill:${se.color}; fill-opacity:0.08; stroke:none` }, g);
      }
      // lines
      for (const se of o.series) {
        const off = se.offset || 0;
        let d = "";
        let pen = false;
        for (const p of se.points) {
          if (p.y === null || p.y === undefined || Number.isNaN(p.y)) { pen = false; continue; }
          const cmd = pen ? "L" : "M";
          d += `${cmd}${(X(p) + off).toFixed(1)},${ys(p.y).toFixed(1)}`;
          pen = true;
        }
        if (se.noLine) continue;
        s("path", { d, fill: "none", style: `stroke:${se.color}; stroke-width:${se.width || 2}; stroke-linejoin:round; stroke-linecap:round;${se.dash ? "stroke-dasharray:4 3;" : ""}` }, g);
      }
      // emphasised last value
      for (const se of o.series) {
        if (!se.endDot || se.dots) continue;
        const pts = se.points.filter((p) => p.y !== null && p.y !== undefined && !Number.isNaN(p.y));
        const p = pts[pts.length - 1];
        if (p) s("circle", { cx: X(p) + (se.offset || 0), cy: ys(p.y), r: 3.5, style: `fill:${se.color}; stroke: var(--surface); stroke-width:2` }, g);
      }
      // dots
      for (const se of o.series) {
        if (!se.dots) continue;
        const off = se.offset || 0;
        for (const p of se.points) {
          if (p.y === null || p.y === undefined) continue;
          s("circle", { cx: X(p) + off, cy: ys(p.y), r: 4, style: `fill:${se.color}; stroke: var(--surface); stroke-width:2` }, g);
        }
      }
      // end labels
      if (o.endLabels && W >= 380) {
        const ends = o.series.map((se) => {
          const pts = se.points.filter((p) => p.y !== null && p.y !== undefined);
          const p = pts[pts.length - 1];
          return p ? { se, y: ys(p.y), x: X(p) } : null;
        }).filter(Boolean).sort((a, b) => a.y - b.y);
        let collide = false;
        for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) collide = true;
        if (!collide) for (const e of ends) txt(g, e.x + 8, e.y + 4, e.se.label, { class: "lbl" });
      }
      // crosshair / hover
      const xsList = cat ? o.x.cats.map((_, i) => i) : Array.from(new Set(o.series.flatMap((se) => se.points.map((p) => p.x)))).sort((a, b) => a - b);
      const cross = s("line", { class: "cross", y1: m.t, y2: m.t + ih, x1: -10, x2: -10, visibility: "hidden" }, g);
      const hit = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", tabindex: 0, style: "cursor:crosshair; outline:none" }, svg);
      let idx = -1;
      const showAt = (i, cx, cy) => {
        idx = i;
        const xv = xsList[i];
        const px = cat ? xs(xv) : xs(xv);
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
        const rows = [];
        for (const se of o.series) {
          const p = se.points.find((q) => q.x === xv);
          if (!p || p.y === null || p.y === undefined) continue;
          rows.push({ label: se.label, color: se.color, value: (o.tipFmt || String)(p.y, p, se) });
        }
        const head = o.tipHead ? o.tipHead(xv) : cat ? o.x.cats[xv] : String(xv);
        const r = svg.getBoundingClientRect();
        tip.show(head, rows, cx !== undefined ? cx : r.left + px, cy !== undefined ? cy : r.top + m.t);
      };
      const nearest = (px) => {
        let best = 0, bd = Infinity;
        xsList.forEach((xv, i) => { const d = Math.abs((cat ? xs(xv) : xs(xv)) - px); if (d < bd) { bd = d; best = i; } });
        return best;
      };
      hit.addEventListener("pointermove", (ev) => {
        const r = svg.getBoundingClientRect();
        showAt(nearest(ev.clientX - r.left), ev.clientX, ev.clientY);
      });
      hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
      hit.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
      hit.addEventListener("keydown", (ev) => {
        if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
        ev.preventDefault();
        const i = Math.max(0, Math.min(xsList.length - 1, (idx < 0 ? 0 : idx) + (ev.key === "ArrowRight" ? 1 : -1)));
        showAt(i);
      });
      hit.addEventListener("focus", () => showAt(idx < 0 ? xsList.length - 1 : idx));
    });
  }

  // ------------------------------------------------------------------ 100% stacked rows (Grotta bars)
  /** grotta(body, {rows:[{label, n, segs:[number %], suppressed, note}], colors:[css], segLabels:[str], nLabel}) */
  let clipSeq = 0;
  function grotta(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const narrow = W < 520;
      const labelW = narrow ? 0 : Math.min(190, Math.round(W * 0.28));
      const nW = 56;
      const rowH = narrow ? 46 : 32;
      const barH = 20;
      const H = o.rows.length * rowH + 8;
      const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
      const defs = s("defs", null, svg);
      const x0 = labelW + (labelW ? 10 : 0), x1 = W - nW;
      const bw = x1 - x0;
      o.rows.forEach((r, i) => {
        const yTop = i * rowH + (narrow ? 18 : 6);
        const labY = narrow ? i * rowH + 13 : yTop + barH / 2 + 4;
        txt(svg, narrow ? 0 : 0, labY, r.label, { class: "lbl" });
        if (r.suppressed || !r.segs) {
          txt(svg, x0, narrow ? yTop + 13 : labY, o.suppressedText || "", {});
          return;
        }
        // round the two outer ends of each 100% bar
        const cid = `mtop-gclip-${++clipSeq}`;
        s("rect", { x: x0, y: yTop, width: Math.max(0, bw - 2), height: barH, rx: 5 }, s("clipPath", { id: cid }, defs));
        const row = s("g", { "clip-path": `url(#${cid})` }, svg);
        let x = x0;
        r.segs.forEach((v, k) => {
          const w = (v / 100) * bw;
          if (w <= 0) return;
          const rx = x, rw = Math.max(0, w - 2);
          const rect = s("rect", { x: rx, y: yTop, width: rw, height: barH, rx: 0, style: `fill:${o.colors[k]}` }, row);
          rect.setAttribute("tabindex", "0");
          const show = (cx, cy) => tip.show(r.label, [{ label: o.segLabels[k], color: o.colors[k], value: `${o.fmt(v)}` }], cx, cy);
          rect.addEventListener("pointermove", (ev) => { rect.style.opacity = "0.85"; show(ev.clientX, ev.clientY); });
          rect.addEventListener("pointerleave", () => { rect.style.opacity = ""; tip.hide(); });
          rect.addEventListener("focus", () => { const b = rect.getBoundingClientRect(); show(b.left + b.width / 2, b.top); });
          rect.addEventListener("blur", () => tip.hide());
          if (rw >= 30) {
            txt(svg, rx + rw / 2, yTop + barH / 2 + 4, `${Math.round(v)}`, {
              "text-anchor": "middle", style: `fill:${inkFor(rect, o.colors[k])}; font-size:11px; font-weight:600; pointer-events:none`,
            });
          }
          x += w;
        });
        txt(svg, W, narrow ? yTop + 13 : labY, `${o.nLabel || "n"} ${r.n}`, { "text-anchor": "end" });
      });
    });
  }

  /** Text colour for a label on a coloured fill: the fill's own "-ink" token when it has one (so it follows
   *  the light or dark theme), otherwise white or ink by the fill's luminance. */
  function inkFor(node, color) {
    const v = /^var\((--[\w-]+)\)$/.exec(color || "");
    if (v && getComputedStyle(document.documentElement).getPropertyValue(`${v[1]}-ink`).trim()) return `var(${v[1]}-ink)`;
    const c = getComputedStyle(node).fill || "";
    const m = c.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)/);
    if (!m) return "#ffffff";
    const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    const L = 0.2126 * lin(+m[1]) + 0.7152 * lin(+m[2]) + 0.0722 * lin(+m[3]);
    return L > 0.36 ? "#0f1a2f" : "#ffffff";
  }

  // ------------------------------------------------------------------ horizontal bars (single series)
  /** hbars(body, {rows:[{label, value, display}], color, max}) */
  function hbars(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const rowH = 24;
      const H = o.rows.length * rowH + 6;
      const labelW = Math.min(200, Math.round(W * 0.42));
      const valW = 46;
      const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
      const x0 = labelW + 8, x1 = W - valW;
      const max = o.max || Math.max(...o.rows.map((r) => r.value)) || 1;
      o.rows.forEach((r, i) => {
        const y = i * rowH + 4;
        txt(svg, labelW, y + 13, r.label, { "text-anchor": "end", class: "lbl" });
        const w = Math.max(1, ((x1 - x0) * r.value) / max);
        const bar = s("path", { d: roundedRight(x0, y + 3, w, 12, 4), style: `fill:${o.color}` }, svg);
        bar.setAttribute("tabindex", "0");
        const show = (cx, cy) => tip.show(r.label, [{ label: o.valueLabel || "", color: o.color, value: r.display }], cx, cy);
        bar.addEventListener("pointermove", (ev) => show(ev.clientX, ev.clientY));
        bar.addEventListener("pointerleave", () => tip.hide());
        bar.addEventListener("focus", () => { const b = bar.getBoundingClientRect(); show(b.right, b.top); });
        bar.addEventListener("blur", () => tip.hide());
        txt(svg, x0 + w + 6, y + 13, r.display, { class: "lbl" });
      });
    });
  }
  function roundedRight(x, y, w, hgt, r) {
    r = Math.min(r, w, hgt / 2);
    return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt - r}Q${x + w},${y + hgt} ${x + w - r},${y + hgt}H${x}Z`;
  }

  // ------------------------------------------------------------------ forest plot
  /**
   * forest(body, { rows:[{label, sub, est, lo, hi, weight, display}], pooled:{label, est, lo, hi, piLo, piHi, display},
   *               domain:[a,b], ticks, fmt, refLabel })
   */
  function forest(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const narrow = W < 560;
      const rowH = narrow ? 40 : 28;
      const labelW = narrow ? 0 : Math.min(220, Math.round(W * 0.32));
      const valW = narrow ? 0 : 150;
      const top = 6, axisH = 30;
      const nRows = o.rows.length + (o.pooled ? 1 : 0);
      const H = top + nRows * rowH + (o.pooled ? 8 : 0) + axisH;
      const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
      const x0 = labelW + (labelW ? 12 : 4), x1 = W - valW - 8;
      const xs = linear(o.domain[0], o.domain[1], x0, x1);
      const yEnd = top + nRows * rowH + (o.pooled ? 8 : 0);
      const tk = o.ticks || ticks(o.domain[0], o.domain[1], 5);
      for (const v of tk) {
        s("line", { class: "grid", x1: xs(v), x2: xs(v), y1: top, y2: yEnd }, svg);
        txt(svg, xs(v), yEnd + 16, o.fmt ? o.fmt(v) : String(v), { "text-anchor": "middle" });
      }
      if (o.pooled) s("line", { x1: xs(o.pooled.est), x2: xs(o.pooled.est), y1: top, y2: yEnd, style: "stroke: var(--accent); stroke-width:1; opacity:0.6" }, svg);
      const maxW = Math.max(...o.rows.map((r) => r.weight || 0), 0.0001);
      const show = (r, cx, cy) => tip.show(r.label, [{ label: r.sub || "", value: r.display }], cx, cy);
      o.rows.forEach((r, i) => {
        const yMid = top + i * rowH + (narrow ? 26 : rowH / 2);
        if (narrow) txt(svg, 4, top + i * rowH + 11, `${r.label} · ${r.display}`, { class: "lbl" });
        else {
          txt(svg, 0, yMid + 4, r.label, { class: "lbl" });
          if (r.sub) txt(svg, labelW, yMid + 4, r.sub, { "text-anchor": "end" });
          txt(svg, W, yMid + 4, r.display, { "text-anchor": "end", class: "lbl" });
        }
        if (r.est === null || r.est === undefined) return;
        s("line", { x1: xs(Math.max(o.domain[0], r.lo)), x2: xs(Math.min(o.domain[1], r.hi)), y1: yMid, y2: yMid, style: `stroke:${r.color || "var(--ink-2)"}; stroke-width:1.5` }, svg);
        const side = 6 + 8 * Math.sqrt((r.weight || maxW) / maxW);
        const sq = s("rect", { x: xs(r.est) - side / 2, y: yMid - side / 2, width: side, height: side, rx: 1.5, tabindex: 0,
          style: `fill:${r.color || "var(--c1)"}; stroke: var(--surface); stroke-width: 2` }, svg);
        sq.addEventListener("pointermove", (ev) => show(r, ev.clientX, ev.clientY));
        sq.addEventListener("pointerleave", () => tip.hide());
        sq.addEventListener("focus", () => { const b = sq.getBoundingClientRect(); show(r, b.right, b.top); });
        sq.addEventListener("blur", () => tip.hide());
      });
      if (o.pooled) {
        const p = o.pooled;
        const yMid = top + o.rows.length * rowH + 8 + (narrow ? 26 : rowH / 2);
        s("line", { x1: x0, x2: x1, y1: yMid - rowH / 2 - 2, y2: yMid - rowH / 2 - 2, class: "axis" }, svg);
        if (p.piLo !== undefined) s("line", { x1: xs(Math.max(o.domain[0], p.piLo)), x2: xs(Math.min(o.domain[1], p.piHi)), y1: yMid, y2: yMid, style: "stroke: var(--accent); stroke-width:1.5; opacity:0.55" }, svg);
        const d = `M${xs(p.lo)},${yMid}L${xs(p.est)},${yMid - 7}L${xs(p.hi)},${yMid}L${xs(p.est)},${yMid + 7}Z`;
        s("path", { d, style: "fill: var(--accent)" }, svg);
        if (narrow) txt(svg, 4, top + o.rows.length * rowH + 8 + 11, `${p.label} · ${p.display}`, { class: "lbl-strong" });
        else {
          txt(svg, 0, yMid + 4, p.label, { class: "lbl-strong" });
          txt(svg, W, yMid + 4, p.display, { "text-anchor": "end", class: "lbl-strong" });
        }
      }
    });
  }

  // ------------------------------------------------------------------ odds-ratio forest (log axis, two estimates per row)
  /**
   * orForest(body, { groups:[{label, rows:[{label, unit, a:{or,lo,hi}, b:{or,lo,hi}, display, strong, tip:[{label, value}]}]}],
   *                  color, left, right, leftShort, rightShort, tickFmt, aria })
   * a: hollow circle (e.g. unadjusted), b: filled square (e.g. adjusted); a line at OR = 1.
   */
  function orForest(body, o) {
    const CAND = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20];
    let lo = 1, hi = 1;
    for (const g of o.groups) for (const r of g.rows) for (const e of [r.a, r.b]) if (e) { lo = Math.min(lo, e.lo); hi = Math.max(hi, e.hi); }
    const dLo = Math.max(0.1, CAND.filter((v) => v <= lo).pop() || CAND[0]);
    const dHi = Math.min(10, CAND.find((v) => v >= hi) || CAND[CAND.length - 1]);
    const tk = CAND.filter((v) => v >= dLo - 1e-9 && v <= dHi + 1e-9);
    const color = o.color || "var(--accent)";
    responsive(body, (W) => {
      body.replaceChildren();
      const narrow = W < 600;
      const rowH = narrow ? 44 : 29, headH = 30;
      const labelW = narrow ? 0 : Math.min(290, Math.round(W * 0.37));
      const valW = narrow ? 0 : 112;
      const top = 0, axisH = 50;
      const nRows = o.groups.reduce((a, g) => a + g.rows.length, 0);
      const yEnd = top + o.groups.length * headH + nRows * rowH;
      const H = yEnd + axisH;
      const svg = s("svg", { class: "orf", width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
      const x0 = labelW ? labelW + 16 : 8, x1 = W - (valW ? valW + 16 : 8);
      const lx = linear(Math.log(dLo), Math.log(dHi), x0, x1);
      const X = (v) => lx(Math.log(Math.max(dLo, Math.min(dHi, v))));
      const fmt = o.tickFmt || String;
      for (const v of tk) {
        s("line", { class: v === 1 ? "one" : "grid", x1: X(v), x2: X(v), y1: top + 8, y2: yEnd }, svg);
        txt(svg, X(v), yEnd + 16, fmt(v), { "text-anchor": "middle" });
      }
      txt(svg, X(1) - 8, yEnd + 36, narrow ? o.leftShort || o.left : o.left, { "text-anchor": "end", class: "dir" });
      txt(svg, X(1) + 8, yEnd + 36, narrow ? o.rightShort || o.right : o.right, { "text-anchor": "start", class: "dir" });
      const arrow = (x, y, dir, col) => s("path", { d: `M${x},${y - 4}L${x + 6 * dir},${y}L${x},${y + 4}Z`, style: `fill:${col}`, "pointer-events": "none" }, svg);
      const ci = (e, y, col, w) => {
        s("line", { x1: X(e.lo), x2: X(e.hi), y1: y, y2: y, style: `stroke:${col}; stroke-width:${w}`, "pointer-events": "none" }, svg);
        if (e.lo < dLo) arrow(X(dLo), y, -1, col);
        if (e.hi > dHi) arrow(X(dHi), y, 1, col);
      };
      let y = top;
      o.groups.forEach((g, gi) => {
        if (gi) s("line", { class: "sep", x1: 0, x2: W, y1: y + 6, y2: y + 6 }, svg);
        txt(svg, 0, y + headH - 8, g.label.toUpperCase(), { class: "grp" });
        y += headH;
        for (const r of g.rows) {
          const rowTop = y;
          const hit = s("rect", { class: "hit", x: -6, y: rowTop, width: W + 12, height: rowH, rx: 6, tabindex: 0, "aria-label": `${r.label} ${r.unit || ""}: ${r.display}` }, svg);
          const yMid = narrow ? rowTop + 30 : rowTop + rowH / 2;
          // label (+ unit) on the left, adjusted estimate right-aligned; on a phone both sit above the bars
          const lineY = narrow ? rowTop + 14 : yMid + 4;
          const val = txt(svg, W, lineY, r.display, { "text-anchor": "end", class: r.strong ? "lbl-strong" : "lbl" });
          const lab = txt(svg, 0, lineY, "", { class: r.strong ? "lbl-strong" : "lbl" });
          lab.appendChild(document.createTextNode(r.label));
          const unit = r.unit ? s("tspan", { class: "unit", dx: 5 }, lab) : null;
          if (unit) unit.textContent = r.unit;
          const len = () => (lab.getComputedTextLength ? lab.getComputedTextLength() : 0);
          const room = narrow ? W - (val.getComputedTextLength ? val.getComputedTextLength() : 90) - 10 : labelW;
          if (unit && len() > room * 1.1) lab.removeChild(unit);          // the unit stays in the tooltip and the table
          if (len() > room) { lab.setAttribute("textLength", room); lab.setAttribute("lengthAdjust", "spacingAndGlyphs"); }
          if (r.a) {
            ci(r.a, yMid - 4, "var(--line-3)", 1.5);
            s("circle", { cx: X(r.a.or), cy: yMid - 4, r: 3.6, class: "est-a", "pointer-events": "none" }, svg);
          }
          if (r.b) {
            ci(r.b, yMid + 4, color, 2);
            s("rect", { x: X(r.b.or) - 4.5, y: yMid + 4 - 4.5, width: 9, height: 9, rx: 1.5, style: `fill:${color}`, class: "est-b", "pointer-events": "none" }, svg);
          }
          const show = (cx, cy) => tip.show(r.unit ? `${r.label} · ${r.unit}` : r.label, r.tip || [], cx, cy);
          hit.addEventListener("pointermove", (ev) => show(ev.clientX, ev.clientY));
          hit.addEventListener("pointerleave", () => tip.hide());
          hit.addEventListener("focus", () => { const b = hit.getBoundingClientRect(); show(b.left + b.width * 0.6, b.top); });
          hit.addEventListener("blur", () => tip.hide());
          y += rowH;
        }
      });
    });
  }

  // ------------------------------------------------------------------ icon array (100 people)
  /** iconArray(body, { counts:[ints summing to 100], colors:[css], labels:[str] }) */
  function iconArray(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const cols = 20, rows = 5;
      const cell = Math.max(12, Math.min(26, Math.floor(W / cols)));
      const Wd = cell * cols, Hd = cell * rows * 1.55;
      const svg = s("svg", { width: Wd, height: Hd, viewBox: `0 0 ${Wd} ${Hd}`, role: "img", "aria-label": o.aria || "" }, body);
      let k = 0;
      const groups = [];
      o.counts.forEach((c, g) => { for (let i = 0; i < c; i++) groups.push(g); });
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const g = groups[k++];
          if (g === undefined) continue;
          const cx = c * cell + cell / 2, cy = r * cell * 1.55 + cell * 0.32;
          const col = o.colors[g];
          const grp = s("g", { tabindex: -1 }, svg);
          s("circle", { cx, cy, r: cell * 0.2, style: `fill:${col}` }, grp);
          const bw = cell * 0.56, bh = cell * 0.78, bx = cx - bw / 2, by = cy + cell * 0.25;
          s("path", { d: `M${bx},${by + bh}V${by + bw / 2}A${bw / 2},${bw / 2} 0 0 1 ${bx + bw},${by + bw / 2}V${by + bh}Z`, style: `fill:${col}` }, grp);
          grp.addEventListener("pointermove", (ev) => tip.show(o.labels[g], [{ label: "", value: `${o.counts[g]} / 100` }], ev.clientX, ev.clientY));
          grp.addEventListener("pointerleave", () => tip.hide());
        }
      }
    });
  }

  // ------------------------------------------------------------------ heat grid (single hue, opacity ramp)
  /** heat(body, { rows:[{label, values:[0..100|null]}], cols:[labels], fmt }) */
  function heat(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const labelW = Math.min(150, Math.round(W * 0.3));
      const n = o.cols.length;
      const cw = Math.max(10, (W - labelW - 4) / n);
      const rh = 24, top = 4, axisH = 40;
      const H = top + o.rows.length * rh + axisH;
      const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
      o.rows.forEach((r, i) => {
        const y = top + i * rh;
        txt(svg, labelW - 8, y + rh / 2 + 4, r.label, { "text-anchor": "end", class: "lbl" });
        r.values.forEach((v, j) => {
          const x = labelW + j * cw;
          const rect = s("rect", { x: x + 1, y: y + 1, width: Math.max(1, cw - 2), height: rh - 2, rx: 2,
            style: v === null ? "fill: var(--surface-2)" : `fill: var(--accent); fill-opacity:${(0.08 + 0.92 * (v / 100)).toFixed(3)}` }, svg);
          rect.addEventListener("pointermove", (ev) => tip.show(`${r.label} · ${o.cols[j]}`, [{ label: o.valueLabel || "", value: v === null ? "–" : o.fmt(v) }], ev.clientX, ev.clientY));
          rect.addEventListener("pointerleave", () => tip.hide());
          if (cw >= 30 && v !== null) {
            txt(svg, x + cw / 2, y + rh / 2 + 4, String(Math.round(v)), { "text-anchor": "middle", style: `font-size:10.5px; pointer-events:none; fill:${v > 55 ? "#ffffff" : "var(--ink-2)"}` });
          }
        });
      });
      const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor((W - labelW) / 44))));
      o.cols.forEach((c, j) => {
        if (j % every !== 0 && j !== n - 1) return;
        const x = labelW + j * cw + cw / 2;
        const tt = txt(svg, x, top + o.rows.length * rh + 14, c, { "text-anchor": "end", transform: `rotate(-35 ${x} ${top + o.rows.length * rh + 14})` });
        tt.setAttribute("dy", "0");
      });
    });
  }

  // ------------------------------------------------------------------ column chart (histograms)
  /**
   * columns(body, { cats:[labels], values:[number|null], small:[bool], color, fmt:(v)=>str, smallLabel,
   *                 valueLabel, height, aria })
   * A value of null with small=true is a suppressed cell (fewer patients than the privacy limit):
   * no bar is drawn and the cell is labelled smallLabel.
   */
  function columns(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const n = o.cats.length;
      const H = o.height || 170;
      const m = { l: 4, r: 4, t: 18, b: 22 };
      const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
      const bw = (W - m.l - m.r) / Math.max(1, n);
      const vals = o.values.filter((v) => v !== null && v !== undefined);
      const max = o.max || (vals.length ? Math.max(...vals) : 1) || 1;
      const Y = linear(0, max, H - m.b, m.t);
      const narrow = bw < 34;
      o.cats.forEach((c, i) => {
        const v = o.values[i];
        const x = m.l + i * bw;
        const cx = x + bw / 2;
        if (v !== null && v !== undefined) {
          const y = Y(v);
          const bar = s("rect", { x: x + bw * 0.14, y, width: Math.max(1, bw * 0.72), height: Math.max(0.5, H - m.b - y), rx: 3, style: `fill:${o.color || "var(--accent)"}` }, svg);
          bar.setAttribute("tabindex", "0");
          const show = (px, py) => tip.show(String(c), [{ label: o.valueLabel || "", color: o.color, value: o.fmt ? o.fmt(v) : String(v) }], px, py);
          bar.addEventListener("pointermove", (ev) => show(ev.clientX, ev.clientY));
          bar.addEventListener("pointerleave", () => tip.hide());
          bar.addEventListener("focus", () => { const b = bar.getBoundingClientRect(); show(b.right, b.top); });
          bar.addEventListener("blur", () => tip.hide());
          if (!narrow || n <= 8) txt(svg, cx, y - 5, o.fmt ? o.fmt(v) : String(v), { "text-anchor": "middle", class: "lbl", style: narrow ? "font-size:9.5px" : null });
        } else if (o.small && o.small[i]) {
          txt(svg, cx, H - m.b - 5, o.smallLabel || "–", { "text-anchor": "middle", class: "lbl", style: "fill: var(--faint)" });
        }
        txt(svg, cx, H - m.b + 15, String(c), { "text-anchor": "middle", style: narrow ? "font-size:9.5px" : null });
      });
      s("line", { x1: m.l, x2: W - m.r, y1: H - m.b + 0.5, y2: H - m.b + 0.5, class: "axis" }, svg);
    });
  }

  root.MTOPCharts = { h, s, figure, dataTable, lineChart, grotta, hbars, columns, tip, ticks, linear, forest, orForest, iconArray, heat };
})(typeof globalThis !== "undefined" ? globalThis : this);
