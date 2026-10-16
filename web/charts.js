/* MTOP charts: small, dependency-free SVG charts with hover tooltips, keyboard access,
 * and a table view for every figure. Colours come from CSS tokens so both themes work. */
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
    for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(10));
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
        const key = h("span", { class: "key" }, h("i", { class: l.type === "line" ? "line" : "", style: `background:${l.color}` }), l.label);
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
        o.x.cats.forEach((c, i) => {
          if (i % every !== 0 && i !== n - 1) return;
          txt(g, xs(i), m.t + ih + 16, c, { "text-anchor": "middle" });
        });
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
        s("line", { x1: mx, x2: mx, y1: m.t, y2: m.t + ih, style: "stroke: var(--ink-2); stroke-width:1" }, g);
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
  function grotta(body, o) {
    responsive(body, (W) => {
      body.replaceChildren();
      const narrow = W < 520;
      const labelW = narrow ? 0 : Math.min(190, Math.round(W * 0.28));
      const nW = 56;
      const rowH = narrow ? 44 : 30;
      const barH = 18;
      const H = o.rows.length * rowH + 8;
      const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria || "" }, body);
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
        let x = x0;
        r.segs.forEach((v, k) => {
          const w = (v / 100) * bw;
          if (w <= 0) return;
          const rx = x, rw = Math.max(0, w - 2);
          const rect = s("rect", { x: rx, y: yTop, width: rw, height: barH, rx: 0, style: `fill:${o.colors[k]}` }, svg);
          rect.setAttribute("tabindex", "0");
          const show = (cx, cy) => tip.show(r.label, [{ label: o.segLabels[k], color: o.colors[k], value: `${o.fmt(v)}` }], cx, cy);
          rect.addEventListener("pointermove", (ev) => { rect.style.opacity = "0.85"; show(ev.clientX, ev.clientY); });
          rect.addEventListener("pointerleave", () => { rect.style.opacity = ""; tip.hide(); });
          rect.addEventListener("focus", () => { const b = rect.getBoundingClientRect(); show(b.left + b.width / 2, b.top); });
          rect.addEventListener("blur", () => tip.hide());
          if (rw >= 30) {
            txt(svg, rx + rw / 2, yTop + barH / 2 + 4, `${Math.round(v)}`, {
              "text-anchor": "middle", style: `fill:${inkFor(rect)}; font-size:11px; font-weight:600; pointer-events:none`,
            });
          }
          x += w;
        });
        txt(svg, W, narrow ? yTop + 13 : labY, `${o.nLabel || "n"} ${r.n}`, { "text-anchor": "end" });
      });
    });
  }

  /** White or ink text for a label sitting on a coloured fill, by the fill's luminance. */
  function inkFor(node) {
    const c = getComputedStyle(node).fill || "";
    const m = c.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)/);
    if (!m) return "#ffffff";
    const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    const L = 0.2126 * lin(+m[1]) + 0.7152 * lin(+m[2]) + 0.0722 * lin(+m[3]);
    return L > 0.36 ? "#111820" : "#ffffff";
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

  root.MTOPCharts = { h, s, figure, dataTable, lineChart, grotta, hbars, tip, ticks, linear };
})(typeof globalThis !== "undefined" ? globalThis : this);
