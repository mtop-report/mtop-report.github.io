/* MTOP "Federated" tab: a working federated network, shown on four virtual centres.
 * Everything here comes from artifacts/public/federated.json (aggregates only). */
(function () {
  "use strict";
  const A = window.MTOPApp;
  if (!A) return;
  const C = window.MTOPCharts;
  const { h, tx, num, signed, S } = A;
  const F = A.DATA.federated;
  S.fedTarget = S.fedTarget || "good_outcome";
  const COL = { alone: "var(--c2)", network: "var(--c1)", adapted: "var(--c3)" };
  const MODELS = ["alone", "network", "adapted"];
  let anim = 0;
  let played = false;

  const reduced = () => !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const name = {
    alone: () => tx("Centre alone", "Centrum samo"),
    network: () => tx("Network model", "Model sítě"),
    adapted: () => tx("Network, adapted locally", "Síť + místní doladění"),
  };
  const target = (t) => (t === "good_outcome" ? tx("Functional independence", "Funkční nezávislost") : tx("Death by 90 days", "Úmrtí do 90 dní"));
  const centre = (id) => tx(`Centre ${id}`, `Centrum ${id}`);
  const era = (id) => (F.centres.find((c) => c.id === id) || {}).era || "";
  const pts = (n) => tx(`${num(n)} patients`, `${num(n)} pacientů`);
  const stageName = (s) => A.stageLabel(s);
  const svgEl = (tag, attrs, parent) => C.s(tag, attrs, parent);
  const viewWidth = () => (document.getElementById("view-federated") || document.body).clientWidth;
  function responsive(el, draw) {
    let last = 0;
    const run = () => { const w = Math.floor(el.clientWidth); if (w > 0 && Math.abs(w - last) > 2) { last = w; draw(w); } };
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(run).observe(el); else window.addEventListener("resize", run);
    requestAnimationFrame(run);
  }

  function outcomeSeg(onChange) {
    return h("div", { class: "mini-seg", role: "group", "aria-label": A.t().outcome },
      [["good_outcome", A.t().targetGood], ["death90", A.t().targetDeath]].map(([k, lbl]) => h("button", {
        type: "button", "aria-pressed": String(S.fedTarget === k), onclick: () => { if (S.fedTarget !== k) { S.fedTarget = k; onChange(); } },
      }, lbl)));
  }

  // ================================================================== render
  function render(root) {
    cancelAnimationFrame(anim);
    root.replaceChildren();
    if (!F || !F.outcomes) {
      root.appendChild(h("div", { class: "view-head" }, h("div", null, h("h2", { text: tx("Federated learning", "Federované učení") }),
        h("p", { text: tx("The federated demo has not been built yet (python -m mtop fed-demo).", "Ukázka federovaného učení zatím nebyla sestavena (python -m mtop fed-demo).") }))));
      return;
    }
    const res = F.outcomes[S.fedTarget];
    const rerender = () => render(root);
    root.appendChild(h("div", { class: "view-head fx-head" }, h("div", null,
      h("div", { class: "fx-eyebrow" }, A.statusChip("beta"), h("span", { text: tx("Working network · demo on FNO data", "Funkční síť · ukázka na datech FNO") })),
      h("h2", { text: tx("Learn together. Keep data at home.", "Učit se společně. Data nechat doma.") }),
      h("p", { text: tx(
        "MTOP can be trained across stroke centres without moving a single patient record. Each hospital runs a small node that answers the network's questions with masked sums; the coordinator adds them up and learns exactly what pooling the data would have taught. Below, a working network of four centres.",
        "MTOP lze trénovat napříč iktovými centry bez přesunu jediného záznamu pacienta. Každá nemocnice provozuje malý uzel, který na dotazy sítě odpovídá maskovanými součty; koordinátor je sečte a naučí se přesně to, co by ho naučila sloučená data. Níže je funkční síť čtyř center.") }))));
    root.appendChild(kpis());
    root.appendChild(liveCard(res, rerender));
    root.appendChild(centresCard(res, rerender));
    root.appendChild(h("div", { class: "grid-2 block" }, sizeCard(res), growthCard(res)));
    root.appendChild(stagesCard());
    root.appendChild(losoCard(res));
    root.appendChild(privacyCard());
    root.appendChild(proofCard());
    root.appendChild(joinCard());
    root.appendChild(h("p", { class: "note fx-demo-note", text: tx(
      `About this demo: the four centres are FNO's own thrombectomy registry split by treatment era (A ${era("A")}, B ${era("B")}, C ${era("C")}, D ${era("D")}). Each centre's data were exported in the MTOP template and read by its own node. Inside every centre, ${F.folds}-fold cross-validation kept a fifth of the patients aside, and each centre scored its own held-out patients. Model: boosted trees at admission (${F.recipe.trees} trees, depth ${F.recipe.max_depth}, learning rate ${num(F.recipe.learning_rate, 2)}, at least ${F.recipe.min_child} patients per leaf, monotone constraints). Real partner centres differ more in case mix and practice; measuring that is what the network is for.`,
      `O této ukázce: čtyři centra jsou registr trombektomií FNO rozdělený podle období léčby (A ${era("A")}, B ${era("B")}, C ${era("C")}, D ${era("D")}). Data každého centra byla exportována do šablony MTOP a načtena jeho vlastním uzlem. V každém centru se při ${F.folds}násobné křížové validaci odložila pětina pacientů a každé centrum hodnotilo své vlastní odložené pacienty. Model: boosting stromů při přijetí (${F.recipe.trees} stromů, hloubka ${F.recipe.max_depth}, rychlost učení ${num(F.recipe.learning_rate, 2)}, nejméně ${F.recipe.min_child} pacientů v listu, monotónní omezení). Skutečná partnerská centra se liší více, a právě to by síť měřila.`) }));
  }

  // ================================================================== headline numbers
  function kpis() {
    const g = F.outcomes.good_outcome, d = F.outcomes.death90;
    const best = (r) => Math.max(r.mean.network, r.mean.adapted) - r.mean.alone;
    const sz = d.size.sizes[0];
    const small = F.outcomes[S.fedTarget].size;
    const kp = (v, l, s) => h("div", { class: "kpi" }, h("div", { class: "l", text: l }), h("div", { class: "v", text: v }), s ? h("div", { class: "s", text: s }) : null);
    const eq = F.equivalence.gbdt;
    return h("div", { class: "kpis fx-kpis" },
      kp(signed(best(d), 2), tx("AUROC gain for 90-day death", "Zisk AUROC pro úmrtí do 90 dní"), tx("average over the four centres", "průměr čtyř center")),
      kp(signed(best(g), 2), tx("AUROC gain for mRS 0–2", "Zisk AUROC pro mRS 0–2"), tx("average over the four centres", "průměr čtyř center")),
      kp(`${num(small.alone[0], 2)} → ${num(small.network[0], 2)}`, tx(`A centre with only ${sz} patients`, `Centrum s pouhými ${sz} pacienty`),
        tx(`AUROC alone → in the network (${target(S.fedTarget).toLowerCase()})`, `AUROC samo → v síti (${target(S.fedTarget).toLowerCase()})`)),
      kp(eq.identical ? `${eq.trees}/${eq.trees}` : "–", tx("trees identical to pooled data", "stromů shodných se sloučenými daty"), tx("not approximately: exactly", "ne přibližně, ale přesně")),
      kp("0", tx("patient records left a hospital", "záznamů pacientů opustilo nemocnici"), tx("only masked sums travel", "putují jen maskované součty")));
  }

  // ================================================================== live replay
  function liveCard(res, rerender) {
    const card = h("div", { class: "card block fx-live" });
    card.appendChild(h("div", { class: "panel-head" }, h("h3", { text: tx("The network at work", "Síť při práci") }), outcomeSeg(rerender)));
    const R = F.run || {};
    card.appendChild(h("p", { class: "panel-sub", text: tx(
      `Replay of a training run on the four nodes: ${F.recipe.trees} trees, ${num(R.rounds || 0)} rounds. In every round each centre sends masked sums; only their total means anything.`,
      `Přehrání trénování na čtyřech uzlech: ${F.recipe.trees} stromů, ${num(R.rounds || 0)} kol. V každém kole posílá každé centrum maskované součty; význam má jen jejich celek.`) }));
    const grid = h("div", { class: "fx-live-grid" });
    const netBox = h("div", { class: "fx-net" });
    const curveBox = h("div", { class: "fx-curve" });
    grid.append(netBox, curveBox);
    card.appendChild(grid);
    const counters = h("div", { class: "fx-counters", "aria-live": "off" });
    const btn = h("button", { class: "btn", type: "button" });
    const skip = h("button", { class: "btn btn-ghost", type: "button", text: tx("Skip to end", "Přeskočit na konec") });
    card.appendChild(h("div", { class: "fx-live-foot" }, counters, h("div", { class: "fx-live-btns" }, skip, btn)));

    const narrow = viewWidth() < 640;
    const net = drawNetwork(netBox, narrow);
    const curve = drawCurve(curveBox, res, narrow);
    const T = F.recipe.trees;
    const rounds = R.rounds || 1 + 3 * T;
    const perTree = 240;
    const setCounters = (tree, round) => {
      counters.replaceChildren(
        h("span", null, h("b", { text: `${tree}` }), ` / ${T} ${tx("trees", "stromů")}`),
        h("span", null, h("b", { text: num(round) }), ` / ${num(rounds)} ${tx("rounds", "kol")}`),
        h("span", null, h("b", { text: "0" }), ` ${tx("patient records sent", "odeslaných záznamů")}`));
    };
    const finish = () => { net.idle(); curve.upTo(T); setCounters(T, rounds); btn.textContent = tx("Replay", "Přehrát znovu"); played = true; };
    let start = 0;
    const frame = (ts) => {
      if (!card.isConnected) return;
      if (!start) start = ts;
      const pos = (ts - start) / perTree;
      const tree = Math.min(T, Math.floor(pos));
      if (tree >= T) { finish(); return; }
      const phase = pos - tree;
      net.step(phase, tree);
      curve.upTo(tree + phase);
      setCounters(tree, Math.min(rounds, 1 + 3 * tree + Math.floor(phase * 3)));
      anim = requestAnimationFrame(frame);
    };
    const play = () => {
      cancelAnimationFrame(anim);
      if (reduced()) { finish(); return; }
      start = 0;
      btn.textContent = tx("Restart", "Znovu");
      anim = requestAnimationFrame(frame);
    };
    btn.addEventListener("click", play);
    skip.addEventListener("click", () => { cancelAnimationFrame(anim); finish(); });
    if (played || reduced()) finish();
    else { setCounters(0, 0); curve.upTo(0); btn.textContent = tx("Restart", "Znovu"); requestAnimationFrame(play); }
    return card;
  }

  function drawNetwork(box, narrow) {
    const W = narrow ? 360 : 560, H = narrow ? 400 : 330;
    const hub = { x: W / 2, y: H / 2, r: narrow ? 44 : 50 };
    const nw = narrow ? 150 : 170, nh = narrow ? 60 : 64;
    const pos = narrow
      ? [[90, 52], [W - 90, 52], [90, H - 52], [W - 90, H - 52]]
      : [[104, 58], [W - 104, 58], [104, H - 58], [W - 104, H - 58]];
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "fx-net-svg",
      "aria-label": tx("Four hospitals connected to a coordinator; masked sums flow in, the shared model flows out.",
        "Čtyři nemocnice spojené s koordinátorem; dovnitř proudí maskované součty, ven sdílený model.") }, box);
    const lines = svgEl("g", null, svg);
    const packets = svgEl("g", null, svg);
    const chips = svgEl("g", null, svg);
    const nodes = svgEl("g", null, svg);
    const ids = F.centres.map((c) => c.id);
    const el = ids.map((id, i) => {
      const [x, y] = pos[i];
      svgEl("line", { x1: x, y1: y, x2: hub.x, y2: hub.y, class: "fx-link" }, lines);
      const p = svgEl("circle", { r: 7, cx: x, cy: y, class: "fx-packet", visibility: "hidden" }, packets);
      // label beside the middle of the visible part of the link (between the box edge and the hub)
      const dx = hub.x - x, dy = hub.y - y, len = Math.hypot(dx, dy);
      const tOut = Math.min((nw / 2) / Math.abs(dx || 1e-9), (nh / 2) / Math.abs(dy || 1e-9));
      const tMid = (Math.min(tOut, 0.45) + (1 - hub.r / len)) / 2;
      const mx = x + dx * tMid, my = y + dy * tMid;
      const left = x < hub.x;
      const chip = C.s("text", { x: left ? mx - 10 : mx + 10, y: my + 4, class: "fx-hex", "text-anchor": left ? "end" : "start", visibility: "hidden" }, chips);
      const g = svgEl("g", { class: "fx-node" }, nodes);
      svgEl("rect", { x: x - nw / 2, y: y - nh / 2, width: nw, height: nh, rx: 14 }, g);
      const c = F.centres[i];
      const t1 = C.s("text", { x, y: y - 4, "text-anchor": "middle", class: "fx-node-t" }, g);
      t1.textContent = centre(id);
      const t2 = C.s("text", { x, y: y + 16, "text-anchor": "middle", class: "fx-node-s" }, g);
      t2.textContent = narrow ? `${c.era} · ${num(c[S.fedTarget].n)}` : `${c.era} · ${pts(c[S.fedTarget].n)}`;
      return { x, y, p, chip };
    });
    const hg = svgEl("g", { class: "fx-hub" }, nodes);
    const ring = svgEl("circle", { cx: hub.x, cy: hub.y, r: hub.r + 8, class: "fx-hub-ring" }, hg);
    svgEl("circle", { cx: hub.x, cy: hub.y, r: hub.r, class: "fx-hub-core" }, hg);
    const sig = C.s("text", { x: hub.x, y: hub.y + 4, "text-anchor": "middle", class: "fx-hub-t" }, hg);
    sig.textContent = "Σ";
    const sub = C.s("text", { x: hub.x, y: hub.y + 24, "text-anchor": "middle", class: "fx-hub-s" }, hg);
    sub.textContent = tx("coordinator", "koordinátor");
    const hex = () => Array.from({ length: 4 }, () => Math.floor(Math.random() * 65536).toString(16).padStart(4, "0")).join(" ").slice(0, 14);
    let lastHex = 0;
    return {
      step(phase, tree) {
        const now = performance.now();
        const refresh = now - lastHex > 90;
        if (refresh) lastHex = now;
        for (const n of el) {
          let s = null, cls = "fx-packet";
          if (phase < 0.28) { s = 1 - phase / 0.28; cls = "fx-packet task"; }
          else if (phase < 0.78) { s = (phase - 0.28) / 0.5; cls = "fx-packet up"; }
          if (s === null) { n.p.setAttribute("visibility", "hidden"); n.chip.setAttribute("visibility", "hidden"); continue; }
          n.p.setAttribute("visibility", "visible");
          n.p.setAttribute("class", cls);
          n.p.setAttribute("cx", n.x + (hub.x - n.x) * s);
          n.p.setAttribute("cy", n.y + (hub.y - n.y) * s);
          const up = cls.endsWith("up");
          n.chip.setAttribute("visibility", up ? "visible" : "hidden");
          if (up && refresh) n.chip.textContent = hex();
        }
        const pulse = phase >= 0.78 ? (phase - 0.78) / 0.22 : 0;
        ring.setAttribute("r", hub.r + 8 + 10 * pulse);
        ring.style.opacity = pulse ? String(0.55 * (1 - pulse)) : "0";
        sub.textContent = phase >= 0.78 ? tx(`tree ${tree + 1}`, `strom ${tree + 1}`) : tx("coordinator", "koordinátor");
      },
      idle() {
        for (const n of el) { n.p.setAttribute("visibility", "hidden"); n.chip.setAttribute("visibility", "hidden"); }
        ring.style.opacity = "0";
        sub.textContent = tx("coordinator", "koordinátor");
      },
    };
  }

  function drawCurve(box, res, narrow) {
    const cv = res.curves;
    const xs = cv.trees;
    const net = cv.mean.network, alone = cv.mean.alone;
    const all = net.concat(alone).filter((v) => v !== null);
    const lo = Math.floor((Math.min(...all) - 0.01) * 50) / 50, hi = Math.ceil((Math.max(...all) + 0.005) * 50) / 50;
    const W = narrow ? 340 : 440, H = narrow ? 260 : 300, m = { t: 24, r: narrow ? 58 : 92, b: 40, l: 44 };
    box.appendChild(h("div", { class: "fx-curve-title", text: tx("AUROC on each centre's held-out patients", "AUROC na odložených pacientech každého centra") }));
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", class: "fx-curve-svg",
      "aria-label": tx(`Learning curve: network ${num(net[net.length - 1], 2)} versus centre alone ${num(alone[alone.length - 1], 2)}`,
        `Křivka učení: síť ${num(net[net.length - 1], 2)} oproti centru samotnému ${num(alone[alone.length - 1], 2)}`) }, box);
    const X = C.linear(0, F.recipe.trees, m.l, W - m.r), Y = C.linear(lo, hi, H - m.b, m.t);
    for (const v of C.ticks(lo, hi, 4)) {
      svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }, svg);
      const t = C.s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "fx-axis" }, svg);
      t.textContent = num(v, 2);
    }
    for (const v of C.ticks(0, F.recipe.trees, 4)) {
      const t = C.s("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", class: "fx-axis" }, svg);
      t.textContent = String(v);
    }
    const xl = C.s("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "fx-axis" }, svg);
    xl.textContent = tx("trees in the model", "stromů v modelu");
    const pa = svgEl("path", { class: "fx-line alone" }, svg);
    const pn = svgEl("path", { class: "fx-line network" }, svg);
    const da = svgEl("circle", { r: 4, class: "fx-dot alone" }, svg);
    const dn = svgEl("circle", { r: 4.5, class: "fx-dot network" }, svg);
    const la = C.s("text", { class: "fx-end alone" }, svg);
    const ln = C.s("text", { class: "fx-end network" }, svg);
    const at = (arr, x) => {
      if (x <= xs[0]) return arr[0];
      for (let i = 1; i < xs.length; i++) if (x <= xs[i]) { const f = (x - xs[i - 1]) / (xs[i] - xs[i - 1]); return arr[i - 1] + f * (arr[i] - arr[i - 1]); }
      return arr[arr.length - 1];
    };
    const path = (arr, x) => {
      if (x < xs[0]) return "";
      let d = `M${X(xs[0]).toFixed(1)},${Y(arr[0]).toFixed(1)}`;
      for (let i = 1; i < xs.length && xs[i] <= x; i++) d += `L${X(xs[i]).toFixed(1)},${Y(arr[i]).toFixed(1)}`;
      return d + `L${X(x).toFixed(1)},${Y(at(arr, x)).toFixed(1)}`;
    };
    const legend = h("div", { class: "legend fx-legend" },
      h("span", { class: "key" }, h("i", { class: "line", style: `background:${COL.network}` }), tx("Network model", "Model sítě")),
      h("span", { class: "key" }, h("i", { class: "line dash", style: `background:${COL.alone}` }), tx("Each centre alone (average)", "Každé centrum samo (průměr)")));
    box.appendChild(legend);
    return {
      upTo(x) {
        const vis = x >= xs[0];
        pa.setAttribute("d", path(alone, x));
        pn.setAttribute("d", path(net, x));
        for (const [dot, lab, arr, dy] of [[da, la, alone, 0], [dn, ln, net, 0]]) {
          dot.setAttribute("visibility", vis ? "visible" : "hidden");
          lab.setAttribute("visibility", vis ? "visible" : "hidden");
          if (!vis) continue;
          const v = at(arr, x);
          dot.setAttribute("cx", X(x)); dot.setAttribute("cy", Y(v));
          lab.setAttribute("x", X(x) + 9); lab.setAttribute("y", Y(v) + 4 + dy);
          lab.textContent = num(v, 3);
        }
        const yn = Number(ln.getAttribute("y")), ya = Number(la.getAttribute("y"));
        if (vis && Math.abs(yn - ya) < 14) { if (yn <= ya) la.setAttribute("y", yn + 15); else ln.setAttribute("y", ya + 15); }
      },
    };
  }

  // ================================================================== per centre
  function centresCard(res, rerender) {
    const card = h("div", { class: "card block" });
    card.appendChild(h("div", { class: "panel-head" }, h("h3", { text: tx("What each centre gets", "Co získá každé centrum") }), outcomeSeg(rerender)));
    card.appendChild(h("p", { class: "panel-sub", text: tx(
      `${target(S.fedTarget)}: AUROC on each centre's own held-out patients. "Adapted locally" is the network model plus ${F.adapt_trees} boosting steps on the centre's own patients, done inside the hospital.`,
      `${target(S.fedTarget)}: AUROC na vlastních odložených pacientech každého centra. „Místní doladění“ je model sítě a ${F.adapt_trees} kroků boostingu na vlastních pacientech centra, provedených uvnitř nemocnice.`) }));
    const body = h("div", { class: "fx-dumb" });
    card.appendChild(h("div", { class: "legend fx-legend fx-pad" }, MODELS.map((m) =>
      h("span", { class: "key" }, h("i", { class: `dot ${m}`, style: `background:${COL[m]}` }), name[m]()))));
    card.appendChild(body);
    dumbbell(body, res.per_centre);
    const worse = res.per_centre.filter((e) => Math.max(e.network.auroc, e.adapted.auroc) < e.alone.auroc);
    const narrowNote = viewWidth() < 600 ? tx(" On small screens the right-hand numbers are the gains of the network model / adapted model.", " Na malých obrazovkách jsou čísla vpravo zisky modelu sítě / doladěného modelu.") : "";
    card.appendChild(h("p", { class: "note fx-dumb-note", text: tx(
      `Gains are AUROC differences against the centre alone, in bold green where the 95% bootstrap interval excludes zero. Hover over or tap a marker for intervals and calibration.${narrowNote}`,
      `Zisky jsou rozdíly AUROC oproti samotnému centru, tučně zeleně tam, kde 95% bootstrapový interval nezahrnuje nulu. Intervaly a kalibrace se zobrazí po najetí nebo klepnutí na značku.${narrowNote}`) }));
    if (worse.length) {
      const ids = worse.map((e) => e.id).join(", ");
      const one = worse.length === 1;
      let en = `${one ? `Centre ${ids} did` : `Centres ${ids} did`} better alone for this outcome.`;
      let cs = one ? `Centrum ${ids} si u tohoto výsledku vedlo lépe samo.` : `Centra ${ids} si u tohoto výsledku vedla lépe sama.`;
      if (worse.some((e) => e.id === "D")) {
        en += " In this demo, Centre D is FNO's most recent era, whose outcomes changed after 2020 (see Validation), so the older eras teach it less.";
        cs += " V této ukázce je centrum D nejnovějším obdobím FNO, jehož výsledky se po roce 2020 změnily (viz Validace), takže se od starších období naučí méně.";
      }
      en += " Making such differences visible is part of what a real network is for; adapting the shared model locally narrows them.";
      cs += " Zviditelnit takové rozdíly je součástí smyslu skutečné sítě; místní doladění sdíleného modelu je zmenšuje.";
      card.appendChild(h("p", { class: "note fx-dumb-note", text: tx(en, cs) }));
    }
    // calibration detail
    const det = h("details", { class: "fx-details" });
    det.appendChild(h("summary", { text: tx("Calibration at each centre", "Kalibrace v jednotlivých centrech") }));
    const rows = res.per_centre.map((e) => [centre(e.id), ...MODELS.map((m) => signed(e[m].citl, 2)), ...MODELS.map((m) => num(e[m].slope, 2))]);
    const calTbl = C.dataTable([tx("Centre", "Centrum"),
      ...MODELS.map((m) => `CITL · ${name[m]()}`), ...MODELS.map((m) => `${tx("Slope", "Sklon")} · ${name[m]()}`)], rows);
    calTbl.className = "data compact";
    det.appendChild(h("div", { class: "table-scroll" }, calTbl));
    det.appendChild(h("p", { class: "note", text: tx(
      "Calibration-in-the-large (CITL, 0 = predictions match the observed rate; negative = the model predicts too high) and calibration slope (1 = ideal). A shared model carries the network's average outcome rate; a few local steps bring it to the centre's own.",
      "Kalibrace v celku (CITL, 0 = predikce odpovídají pozorovanému podílu; záporná = model predikuje příliš vysoko) a sklon kalibrace (1 = ideál). Sdílený model nese průměrný výsledek sítě; několik místních kroků ho přizpůsobí vlastnímu centru.") }));
    card.appendChild(det);
    return card;
  }

  function dumbbell(body, rows) {
    responsive(body, (W) => {
      body.replaceChildren();
      const narrow = W < 600;
      const vals = rows.flatMap((e) => MODELS.map((m) => e[m].auroc));
      const lo = Math.floor((Math.min(...vals) - 0.02) * 20) / 20, hi = Math.ceil((Math.max(...vals) + 0.01) * 20) / 20;
      const labelW = narrow ? 0 : 190, gainW = narrow ? 0 : 150;
      const rowH = narrow ? 58 : 52, top = 6, axisH = 28;
      const H = top + rows.length * rowH + axisH;
      const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", class: "fx-dumb-svg",
        "aria-label": rows.map((e) => `${centre(e.id)}: ${MODELS.map((m) => `${name[m]()} ${num(e[m].auroc, 2)}`).join(", ")}`).join("; ") }, body);
      const x0 = labelW + (narrow ? 8 : 16), x1 = W - gainW - (narrow ? 8 : 20);
      const X = C.linear(lo, hi, x0, x1);
      const yEnd = top + rows.length * rowH;
      for (const v of C.ticks(lo, hi, narrow ? 4 : 6)) {
        svgEl("line", { x1: X(v), x2: X(v), y1: top, y2: yEnd, class: "grid" }, svg);
        const t = C.s("text", { x: X(v), y: yEnd + 18, "text-anchor": "middle", class: "fx-axis" }, svg);
        t.textContent = num(v, 2);
      }
      rows.forEach((e, i) => {
        const yMid = top + i * rowH + (narrow ? 38 : rowH / 2);
        const c = F.centres.find((q) => q.id === e.id);
        if (narrow) {
          const t = C.s("text", { x: 0, y: top + i * rowH + 14, class: "fx-row-t" }, svg);
          t.textContent = `${centre(e.id)} · ${c.era}`;
          const g = e.gain_network, ga = e.gain_adapted;
          const t2 = C.s("text", { x: W, y: top + i * rowH + 14, "text-anchor": "end", class: `fx-gain ${ga && ga.ci[0] > 0 ? "sig" : ""}` }, svg);
          t2.textContent = g ? `${signed(g.mean, 3)} / ${signed(ga.mean, 3)}` : "";
        } else {
          const t = C.s("text", { x: 0, y: yMid - 3, class: "fx-row-t" }, svg);
          t.textContent = centre(e.id);
          const s2 = C.s("text", { x: 0, y: yMid + 14, class: "fx-row-s" }, svg);
          s2.textContent = `${c.era} · ${pts(e.n)}`;
        }
        const v = MODELS.map((m) => e[m].auroc);
        svgEl("line", { x1: X(Math.min(...v)), x2: X(Math.max(...v)), y1: yMid, y2: yMid, class: "fx-dumb-bar" }, svg);
        MODELS.forEach((m) => {
          const dot = svgEl("circle", { cx: X(e[m].auroc), cy: yMid, r: 7, class: `fx-mark ${m}`, tabindex: 0,
            "aria-label": `${centre(e.id)}, ${name[m]()}: AUROC ${num(e[m].auroc, 3)}` }, svg);
          const show = (cx, cy) => C.tip.show(`${centre(e.id)} · ${name[m]()}`, [
            { label: "AUROC (95% CI)", value: e[m].auroc_ci ? `${num(e[m].auroc, 3)} (${num(e[m].auroc_ci[0], 2)}–${num(e[m].auroc_ci[1], 2)})` : num(e[m].auroc, 3) },
            ...(m !== "alone" && e[`gain_${m}`] ? [{ label: tx("Gain vs alone (95% CI)", "Zisk oproti samotnému (95% CI)"),
              value: `${signed(e[`gain_${m}`].mean, 3)} (${signed(e[`gain_${m}`].ci[0], 3)} až ${signed(e[`gain_${m}`].ci[1], 3)})`.replace(" až ", tx(" to ", " až ")) }] : []),
            { label: "CITL", value: signed(e[m].citl, 2) }, { label: tx("Calibration slope", "Sklon kalibrace"), value: num(e[m].slope, 2) }], cx, cy);
          dot.addEventListener("pointermove", (ev) => show(ev.clientX, ev.clientY));
          dot.addEventListener("pointerleave", () => C.tip.hide());
          dot.addEventListener("focus", () => { const bb = dot.getBoundingClientRect(); show(bb.right, bb.top); });
          dot.addEventListener("blur", () => C.tip.hide());
        });
        if (!narrow) {
          [["network", -3], ["adapted", 14]].forEach(([m, dy]) => {
            const g = e[`gain_${m}`];
            if (!g) return;
            const t = C.s("text", { x: W, y: yMid + dy, "text-anchor": "end", class: `fx-gain ${g.ci[0] > 0 ? "sig" : ""}` }, svg);
            t.textContent = `${m === "network" ? tx("network", "síť") : tx("adapted", "doladěný")} ${signed(g.mean, 3)}`;
            const tt = C.s("title", null, t);
            tt.textContent = `95% CI ${signed(g.ci[0], 3)} – ${signed(g.ci[1], 3)}`;
          });
        }
      });
    });
  }

  // ================================================================== size and growth
  function sizeCard(res) {
    const z = res.size;
    const cats = z.sizes.map(String).concat([tx(`all (≈${z.full.n_train})`, `všichni (≈${z.full.n_train})`)]);
    const alone = z.alone.concat([z.full.alone]), net = z.network.concat([z.full.network]);
    const box = h("div", { class: "card card-pad" });
    const fig = C.figure(box, {
      title: tx("Small centres gain most", "Nejvíce získají malá centra"),
      sub: tx(`${target(S.fedTarget)}: AUROC at a centre that has only N of its patients to learn from`, `${target(S.fedTarget)}: AUROC v centru, které se může učit jen z N svých pacientů`),
      legend: [{ label: name.alone(), color: COL.alone, type: "line" }, { label: tx("Same centre in the network", "Totéž centrum v síti"), color: COL.network, type: "line" }],
      table: { columns: [tx("Patients at the centre", "Pacientů v centru"), name.alone(), tx("In the network", "V síti")], rows: cats.map((c, i) => [c, num(alone[i], 3), num(net[i], 3)]) },
      labels: { table: A.t().showTable, chart: A.t().showChart },
    });
    const vals = alone.concat(net);
    C.lineChart(fig.body, {
      height: 240, x: { type: "cat", cats, title: tx("patients the centre contributes", "pacientů, kterými centrum přispívá") },
      y: { domain: [Math.floor((Math.min(...vals) - 0.02) * 20) / 20, Math.ceil((Math.max(...vals) + 0.01) * 20) / 20], fmt: (v) => num(v, 2) },
      series: [{ label: name.alone(), color: COL.alone, dash: true, dots: true, points: alone.map((y, i) => ({ x: i, y })) },
        { label: tx("In the network", "V síti"), color: COL.network, dots: true, width: 2.5, points: net.map((y, i) => ({ x: i, y })) }],
      tipFmt: (v) => num(v, 3),
    });
    box.appendChild(h("p", { class: "note", text: tx(
      `Each centre was given only ${z.sizes.join(", ")} of its training patients (repeated draws). Alone, it learns from those; in the network, together with the other centres. Scored on the centre's own held-out patients.`,
      `Každé centrum dostalo jen ${z.sizes.join(", ")} svých trénovacích pacientů (opakované výběry). Samo se učí jen z nich, v síti spolu s ostatními centry. Hodnoceno na vlastních odložených pacientech centra.`) }));
    return box;
  }

  function growthCard(res) {
    const g = res.growth;
    const cats = g.centres.map(String);
    const box = h("div", { class: "card card-pad" });
    const ids = F.centres.map((c) => c.id);
    const fig = C.figure(box, {
      title: tx("What each additional centre adds", "Co přidá každé další centrum"),
      sub: tx(`${target(S.fedTarget)}: AUROC at a centre by the number of centres learning together`, `${target(S.fedTarget)}: AUROC v centru podle počtu společně se učících center`),
      legend: [{ label: tx("Average", "Průměr"), color: COL.network, type: "line" }, { label: tx("Individual centres", "Jednotlivá centra"), color: "var(--line-3)", type: "line" }],
      table: { columns: [tx("Centres", "Center"), ...ids.map(centre), tx("Average", "Průměr")], rows: cats.map((c, i) => [c, ...ids.map((id) => num(g[id][i], 3)), num(g.mean[i], 3)]) },
      labels: { table: A.t().showTable, chart: A.t().showChart },
    });
    const vals = ids.flatMap((id) => g[id]);
    C.lineChart(fig.body, {
      height: 240, x: { type: "cat", cats, title: tx("centres in the network", "center v síti") },
      y: { domain: [Math.floor((Math.min(...vals) - 0.01) * 20) / 20, Math.ceil((Math.max(...vals) + 0.01) * 20) / 20], fmt: (v) => num(v, 2) },
      series: ids.map((id) => ({ label: centre(id), color: "var(--line-3)", width: 1.25, points: g[id].map((y, i) => ({ x: i, y })) }))
        .concat([{ label: tx("Average", "Průměr"), color: COL.network, width: 2.75, dots: true, points: g.mean.map((y, i) => ({ x: i, y })) }]),
      tipFmt: (v) => num(v, 3), tipHead: (i) => tx(`${cats[i]} ${i === 0 ? "centre" : "centres"}`, `${cats[i]} ${i === 0 ? "centrum" : i < 4 ? "centra" : "center"}`),
    });
    box.appendChild(h("p", { class: "note", text: tx(
      `Averaged over every possible group of that size that includes the centre. The network itself requires at least ${F.settings.min_sites} centres; groups of two are shown for the trend.`,
      `Průměr přes všechny možné skupiny dané velikosti, které centrum obsahují. Síť sama vyžaduje nejméně ${F.settings.min_sites} centra; dvojice jsou uvedeny pro trend.`) }));
    return box;
  }

  // ================================================================== stages
  function stagesCard() {
    const card = h("div", { class: "card card-pad block" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Where the network matters most", "Kde síť pomáhá nejvíce") }),
      h("div", { class: "chart-sub", text: tx("Mean AUROC over the four centres, each scored on its own held-out patients", "Průměrné AUROC čtyř center, každé hodnocené na vlastních odložených pacientech") }))));
    const tbl = h("table", { class: "data fx-table" });
    tbl.appendChild(h("thead", null, h("tr", null, [tx("Moment", "Okamžik"), tx("Outcome", "Výsledek"), tx("Patients per centre", "Pacientů na centrum"),
      name.alone(), name.network(), name.adapted(), tx("Gain", "Zisk")].map((c, i) => h("th", { scope: "col", class: i > 0 && i < 3 ? "txt" : i > 2 ? "num" : "", text: c })))));
    const tb = h("tbody");
    for (const st of F.stages) {
      const ns = st.per_centre.map((e) => e.n);
      const g = Math.max(st.mean.network, st.mean.adapted) - st.mean.alone;
      tb.appendChild(h("tr", null,
        h("th", { scope: "row", text: stageName(st.stage) }), h("td", { class: "txt", text: target(st.target) }),
        h("td", { class: "txt", text: `${num(Math.min(...ns))}–${num(Math.max(...ns))}` }),
        h("td", { class: "num", text: num(st.mean.alone, 3) }), h("td", { class: "num", text: num(st.mean.network, 3) }),
        h("td", { class: "num", text: num(st.mean.adapted, 3) }),
        h("td", { class: "num" }, h("span", { class: `fx-chip ${g >= 0.03 ? "hi" : g >= 0.01 ? "mid" : "lo"}`, text: signed(g, 3) }))));
    }
    tbl.appendChild(tb);
    card.appendChild(h("div", { class: "table-scroll" }, tbl));
    const all = F.stages.flatMap((st) => st.per_centre);
    const better = all.filter((e) => Math.max(e.network, e.adapted) > e.alone).length;
    const gainOf = (st) => Math.max(st.mean.network, st.mean.adapted) - st.mean.alone;
    const top = F.stages.reduce((a, b) => (gainOf(b) > gainOf(a) ? b : a));
    const late = F.stages.filter((st) => st.stage === "h24" || st.stage === "d7");
    const lateAll = late.length > 0 && late.every((st) => st.per_centre.every((e) => Math.max(e.network, e.adapted) > e.alone));
    card.appendChild(h("p", { class: "note", text: tx(
      `Gain: the better of the network model and the locally adapted network model, against the centre alone. In ${better} of ${all.length} comparisons (${F.stages.length} settings × ${F.centres.length} centres) the centre did better with the network.`
        + (top.target === "death90" ? " The largest gain is for death, the outcome with the fewest events per centre." : "")
        + (lateAll ? " At 24 hours and day 7 every centre gains, though from an already high level." : ""),
      `Zisk: lepší z modelu sítě a místně doladěného modelu sítě oproti samotnému centru. V ${better} z ${all.length} srovnání (${F.stages.length} nastavení × ${F.centres.length} centra) si centrum se sítí vedlo lépe.`
        + (top.target === "death90" ? " Největší zisk je u úmrtí, výsledku s nejmenším počtem událostí na centrum." : "")
        + (lateAll ? " Za 24 hodin a 7. den získává každé centrum, i když z již vysoké úrovně." : "")) }));
    return card;
  }

  // ================================================================== leave one centre out
  function losoCard(res) {
    const box = h("div", { class: "card card-pad block" });
    const fig = C.figure(box, {
      title: tx("Validated where it would be used", "Ověřeno tam, kde by se používal"),
      sub: tx(`${target(S.fedTarget)}: a network of three centres trained the model; the fourth tested it on all its patients, at home. Only the result left the centre.`,
        `${target(S.fedTarget)}: model natrénovala síť tří center; čtvrté ho otestovalo na všech svých pacientech, doma. Centrum opustil jen výsledek.`),
      table: { columns: [tx("Tested at", "Testováno v"), tx("Trained by", "Trénovaly"), "AUROC (95% CI)", "CITL", tx("Slope", "Sklon")],
        rows: res.loso.map((r) => [centre(r.id), r.trained_on.join(" + "), `${num(r.auroc, 3)} (${num(r.auroc_ci[0], 2)}–${num(r.auroc_ci[1], 2)})`, signed(r.citl, 2), num(r.slope, 2)]) },
      labels: { table: A.t().showTable, chart: A.t().showChart },
    });
    const vals = res.loso.flatMap((r) => r.auroc_ci);
    C.forest(fig.body, {
      rows: res.loso.map((r) => ({ label: centre(r.id), sub: tx(`trained by ${r.trained_on.join(" + ")}`, `trénovaly ${r.trained_on.join(" + ")}`), est: r.auroc, lo: r.auroc_ci[0], hi: r.auroc_ci[1],
        weight: r.n, display: `${num(r.auroc, 2)} (${num(r.auroc_ci[0], 2)}–${num(r.auroc_ci[1], 2)})`, color: "var(--c1)" })),
      domain: [Math.floor((Math.min(...vals) - 0.02) * 20) / 20, Math.min(1, Math.ceil((Math.max(...vals) + 0.02) * 20) / 20)], fmt: (v) => num(v, 2),
    });
    const worst = res.loso.reduce((a, b) => (Math.abs(b.citl) > Math.abs(a.citl) ? b : a));
    box.appendChild(h("p", { class: "note", text: tx(
      `Discrimination carries over to a centre the model has never seen. Calibration is where centres differ (largest at ${centre(worst.id)}: CITL ${signed(worst.citl, 2)}), which a centre can measure and correct locally before use.`,
      `Rozlišovací schopnost se přenáší i do centra, které model nikdy neviděl. Centra se liší v kalibraci (nejvíce ${centre(worst.id)}: CITL ${signed(worst.citl, 2)}), kterou si centrum může před použitím změřit a opravit na místě.`) }));
    return box;
  }

  // ================================================================== privacy
  const ICON = {
    lock: "M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z",
    check: "M5 12.5l4.2 4.2L19 7",
    log: "M7 4h10v16H7zM10 8h4M10 12h4M10 16h2",
    out: "M7 17L17 7M9 7h8v8",
    bands: "M4 18h16M6 18V9M10 18V5M14 18v-7M18 18v-4",
    people: "M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM3 20a6 6 0 0 1 12 0M16 11a2.5 2.5 0 1 0 0-5M17.5 20a5 5 0 0 0-2-4",
    key: "M8 20a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9zM11.2 12.3L20 3.5M16.5 7l2.5 2.5M14 9.5l2 2",
    code: "M9 8l-4 4 4 4M15 8l4 4-4 4",
  };
  const icon = (k) => {
    const svg = svgEl("svg", { viewBox: "0 0 24 24", class: "fx-ico", "aria-hidden": "true" });
    svgEl("path", { d: ICON[k] }, svg);
    return svg;
  };

  function privacyCard() {
    const card = h("div", { class: "card card-pad block" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("What leaves a hospital", "Co opouští nemocnici") }),
      h("div", { class: "chart-sub", text: tx("The first message of the run above, as each party sees it", "První zpráva výše přehraného běhu tak, jak ji vidí každá strana") }))));
    const smp = (F.run && F.run.sample) || { masked: ["9f3c6781aa4ffc82", "1292fb742c83b7c8"], total: [0, 0] };
    const hex = (s) => s.match(/.{1,4}/g).join(" ");
    const flow = h("div", { class: "fx-flow" },
      h("div", { class: "fx-flow-col" },
        h("div", { class: "fx-flow-k", text: tx("Inside the hospital", "Uvnitř nemocnice") }),
        h("div", { class: "fx-flow-v", text: tx("Patient rows", "Řádky pacientů") }),
        h("p", { text: tx("The registry export stays on the hospital's computer. The node reads it and computes sums over its own patients.",
          "Export registru zůstává na počítači nemocnice. Uzel ho čte a počítá součty přes vlastní pacienty.") })),
      h("div", { class: "fx-flow-col wire" },
        h("div", { class: "fx-flow-k", text: tx(`On the wire from Centre ${smp.site || "A"}`, `Po síti z centra ${smp.site || "A"}`) }),
        h("div", { class: "fx-flow-v mono" }, smp.masked.map((m) => h("div", { text: hex(m) }))),
        h("p", { text: tx("Each number is hidden with masks that only the centres can produce together. On its own it is random.",
          "Každé číslo je skryté maskami, které umějí vytvořit jen centra společně. Samo o sobě je náhodné.") })),
      h("div", { class: "fx-flow-col hub" },
        h("div", { class: "fx-flow-k", text: tx("At the coordinator, after adding all four", "U koordinátora po sečtení všech čtyř") }),
        h("div", { class: "fx-flow-v" }, h("div", { text: tx(`${num(smp.total[0])} patients`, `${num(smp.total[0])} pacientů`) }), h("div", { text: tx(`${num(smp.total[1])} with mRS 0–2`, `${num(smp.total[1])} s mRS 0–2`) })),
        h("p", { text: tx("Only network totals become readable: here, the patient count and outcome count. During training, sums per value band in each tree node.",
          "Čitelné jsou jen součty za celou síť: zde počet pacientů a výsledků. Při trénování součty v pásmech hodnot v každém uzlu stromu.") })));
    card.appendChild(flow);
    const items = [
      ["lock", tx("Secure aggregation", "Bezpečná agregace"), tx("Pairwise masks from X25519 keys and ChaCha20; no single centre's numbers are readable. Every job needs at least three centres.", "Párové masky z klíčů X25519 a ChaCha20; čísla žádného jednotlivého centra nejsou čitelná. Každá úloha potřebuje nejméně tři centra.")],
      ["check", tx("Approval at every centre", "Schválení v každém centru"), tx("A person approves each job in the node's local console; the job declares its purpose and maximum number of rounds in advance.", "Každou úlohu schvaluje člověk v místní konzoli uzlu; úloha předem uvádí účel a nejvyšší počet kol.")],
      ["log", tx("Audit log", "Auditní záznam"), tx("Time, round, size and SHA-256 of every message the hospital sent, kept by the hospital.", "Čas, kolo, velikost a SHA-256 každé zprávy, kterou nemocnice odeslala; vede ho nemocnice.")],
      ["out", tx("Outbound only", "Jen odchozí spojení"), tx("Nodes connect out to the coordinator over HTTPS. No open port in the hospital network.", "Uzly se připojují ke koordinátorovi přes HTTPS. V síti nemocnice není otevřený port.")],
      ["key", tx("Signed identities", "Podepsané identity"), tx("Job keys are signed with each centre's identity key, so the coordinator cannot pose as a centre.", "Klíče úloh jsou podepsané identitním klíčem centra, koordinátor se tedy nemůže za centrum vydávat.")],
      ["bands", tx("Fixed, public value bands", "Pevná veřejná pásma hodnot"), tx(`Splits fall on bands agreed in advance (at most ${maxBands()} per predictor), never on a centre's data.`, `Dělení stromů padá na předem dohodnutá pásma (nejvýše ${maxBands()} na prediktor), nikdy na data centra.`)],
      ["people", tx("Minimum group sizes", "Minimální velikost skupin"), tx(`At least ${F.settings.min_site_patients} patients per centre and ${F.recipe.min_child} per tree leaf; published summaries suppress groups under 10.`, `Nejméně ${F.settings.min_site_patients} pacientů na centrum a ${F.recipe.min_child} v listu stromu; zveřejněné souhrny potlačují skupiny pod 10.`)],
      ["code", tx("Open and small", "Otevřené a malé"), tx("Python standard library plus one cryptography package; the whole protocol is a few hundred lines anyone can audit.", "Standardní knihovna Pythonu a jeden kryptografický balíček; celý protokol má pár set řádků, které může kdokoli prověřit.")],
    ];
    card.appendChild(h("div", { class: "fx-protect" }, items.map(([ic, t1, t2]) => h("div", { class: "fx-protect-item" }, icon(ic), h("div", null, h("b", { text: t1 }), h("span", { text: t2 }))))));
    card.appendChild(h("div", { class: "callout fx-limits" }, h("b", { text: tx("Limits, stated plainly. ", "Omezení, otevřeně. ") }), tx(
      "The coordinator sees network-wide totals inside each tree node, and a narrow band can describe few patients. It is therefore run by a trusted institution under a data-processing agreement, keeps only the finished model, and can never tell which centre a number came from. Secure aggregation assumes the coordinator does not collude with all other centres against one. Formal differential privacy could be added, but for networks of this size it would cost substantial accuracy.",
      "Koordinátor vidí součty celé sítě v každém uzlu stromu a úzké pásmo může popisovat málo pacientů. Provozuje ho proto důvěryhodná instituce na základě smlouvy o zpracování, uchovává jen hotový model a nikdy nepozná, ze kterého centra číslo pochází. Bezpečná agregace předpokládá, že se koordinátor nespolčí se všemi ostatními centry proti jednomu. Formální diferenciální soukromí by bylo možné doplnit, u takto malých sítí by však výrazně snížilo přesnost.")));
    return card;
  }

  function maxBands() {
    return F.settings.bands_max || 30;
  }

  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  function sci(v) {
    if (!v) return "0";
    const e = Math.floor(Math.log10(Math.abs(v)));
    const m = v / Math.pow(10, e);
    return `${num(m, 1)} × 10${String(e).split("").map((c) => SUP[c]).join("")}`;
  }

  // ================================================================== proof
  function proofCard() {
    const E = F.equivalence, R = F.run;
    const card = h("div", { class: "card card-pad block" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("Proof that it is exact", "Důkaz, že je to přesné") }),
      h("div", { class: "chart-sub", text: tx("The same models trained by the network and on pooled data", "Tytéž modely natrénované sítí a na sloučených datech") }))));
    const rows = [
      [tx("Boosted trees (MTOP's model family)", "Boosting stromů (rodina modelů MTOP)"),
        tx(`${E.gbdt.trees} of ${E.gbdt.trees} trees identical, every split and leaf; largest difference in predicted risk ${sci(E.gbdt.max_abs_risk_diff)}`,
          `${E.gbdt.trees} z ${E.gbdt.trees} stromů shodných, každé dělení i list; největší rozdíl v predikovaném riziku ${sci(E.gbdt.max_abs_risk_diff)}`)],
      [tx("Spline logistic regression (the earlier report's model)", "Logistická regrese se splajny (model dřívější zprávy)"),
        tx(`${E.logreg.coefficients} coefficients in ${E.logreg.iterations} Newton rounds; largest difference ${sci(E.logreg.max_abs_coef_diff)}, in risk ${sci(E.logreg.max_abs_risk_diff)}`,
          `${E.logreg.coefficients} koeficientů v ${E.logreg.iterations} Newtonových kolech; největší rozdíl ${sci(E.logreg.max_abs_coef_diff)}, v riziku ${sci(E.logreg.max_abs_risk_diff)}`)],
      [tx("Pooled validation from counts", "Společná validace z počtů"),
        tx(`AUROC ${num(E.eval.auroc_federated, 3)} from masked counts vs ${num(E.eval.auroc_exact, 3)} exact; calibration slope ${num(E.eval.slope_federated, 3)} vs ${num(E.eval.slope_exact, 3)}`,
          `AUROC ${num(E.eval.auroc_federated, 3)} z maskovaných počtů vs ${num(E.eval.auroc_exact, 3)} přesně; sklon kalibrace ${num(E.eval.slope_federated, 3)} vs ${num(E.eval.slope_exact, 3)}`)],
    ];
    if (R) rows.push([tx("Over real network connections", "Přes skutečná síťová spojení"),
      tx(`${num(R.rounds)} rounds in ${num(R.seconds_train, 1)} s, ${num(R.kb_up_per_centre / 1000, 1)} MB sent per centre, median round ${num(R.median_round_ms, 0)} ms; result identical: ${R.identical_to_pooled ? "yes" : "no"}`,
        `${num(R.rounds)} kol za ${num(R.seconds_train, 1)} s, ${num(R.kb_up_per_centre / 1000, 1)} MB odeslaných z centra, medián kola ${num(R.median_round_ms, 0)} ms; výsledek shodný: ${R.identical_to_pooled ? "ano" : "ne"}`)]);
    card.appendChild(h("ul", { class: "fx-proof" }, rows.map(([a, b]) => h("li", null, icon("check"), h("div", null, h("b", { text: a }), h("span", { text: b }))))));
    card.appendChild(h("p", { class: "note", text: tx(
      `Why exact: each centre rounds its patients' gradients to 2${String(-F.settings.fixed_point_bits).split("").map((c) => SUP[c]).join("")} before adding them up, so all sums are whole numbers and the order of addition cannot matter. The coordinator then chooses every split from the same totals pooled training would use. Masks: ${F.crypto.key_agreement}; ${F.crypto.kdf}; ${F.crypto.masks}; ${F.crypto.arithmetic.replace("2^64", "2⁶⁴")}.`,
      `Proč přesně: každé centrum zaokrouhlí gradienty svých pacientů na 2${String(-F.settings.fixed_point_bits).split("").map((c) => SUP[c]).join("")} dříve, než je sečte, takže všechny součty jsou celá čísla a na pořadí sčítání nezáleží. Koordinátor pak volí každé dělení ze stejných součtů, jaké by použilo trénování na sloučených datech. Masky: X25519 s novými klíči pro každou úlohu podepsanými identitními klíči Ed25519; HKDF-SHA256; proud klíče ChaCha20; aritmetika modulo 2⁶⁴.`) }));
    return card;
  }

  // ================================================================== join
  function joinCard() {
    const card = h("div", { class: "card card-pad block fx-join" });
    card.appendChild(h("div", { class: "chart-head" }, h("div", null,
      h("div", { class: "chart-title", text: tx("How a centre joins", "Jak se centrum připojí") }),
      h("div", { class: "chart-sub", text: tx("What it takes for the registry and IT teams", "Co to obnáší pro tým registru a IT") }))));
    const repo = (A.SITE.repo_url || "https://github.com/viktorgajdosik/mt").replace(/\/$/, "");
    const cmd = `pip install "mtop[federated] @ git+${repo}"\npython -m mtop.fed check registry.csv\npython -m mtop.fed site --coordinator https://… --site-id YOUR-CENTRE \\\n    --token-file centre.token --data registry.csv`;
    const code = h("pre", { class: "fx-code" }, h("code", { text: cmd }));
    const copy = h("button", { class: "btn btn-ghost fx-copy", type: "button", text: tx("Copy", "Kopírovat") });
    copy.addEventListener("click", () => {
      const done = () => { copy.textContent = tx("Copied", "Zkopírováno"); setTimeout(() => { copy.textContent = tx("Copy", "Kopírovat"); }, 1600); };
      if (navigator.clipboard) navigator.clipboard.writeText(cmd).then(done, () => {});
    });
    const tplBtn = h("button", { class: "btn", type: "button", text: tx("Download the template", "Stáhnout šablonu") });
    tplBtn.addEventListener("click", () => {
      const cols = A.B.features.filter((f) => f.type !== "derived").map((f) => f.key).concat(["mrs90"]);
      const blob = new Blob([cols.join(",") + "\n"], { type: "text/csv" });
      const a = h("a", { href: URL.createObjectURL(blob), download: "mtop_template.csv" });
      document.body.appendChild(a); a.click(); a.remove();
    });
    const netBtn = h("button", { class: "btn btn-ghost", type: "button", text: tx("Check your data in the browser", "Ověřit data v prohlížeči") });
    netBtn.addEventListener("click", () => A.goTo("network"));
    const steps = [
      [tx("Agree", "Dohoda"), tx("Join the protocol and the shared data dictionary. Patient rows never leave your hospital; the coordinator's role is set out in a processing agreement, and your ethics committee and data protection officer confirm local participation and the agreements your institution needs.",
        "Přistoupení k protokolu a společnému datovému slovníku. Řádky pacientů nikdy neopustí vaši nemocnici; roli koordinátora upravuje smlouva o zpracování a místní účast i potřebné smlouvy potvrdí vaše etická komise a pověřenec pro ochranu osobních údajů.")],
      [tx("Map", "Mapování"), tx("Export your thrombectomy registry in the MTOP template (one row per patient). The Multicentre tab checks the mapping in your browser, without uploading anything.",
        "Exportujte svůj registr trombektomií do šablony MTOP (jeden řádek na pacienta). Záložka Multicentrum ověří mapování ve vašem prohlížeči, nic se nenahrává.")],
      [tx("Install", "Instalace"), tx("Run the node on any computer inside the hospital that can make outgoing HTTPS connections.", "Spusťte uzel na libovolném počítači v nemocnici, který může navazovat odchozí HTTPS spojení.")],
      [tx("Approve and learn", "Schvalovat a učit se"), tx("Approve jobs in the local console. You receive the shared model, your own validation, and the option to adapt the model to your patients.",
        "Schvalujte úlohy v místní konzoli. Dostanete sdílený model, vlastní validaci a možnost přizpůsobit model svým pacientům.")],
    ];
    card.appendChild(h("ol", { class: "fx-steps" }, steps.map(([a, b], i) => h("li", null, h("span", { class: "fx-step-n", text: String(i + 1) }),
      h("div", null, h("b", { text: a }), h("span", { text: b }), i === 2 ? h("div", { class: "fx-code-wrap" }, code, copy) : null,
        i === 1 ? h("div", { class: "fx-btns" }, tplBtn, netBtn) : null)))));
    if (A.SITE.contact) {
      card.appendChild(h("div", { class: "fx-contact" }, h("span", { text: tx("Interested in joining or coordinating?", "Máte zájem se připojit nebo síť koordinovat?") }),
        h("a", { class: "btn btn-primary", href: `mailto:${A.SITE.contact}?subject=${encodeURIComponent("MTOP federated network")}`, text: A.SITE.contact })));
    }
    return card;
  }

  A.registerView("federated", () => tx("Federated", "Federované učení"), render);
})();
