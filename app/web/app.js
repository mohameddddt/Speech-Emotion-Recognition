// Speech Emotion Recognition: showcase page + live demo.
// Audio is decoded and resampled to 16 kHz mono in the browser, sent as WAV to /api/predict.

const $ = (sel, el = document) => el.querySelector(sel);
const h = (tag, attrs = {}, ...children) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) el.append(c);
  return el;
};
const SVG_NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const emoVar = (e) => `var(--${e})`;
const cap = (w) => w.charAt(0).toUpperCase() + w.slice(1);
const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"];
const SAMPLE_RATE = 16000;
const MAX_RECORD_SECONDS = 30;

/* ---------------- theme, nav, reveal ---------------- */
const themeListeners = [];
function effectiveTheme() {
  return document.documentElement.dataset.theme ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
$("#theme-toggle").addEventListener("click", () => {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("ser-theme", next); } catch (e) { /* storage unavailable */ }
  themeListeners.forEach((f) => f());
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => themeListeners.forEach((f) => f()));

const nav = $(".nav");
const navLinks = [...document.querySelectorAll(".nav-links a")].map((a) => ({ a, target: $(a.getAttribute("href")) }));
function onScroll() {
  nav.classList.toggle("scrolled", scrollY > 8);
  // highlight the last nav target whose top has passed under the sticky header
  let current = null;
  for (const l of navLinks) if (l.target && l.target.getBoundingClientRect().top <= 140) current = l;
  if (innerHeight + scrollY >= document.documentElement.scrollHeight - 2) current = navLinks.at(-1);
  navLinks.forEach((l) => l.a.classList.toggle("active", l === current));
}
addEventListener("scroll", onScroll, { passive: true });
onScroll();

const revealer = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); revealer.unobserve(e.target); }
}, { rootMargin: "0px 0px -8% 0px" });
document.querySelectorAll(".section-head, .figure, .step, .limit, .numbers .stat").forEach((el) => {
  el.classList.add("reveal"); revealer.observe(el);
});

/* ---------------- tooltip ---------------- */
const tip = $("#tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}
const hideTip = () => { tip.hidden = true; };
function hoverTip(el, html) {
  el.addEventListener("pointermove", (e) => showTip(typeof html === "function" ? html() : html, e.clientX, e.clientY));
  el.addEventListener("pointerleave", hideTip);
}
const ttRows = (rows) => rows.map(([k, v]) => `<div class="tt-row"><span>${k}</span><b>${v}</b></div>`).join("");

function segmented(container, options, value, onChange) {
  container.replaceChildren();
  container.hidden = options.length < 2;
  const buttons = options.map((o) => {
    const b = h("button", { type: "button", role: "radio", "aria-checked": String(o.value === value) }, o.label);
    b.addEventListener("click", () => {
      buttons.forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      onChange(o.value);
    });
    container.append(b);
    return b;
  });
  return buttons;
}

function table(headers, rows) {
  const t = h("table");
  t.append(h("thead", {}, h("tr", {}, headers.map((x) => h("th", { class: x.num ? "num" : "" }, x.label)))));
  t.append(h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c, i) => h("td", { class: headers[i].num ? "num" : "" }, c))))));
  return h("div", { class: "table-scroll" }, t);
}

/* ---------------- data-bound numbers ---------------- */
function bindNumbers(R) {
  const best = R.models.find((m) => m.id === R.best);
  const hand = Math.max(...R.models.filter((m) => m.family === "hand-crafted").map((m) => m.accuracy));
  const gain = Math.round((best.accuracy - hand) * 100);
  const sel = R.layers.selected;
  const enc = R.encoder || {};
  if (enc.hf_id) {
    const a = $("#enc-link");
    a.href = `https://huggingface.co/${enc.hf_id}`;
    a.textContent = enc.hf_id;
  }
  const values = {
    "enc-name": enc.name, "enc-params": enc.params, "enc-hours": enc.hours,
    "enc-states": enc.layers && String(enc.layers + 1), "enc-dim": enc.dim && String(enc.dim),
    "probe-params": enc.probe_params && enc.probe_params.toLocaleString("en-US"),
    "best-acc": pct(best.accuracy, 1),
    "gain": `+${gain} pts`,
    "gain-words": `${gain} points`,
    "protocol-text": R.protocol === "cv5" ? `${R.protocol_text} · ± ${pct(best.std, 1)}` : R.protocol_text,
    "layer-span": sel.length > 1 ? `layers ${sel[0]}–${sel[sel.length - 1]}` : `layer ${sel[0]}`,
    "pool-title": sel.length > 1 ? "Average time and layers" : "Average over time",
    "pool-text": sel.length > 1
      ? `Frames from layers ${sel[0]}–${sel[sel.length - 1]} are mean-pooled over time and across layers`
      : `Frames from layer ${sel[0]} are mean-pooled over time`,
  };
  document.querySelectorAll("[data-bind]").forEach((el) => {
    const v = values[el.dataset.bind];
    if (v) el.textContent = el.closest(".lede") && el.dataset.bind === "best-acc" ? pct(best.accuracy) : v;
  });
}

/* ---------------- 01 model comparison (HTML bars) ---------------- */
function renderModels(R) {
  const root = $("#model-chart");
  const models = [...R.models].sort((a, b) => b.accuracy - a.accuracy);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  root.replaceChildren();
  const refs = [
    { v: R.chance, label: `chance ${pct(R.chance, 0)}`, short: "chance" },
    { v: R.human, label: `human listeners ${pct(R.human, 0)}`, short: "humans" },
  ].map((r) => {
    const el = h("div", { class: "bars-ref" },
      h("span", {}, h("span", { class: "ref-long" }, r.label), h("span", { class: "ref-short" }, r.short)));
    root.append(el);
    return { ...r, el };
  });
  const rows = models.map((m) => {
    const pre = m.family === "pretrained";
    const bar = h("div", { class: `bar${pre ? " pre" : ""}`, style: "width:0" });
    const val = h("span", { class: "bar-val" }, pct(m.accuracy, 1) + (m.std ? ` ± ${(m.std * 100).toFixed(1)}` : ""));
    const track = h("div", { class: "bar-track" }, bar, val);
    let whisker = null;
    if (m.std) { whisker = h("div", { class: "whisker" }); track.append(whisker); }
    const mixed = R.protocol === "cv5" && m.protocol === "legacy";
    const row = h("div", { class: `bar-row${m.id === R.best ? " best" : ""}`, tabindex: "0" },
      h("span", { class: "bar-name", title: m.name }, h("span", {}, m.name, mixed ? h("sup", { class: "mark" }, "†") : null)), track);
    hoverTip(row, () => `<b>${m.name}</b>` + ttRows([
      ["Accuracy", pct(m.accuracy, 1) + (m.std ? ` ± ${pct(m.std, 1)}` : "")],
      ["Macro F1", m.macro_f1.toFixed(3)],
      ...(m.legacy_accuracy ? [["Original split", pct(m.legacy_accuracy, 1)]] : []),
    ]));
    root.append(row);
    return { m, bar, val, track, whisker };
  });
  const axis = h("div", { class: "bars-axis" }, h("span"),
    h("div", { class: "ticks" }, ticks.map((t) => h("span", { style: `left:${t * 100}%` }, pct(t)))));
  root.append(axis);

  const layout = () => {
    const first = rows[0].track;
    const left = first.offsetLeft, width = first.clientWidth;
    for (const r of refs) r.el.style.left = `${left + r.v * width}px`;
    rows.forEach(({ m, bar, val, whisker }) => {
      bar.style.width = `${m.accuracy * 100}%`;
      const tipX = (m.accuracy + (m.std || 0)) * 100;
      val.style.left = `${tipX}%`;
      if (whisker) { whisker.style.left = `${(m.accuracy - m.std) * 100}%`; whisker.style.width = `${m.std * 200}%`; }
    });
  };
  requestAnimationFrame(() => requestAnimationFrame(layout));
  new ResizeObserver(layout).observe(root);

  $("#model-note").textContent = R.protocol === "cv5"
    ? "Mean over 5 folds of ~18 unseen actors each; whiskers show ±1 standard deviation across folds." +
      (R.models.some((m) => m.protocol === "legacy") ? " † Original 72/19-actor split (the 5-fold re-run of the hand-crafted models is pending)." : "")
    : "Every model is scored on the same 19 held-out actors (1,552 clips).";
  $("#model-table").replaceChildren(table(
    [{ label: "Model" }, { label: "Features" }, { label: "Accuracy", num: true }, { label: "Macro F1", num: true }],
    models.map((m) => [m.name, m.family, pct(m.accuracy, 1) + (m.std ? ` ± ${pct(m.std, 1)}` : ""), m.macro_f1.toFixed(3)])));
}

/* ---------------- 02 layer scan (SVG line) ---------------- */
function renderLayers(R) {
  const { scores, selected, model } = R.layers;
  const root = $("#layer-chart");
  $("#layer-title").textContent = `${model}: linear-probe accuracy by layer`;
  const peak = scores.indexOf(Math.max(...scores));
  const multi = selected.length > 1;
  $("#layer-aside").textContent = `Best single layer: ${peak} (${pct(scores[peak], 1)}). ` +
    (multi ? `The deployed probe averages layers ${selected[0]}–${selected[selected.length - 1]}.`
      : `The deployed probe reads layer ${selected[0]}.`);
  $("#layer-legend").textContent = multi ? "Layers the deployed model averages" : "Layer the deployed model reads";

  const draw = () => {
    const W = Math.max(300, root.clientWidth), H = W < 480 ? 240 : 290;
    const m = { l: 46, r: 14, t: 30, b: 38 };
    const lo = Math.floor((Math.min(...scores) - 0.02) * 20) / 20;
    const hi = Math.ceil((Math.max(...scores) + 0.01) * 20) / 20;
    const x = (i) => m.l + (i / (scores.length - 1)) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Probe accuracy by layer, peak at layer ${peak}` });

    const grid = s("g", { class: "grid" });
    for (let v = lo; v <= hi + 1e-9; v += 0.05) {
      grid.append(s("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      const t = s("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }); t.textContent = pct(v);
      const g = s("g", { class: "tick" }); g.append(t); svg.append(g);
    }
    svg.prepend(grid);
    const bx0 = x(Math.max(0, selected[0] - 0.5)), bx1 = x(Math.min(scores.length - 1, selected[selected.length - 1] + 0.5));
    svg.append(s("rect", { class: "band", x: bx0, y: m.t - 18, width: bx1 - bx0, height: H - m.t - m.b + 18, rx: 6 }));
    const bl = s("text", { class: "band-label", x: (bx0 + bx1) / 2, y: m.t - 6, "text-anchor": "middle" });
    bl.textContent = multi ? "averaged" : "used"; svg.append(bl);

    const pts = scores.map((v, i) => [x(i), y(v)]);
    svg.append(s("path", { class: "area", d: `M${pts[0][0]},${y(lo)} L${pts.map((p) => p.join(",")).join(" L")} L${pts.at(-1)[0]},${y(lo)} Z` }));
    svg.append(s("path", { class: "series", d: `M${pts.map((p) => p.join(",")).join(" L")}` }));
    const dots = pts.map(([px, py]) => { const c = s("circle", { class: "dot", cx: px, cy: py, r: 4 }); svg.append(c); return c; });
    const pl = s("text", { class: "peak-label", x: pts[peak][0], y: pts[peak][1] - 12, "text-anchor": "middle" });
    pl.textContent = pct(scores[peak], 1); svg.append(pl);

    // label every layer only when there is room for it (~22 px per label)
    const every = [1, 2, 4, 6].find((k) => (W - m.l - m.r) / ((scores.length - 1) / k) >= 22) || 8;
    scores.forEach((_, i) => {
      if (i % every) return;
      const t = s("text", { x: x(i), y: H - m.b + 18, "text-anchor": "middle" }); t.textContent = i;
      const g = s("g", { class: "tick" }); g.append(t); svg.append(g);
    });
    const xl = s("text", { class: "axis-label", x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" });
    xl.textContent = "encoder layer"; svg.append(xl);

    const cross = s("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
    svg.append(cross);
    const hit = s("rect", { class: "hit", x: m.l - 10, y: 0, width: W - m.l - m.r + 20, height: H });
    hit.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      const i = Math.max(0, Math.min(scores.length - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (scores.length - 1))));
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
      dots.forEach((d, j) => d.setAttribute("r", j === i ? 6 : 4));
      showTip(`<b>Layer ${i}</b>${i === 0 ? " · CNN front end" : ""}` + ttRows([
        ["CV accuracy", pct(scores[i], 1)], ["Averaged", selected.includes(i) ? "yes" : "no"]]), e.clientX, e.clientY);
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("r", 4)); hideTip(); });
    svg.append(hit);
    root.replaceChildren(svg);
  };
  draw();
  new ResizeObserver(draw).observe(root);
  $("#layer-table").replaceChildren(table(
    [{ label: "Layer" }, { label: "CV accuracy", num: true }, { label: "Averaged" }],
    scores.map((v, i) => [String(i), pct(v, 1), selected.includes(i) ? "yes" : ""])));
}

/* ---------------- 03 confusion heatmap ---------------- */
function renderConfusion(R) {
  const { labels, models } = R.confusion;
  const ids = [R.best, ...Object.keys(models).filter((k) => k !== R.best)].filter((k) => models[k]);
  const draw = (id) => {
    const M = models[id].matrix;
    const rates = M.map((row) => { const n = row.reduce((a, b) => a + b, 0); return row.map((c) => c / n); });
    const grid = $("#heatmap");
    grid.replaceChildren(h("span"));
    labels.forEach((l) => grid.append(h("span", { class: "hm-label hm-col", title: l },
      h("span", { class: "dot", style: `background:${emoVar(l)}` }), l)));
    labels.forEach((t, i) => {
      grid.append(h("span", { class: "hm-label" }, h("span", { class: "dot", style: `background:${emoVar(t)}` }), t));
      labels.forEach((p, j) => {
        const v = rates[i][j];
        const cell = h("span", {
          class: `hm-cell${i === j ? " diag" : ""}`, tabindex: "0",
          style: `background:color-mix(in oklab, var(--seq-hi) ${Math.round(v * 100)}%, var(--surface-2));` +
                 `color:${v >= 0.45 ? "#fff" : "var(--ink-2)"}`,
        }, v >= 0.05 ? `${Math.round(v * 100)}` : "");
        hoverTip(cell, `<b>${cap(t)}</b> heard as <b>${p}</b>` + ttRows([["Share of row", pct(v, 1)], ["Clips", String(M[i][j])]]));
        grid.append(cell);
      });
    });
    const off = [];
    rates.forEach((row, i) => row.forEach((v, j) => { if (i !== j) off.push({ v, t: labels[i], p: labels[j] }); }));
    off.sort((a, b) => b.v - a.v);
    const recall = rates.map((row, i) => ({ v: row[i], l: labels[i] })).sort((a, b) => b.v - a.v);
    $("#confusion-insights").replaceChildren(
      ...off.slice(0, 3).map((o) => h("li", {}, h("span", { class: "pct" }, pct(o.v)), `of ${o.t} clips are heard as ${o.p}`)),
      h("li", {}, h("span", { class: "pct" }, pct(recall[0].v)), `of ${recall[0].l} clips are recognised, the easiest class`),
      h("li", {}, h("span", { class: "pct" }, pct(recall.at(-1).v)), `of ${recall.at(-1).l} clips, the hardest`));
    $("#confusion-table").replaceChildren(table(
      [{ label: "True \\ predicted" }, ...labels.map((l) => ({ label: l, num: true }))],
      labels.map((l, i) => [l, ...rates[i].map((v) => pct(v, 1))])));
  };
  segmented($("#confusion-picker"), ids.map((k) => ({ value: k, label: models[k].name.replace("Hand-crafted · ", "Hand-crafted ") })), ids[0], draw);
  $("#confusion-picker").hidden = false;
  draw(ids[0]);
}

/* ---------------- 05 actor split ---------------- */
function renderActors(R) {
  const grid = $("#actor-grid");
  const dots = R.actors.map((a) => { const d = h("span", { class: "actor", "data-actor": a }); grid.append(d); return d; });
  let current = 0, timer = null;
  const show = (k) => {
    current = k;
    const test = new Set(R.splits[k]);
    dots.forEach((d) => d.classList.toggle("test", test.has(Number(d.dataset.actor))));
    $("#legend-test").textContent = `${test.size} test voices` + (R.splits.length > 1 ? ` in fold ${k + 1}` : "");
    $("#legend-train").textContent = `${R.actors.length - test.size} training voices`;
  };
  dots.forEach((d) => hoverTip(d, () => {
    const a = Number(d.dataset.actor), isTest = R.splits[current].includes(a);
    return `<b>Actor ${a}</b><br>${isTest ? "Test voice: never heard in training" : "Training voice"}`;
  }));
  if (R.splits.length > 1) {
    $("#actor-title").textContent = "Five folds: every voice is tested exactly once";
    const buttons = segmented($("#fold-picker"), R.splits.map((_, k) => ({ value: k, label: `Fold ${k + 1}` })), 0,
      (k) => { clearInterval(timer); show(k); });
    timer = setInterval(() => {
      const k = (current + 1) % R.splits.length;
      buttons.forEach((b, i) => b.setAttribute("aria-checked", String(i === k)));
      show(k);
    }, 2600);
  } else {
    $("#actor-title").textContent = "The 91 CREMA-D actors, split by voice";
  }
  show(0);
}

/* ---------------- 07 roadmap ---------------- */
function renderRoadmap(R) {
  const label = { done: "Done", running: "Running", next: "Next" };
  $("#roadmap").replaceChildren(...R.roadmap.map((r) => h("li", { class: "road card" },
    h("span", { class: `badge ${r.status}` }, label[r.status]), h("h3", {}, r.title), h("p", {}, r.text))));
}

/* ---------------- 04 pipeline illustrations ---------------- */
function renderStepArt(R) {
  const W = 200, H = 76;
  const art = {
    wave() {
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
      for (let i = 0; i < 48; i++) {
        const env = Math.sin((i / 47) * Math.PI) ** 1.4;
        const a = (0.15 + 0.85 * env) * (0.45 + 0.55 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.6)));
        const bh = Math.max(3, a * 50);
        svg.append(s("rect", { x: 16 + i * 3.5, y: H / 2 - bh / 2, width: 2, height: bh, rx: 1, fill: "var(--ink)", opacity: 0.85 }));
      }
      return svg;
    },
    layers() {
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
      const n = R.layers.scores.length, sel = new Set(R.layers.selected);
      const pitch = (H - 20) / (n - 1), bh = Math.min(3, pitch * 0.55);  // keep a gap between bars, even at 25 layers
      for (let i = 0; i < n; i++) {
        const yy = H - 10 - i * pitch;
        svg.append(s("rect", { x: 50, y: yy - bh / 2, width: 100, height: bh, rx: bh / 2, fill: sel.has(i) ? "var(--line-series)" : "var(--axis)" }));
      }
      return svg;
    },
    pool() {
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
      for (let c = 0; c < 9; c++) for (let r = 0; r < 6; r++)
        svg.append(s("rect", { x: 26 + c * 9, y: 13 + r * 9, width: 6, height: 6, rx: 1.5, fill: "var(--axis)" }));
      svg.append(s("path", { d: "M118 38 h22 m-6 -5 l6 5 l-6 5", stroke: "var(--muted)", "stroke-width": 1.5, fill: "none", "stroke-linecap": "round" }));
      for (let r = 0; r < 6; r++) svg.append(s("rect", { x: 154, y: 13 + r * 9, width: 6, height: 6, rx: 1.5, fill: "var(--ink)" }));
      return svg;
    },
    linear() {
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
      const ins = [14, 24, 34, 44, 54, 64], outs = EMOTIONS.map((_, i) => 13 + i * 10);
      ins.forEach((yi) => outs.forEach((yo) => svg.append(s("line", { x1: 64, y1: yi, x2: 136, y2: yo, stroke: "var(--axis)", "stroke-width": 0.8 }))));
      ins.forEach((yi) => svg.append(s("circle", { cx: 64, cy: yi, r: 3, fill: "var(--ink)" })));
      EMOTIONS.forEach((e, i) => svg.append(s("circle", { cx: 136, cy: outs[i], r: 3.5, fill: emoVar(e) })));
      return svg;
    },
    probs() {
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
      const vals = [0.08, 0.05, 0.12, 0.62, 0.04, 0.09];
      EMOTIONS.forEach((e, i) => {
        const bh = 6 + vals[i] * 70;
        svg.append(s("rect", { x: 52 + i * 17, y: H - 12 - bh, width: 11, height: bh, rx: 2.5, fill: emoVar(e) }));
      });
      svg.append(s("line", { x1: 46, x2: 154, y1: H - 11, y2: H - 11, stroke: "var(--axis)", "stroke-width": 1 }));
      return svg;
    },
  };
  document.querySelectorAll("[data-art]").forEach((el) => el.replaceChildren(art[el.dataset.art]()));
}

/* ---------------- demo ---------------- */
function initDemo(config) {
  const stage = $("#stage"), canvas = $("#wave"), status = $("#stage-status"), recBtn = $("#rec-btn");
  const resultEl = $("#result"), errorEl = $("#demo-error"), timelineEl = $("#timeline");
  const playBtn = $("#play-btn"), tools = $("#stage-tools");
  const busyPill = h("div", { class: "chip-btn", style: "position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none" }, "Analysing…");
  busyPill.hidden = true;
  stage.append(busyPill);
  const audio = new Audio();
  const models = Object.fromEntries(config.models.map((m) => [m.id, m]));
  const state = { model: config.default_model, samples: null, result: null, source: null, recording: null, levels: [] };
  // A static deployment (e.g. Netlify without a model backend) still plays the samples from precomputed results.
  const live = config.backend !== false;
  const OFFLINE = "Live analysis isn't connected on this deployment yet. The CREMA-D sample clips below still work.";
  const IDLE = live ? "Tap to record · speak for 2–5 seconds" : "Recording is offline on this deployment · pick a sample clip below";
  status.textContent = IDLE;
  if (!live) {
    $("#demo").classList.add("offline");
    $("#demo-title-text").textContent = "Sample demo";
    $("#upload-link").hidden = true;
    $("#cta-label").textContent = "Try the demo";
    recBtn.setAttribute("aria-disabled", "true");
  }
  let actx = null;
  const ctx = () => (actx ||= new (window.AudioContext || window.webkitAudioContext)());

  segmented($("#model-picker"), config.models.map((m) => ({ value: m.id, label: m.accuracy ? `${m.name} · ${pct(m.accuracy)}` : m.name })),
    state.model, (id) => { state.model = id; if (state.samples) classify(); });

  // samples
  const sampleBtns = config.examples.map((ex) => h("button", {
    class: "sample", type: "button", "aria-pressed": "false", title: `CREMA-D actor ${ex.actor}, acted as ${ex.emotion}`,
    onclick: async (e) => {
      const btn = e.currentTarget;
      try {
        const buf = await (await fetch(ex.url)).arrayBuffer();
        await load(buf, { kind: "sample", ...ex });
        markSample(btn);
      } catch (err) { fail("Could not load that sample."); }
    },
  }, h("span", { class: "dot", style: `background:${emoVar(ex.emotion)}` }), ex.emotion));
  const markSample = (btn) => sampleBtns.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
  $("#sample-list").replaceChildren(...sampleBtns);

  // drawing
  function sizeCanvas() {
    const dpr = devicePixelRatio || 1, r = canvas.getBoundingClientRect();
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    return { g: canvas.getContext("2d"), W: canvas.width, H: canvas.height, dpr };
  }
  function drawLive() {
    const { g, W, H, dpr } = sizeCanvas();
    g.clearRect(0, 0, W, H);
    const bw = 3 * dpr, gap = 2 * dpr, n = Math.floor(W / (bw + gap));
    const lv = state.levels.slice(-n);
    g.fillStyle = cssVar("--rec");
    lv.forEach((v, i) => {
      const bh = Math.max(2 * dpr, Math.min(1, v * 4) * H * 0.42);
      const xx = W - (lv.length - i) * (bw + gap);
      g.globalAlpha = 0.35 + 0.65 * (i / lv.length);
      g.beginPath(); g.roundRect(xx, H * 0.28 - bh / 2, bw, bh, bw / 2); g.fill();
    });
    g.globalAlpha = 1;
  }
  function emotionAt(t) {
    const r = state.result;
    if (!r) return null;
    if (r.segments.length < 2) return topOf(r.scores);
    const segs = timelineRegions(r);
    return (segs.find((sg) => t >= sg.a && t < sg.b) || segs.at(-1)).emotion;
  }
  function drawStatic() {
    if (!state.samples) return;
    const { g, W, H, dpr } = sizeCanvas();
    g.clearRect(0, 0, W, H);
    const data = state.samples, dur = data.length / SAMPLE_RATE;
    const bottom = state.result && state.result.segments.length > 1 ? 50 * dpr : 14 * dpr;
    const top = 54 * dpr, mid = top + (H - top - bottom) / 2, amp = (H - top - bottom) / 2;
    const bw = 2 * dpr, gap = 1.5 * dpr, n = Math.floor((W - 24 * dpr) / (bw + gap)), x0 = 12 * dpr;
    const per = Math.max(1, Math.floor(data.length / n));
    let peakAll = 1e-6;
    for (let i = 0; i < data.length; i += 32) peakAll = Math.max(peakAll, Math.abs(data[i]));
    const played = audio.duration ? audio.currentTime / audio.duration : 0;
    const colors = {};
    for (const e of EMOTIONS) colors[e] = cssVar(`--${e}`);
    const base = cssVar("--axis");
    for (let i = 0; i < n; i++) {
      let pk = 0;
      for (let j = i * per; j < Math.min(data.length, (i + 1) * per); j += 4) pk = Math.max(pk, Math.abs(data[j]));
      const bh = Math.max(1.5 * dpr, (pk / peakAll) * amp * 0.95 * 2);
      const frac = i / n, emo = emotionAt(frac * dur);
      g.fillStyle = emo ? colors[emo] : base;
      g.globalAlpha = state.result ? (frac <= played && played > 0 ? 1 : 0.55) : 0.9;
      g.beginPath(); g.roundRect(x0 + i * (bw + gap), mid - bh / 2, bw, bh, bw / 2); g.fill();
    }
    g.globalAlpha = 1;
    if (played > 0 && played < 1) {
      g.fillStyle = cssVar("--ink");
      g.fillRect(x0 + played * n * (bw + gap), top - 6 * dpr, 1.5 * dpr, H - top - bottom + 12 * dpr);
    }
  }
  themeListeners.push(drawStatic);
  new ResizeObserver(() => (state.recording ? drawLive() : drawStatic())).observe(stage);

  // timeline
  function timelineRegions(r) {
    const segs = r.segments, dur = r.duration;
    const centers = segs.map((sg) => (sg.start + sg.end) / 2);
    return segs.map((sg, i) => ({
      a: i === 0 ? 0 : (centers[i - 1] + centers[i]) / 2,
      b: i === segs.length - 1 ? dur : (centers[i] + centers[i + 1]) / 2,
      emotion: topOf(sg.scores), sg,
    }));
  }
  function renderTimeline(r) {
    timelineEl.replaceChildren();
    timelineEl.hidden = r.segments.length < 2;
    if (timelineEl.hidden) return;
    const regions = timelineRegions(r);
    const merged = [];
    for (const rg of regions) {
      const last = merged.at(-1);
      if (last && last.emotion === rg.emotion) { last.b = rg.b; last.parts.push(rg.sg); }
      else merged.push({ ...rg, parts: [rg.sg] });
    }
    for (const m of merged) {
      const w = (m.b - m.a) / r.duration;
      const el = h("div", { class: "seg", style: `flex:${w};background:${emoVar(m.emotion)}` });
      requestAnimationFrame(() => { el.textContent = el.clientWidth > 64 ? m.emotion : m.emotion[0].toUpperCase(); });
      hoverTip(el, `<b>${m.a.toFixed(1)}–${m.b.toFixed(1)} s · ${m.emotion}</b>` +
        ttRows(Object.entries(avgScores(m.parts)).sort((x, y) => y[1] - x[1]).slice(0, 3).map(([k, v]) => [k, pct(v)])));
      timelineEl.append(el);
    }
  }
  const topOf = (scores) => Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  const avgScores = (parts) => Object.fromEntries(EMOTIONS.map((e) => [e, parts.reduce((t, p) => t + p.scores[e], 0) / parts.length]));

  // result
  function renderResult(r) {
    const ranked = Object.entries(r.scores).sort((a, b) => b[1] - a[1]);
    const [[top, conf], [second, p2]] = ranked;
    const em = $("#result-emotion");
    em.querySelector(".swatch").style.background = emoVar(top);
    em.querySelector(".name").textContent = top;
    $("#result-conf").textContent = pct(conf);
    let sentence = conf < 0.45 ? `Hard to call: closest to ${top}, then ${second}.`
      : p2 >= 0.15 ? `Mostly ${top}, with a trace of ${second}.` : `Confidently ${top}.`;
    if (r.segments.length > 1) {
      const runs = timelineRegions(r).map((x) => x.emotion).filter((e, i, a) => i === 0 || e !== a[i - 1]);
      sentence += runs.length > 1 ? ` Over time it moves ${runs.join(" → ")}.` : ` Steady across all ${r.segments.length} windows.`;
    }
    if (state.source?.kind === "sample") {
      sentence += ` The actor was asked to sound ${state.source.emotion}: ${state.source.emotion === top ? "a match" : "a miss"}.`;
    }
    $("#result-sentence").textContent = sentence;
    const list = $("#prob-list");
    list.replaceChildren(...ranked.map(([e, p], i) => {
      const fill = h("div", { class: "prob-fill", style: `background:${emoVar(e)}` });
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${Math.max(p * 100, 0.6)}%`; }));
      return h("div", { class: `prob${i === 0 ? " top" : ""}` }, h("span", { class: "prob-name" }, e),
        h("div", { class: "prob-track" }, fill), h("span", { class: "prob-val" }, pct(p)));
    }));
    const m = models[r.model];
    const src = state.source?.kind === "sample" ? `CREMA-D actor ${state.source.actor} · ` : "";
    $("#result-meta").textContent = `${src}${m.name} · ${r.duration.toFixed(1)} s of audio · ` +
      (r.precomputed ? "precomputed" : `${r.latency_ms} ms`) +
      (m.calibrated ? " · calibrated" : "");
    resultEl.hidden = false;
  }

  function fail(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = false;
    stage.classList.remove("busy");
    busyPill.hidden = true;
  }

  function show(result) {
    state.result = result;
    renderTimeline(result);
    renderResult(result);
    drawStatic();
  }

  async function classify() {
    errorEl.hidden = true;
    const precomputed = state.source?.kind === "sample" && state.source.results?.[state.model];
    if (precomputed) return show({ ...precomputed, precomputed: true });
    if (!live) return fail(OFFLINE);
    stage.classList.add("busy");
    busyPill.hidden = false;
    try {
      const res = await fetch(`/api/predict?model=${encodeURIComponent(state.model)}`, {
        method: "POST", headers: { "Content-Type": "audio/wav" }, body: encodeWav(state.samples),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) { state.result = null; resultEl.hidden = true; drawStatic(); return fail(body.detail || `Server error (${res.status}).`); }
      show(body);
    } catch (e) {
      fail("Could not reach the model server.");
    } finally {
      stage.classList.remove("busy");
      busyPill.hidden = true;
    }
  }

  async function load(arrayBuffer, source) {
    try {
      const decoded = await ctx().decodeAudioData(arrayBuffer.slice(0));
      const dur = Math.min(decoded.duration, config.max_seconds);
      const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(dur * SAMPLE_RATE)), SAMPLE_RATE);
      const src = off.createBufferSource();
      src.buffer = decoded; src.connect(off.destination); src.start(0, 0, dur);
      state.samples = (await off.startRendering()).getChannelData(0);
    } catch (e) {
      return fail("This file could not be decoded. Try WAV, MP3 or M4A.");
    }
    state.source = source;
    if (source.kind !== "sample") markSample(null);
    state.result = null;
    resultEl.hidden = true;
    timelineEl.hidden = true;
    audio.pause();
    if (audio.src) URL.revokeObjectURL(audio.src);
    audio.src = URL.createObjectURL(encodeWav(state.samples));
    stage.classList.add("has-audio");
    tools.hidden = false;
    drawStatic();
    await classify();
  }

  // recording
  async function startRecording() {
    errorEl.hidden = true;
    if (!live) return fail(OFFLINE);
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) return fail("Recording is not supported in this browser. Upload a file instead.");
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } }); }
    catch (e) { return fail("Microphone access was blocked. Allow it in the browser, or upload a file."); }
    const analyser = ctx().createAnalyser();
    analyser.fftSize = 1024;
    ctx().createMediaStreamSource(stream).connect(analyser);
    const rec = new MediaRecorder(stream), chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      state.recording = null;
      stage.classList.remove("recording");
      recBtn.setAttribute("aria-label", "Start recording");
      const blob = new Blob(chunks, { type: rec.mimeType });
      if ((performance.now() - t0) < 600) { status.textContent = "Too short. Hold on a little longer."; return drawLive(); }
      await load(await blob.arrayBuffer(), { kind: "recording" });
    };
    const t0 = performance.now();
    const buf = new Float32Array(analyser.fftSize);
    state.levels = [];
    state.recording = { rec, stop: () => rec.state !== "inactive" && rec.stop() };
    stage.classList.add("recording");
    recBtn.setAttribute("aria-label", "Stop recording");
    rec.start();
    const tick = () => {
      if (!state.recording) return;
      analyser.getFloatTimeDomainData(buf);
      let sum = 0; for (const v of buf) sum += v * v;
      state.levels.push(Math.sqrt(sum / buf.length));
      const sec = (performance.now() - t0) / 1000;
      status.textContent = `Recording ${Math.floor(sec)}.${Math.floor((sec % 1) * 10)} s · tap to stop`;
      drawLive();
      if (sec >= MAX_RECORD_SECONDS) state.recording.stop(); else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  recBtn.addEventListener("click", () => (state.recording ? state.recording.stop() : startRecording()));

  // playback
  playBtn.addEventListener("click", () => (audio.paused ? audio.play() : audio.pause()));
  audio.addEventListener("play", () => { playBtn.classList.add("playing"); playBtn.lastElementChild.textContent = "Pause"; const step = () => { drawStatic(); if (!audio.paused) requestAnimationFrame(step); }; step(); });
  audio.addEventListener("pause", () => { playBtn.classList.remove("playing"); playBtn.lastElementChild.textContent = "Play"; drawStatic(); });
  audio.addEventListener("ended", () => { audio.currentTime = 0; drawStatic(); });

  $("#reset-btn").addEventListener("click", () => {
    audio.pause();
    Object.assign(state, { samples: null, result: null, source: null });
    markSample(null);
    stage.classList.remove("has-audio");
    tools.hidden = true; timelineEl.hidden = true; resultEl.hidden = true; errorEl.hidden = true;
    status.textContent = IDLE;
    sizeCanvas().g.clearRect(0, 0, canvas.width, canvas.height);
  });

  // upload + drag and drop
  $("#file-input").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (f && !live) fail(OFFLINE);
    else if (f) await load(await f.arrayBuffer(), { kind: "file", name: f.name });
    e.target.value = "";
  });
  const demo = $("#demo");
  demo.addEventListener("dragover", (e) => { e.preventDefault(); demo.classList.add("dragging"); });
  demo.addEventListener("dragleave", (e) => { if (!demo.contains(e.relatedTarget)) demo.classList.remove("dragging"); });
  demo.addEventListener("drop", async (e) => {
    e.preventDefault(); demo.classList.remove("dragging");
    const f = e.dataTransfer.files[0];
    if (f && !live) fail(OFFLINE);
    else if (f) await load(await f.arrayBuffer(), { kind: "file", name: f.name });
  });

  $("#cta-record").addEventListener("click", (e) => {
    e.preventDefault();
    demo.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    recBtn.focus({ preventScroll: true });
  });
}

function encodeWav(samples, sr = SAMPLE_RATE) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/* ---------------- boot ---------------- */
(async () => {
  try {
    const [config, results] = await Promise.all([
      fetch("/api/config").then((r) => r.json()),
      fetch("/api/results").then((r) => r.json()),
    ]);
    bindNumbers(results);
    renderModels(results);
    renderLayers(results);
    renderConfusion(results);
    renderActors(results);
    renderRoadmap(results);
    renderStepArt(results);
    initDemo(config);
  } catch (e) {
    console.error(e);
    const err = $("#demo-error");
    err.textContent = "The model server is not responding. Reload the page in a moment.";
    err.hidden = false;
  }
})();
