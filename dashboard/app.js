"use strict";

const DATA_ROOT = new URLSearchParams(location.search).get("data") || "../analytics";
const MAX_POINTS = 2000;
const THEME_KEY = "alife-analytics-theme";

function currentTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function switchTheme(theme) {
  document.documentElement.dataset.theme = theme === "light" ? "light" : "dark";
  try {
    localStorage.setItem(THEME_KEY, currentTheme());
  } catch (e) {}
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = currentTheme() === "light" ? "Dark" : "Light";
  rebuildCharts();
}

const WORLD_COLORS = [
  "#6bc9ff", "#4ecdc4", "#b983ff", "#ff6b6b", "#ffe66d",
  "#ff9f43", "#1dd1a1", "#ff6348", "#f368e0", "#00d2d3",
];

const KIND_CLASS = {
  none: "kind-none",
  random: "kind-random",
  neural_net: "kind-neural",
  llm: "kind-llm",
};

const METRICS = [
  { key: "population", label: "Population", y: (s) => s.agent_count },
  { key: "energy_grid", label: "Energy · grid", y: (s) => s.grid_energy },
  { key: "energy_agent", label: "Energy · agents", y: (s) => s.agent_energy },
  { key: "energy_unplaced", label: "Energy · unplaced", y: (s) => s.unplaced_energy },
  {
    key: "energy_total",
    label: "Energy · total",
    y: (s) => s.grid_energy.map((g, i) => g + s.agent_energy[i] + s.unplaced_energy[i]),
  },
  { key: "births", label: "Births", y: (s) => s.births },
  { key: "deaths", label: "Deaths", y: (s) => s.deaths },
  { key: "net_growth", label: "Net growth", y: (s) => s.births.map((b, i) => b - s.deaths[i]) },
  { key: "move", label: "Actions · move", y: (s) => s.move },
  { key: "split", label: "Actions · split", y: (s) => s.split },
  { key: "absorb", label: "Actions · absorb", y: (s) => s.absorb },
];

const DEFAULT_METRICS = ["population", "energy_total", "net_growth"];

const state = {
  index: null,
  worlds: new Map(),
  selected: new Set(),
  enabled: new Set(DEFAULT_METRICS),
  charts: [],
  syncing: false,
  buildSeq: 0,
  xTicks: 7,
  yFromZero: false,
  points: false,
  search: "",
  rangeLo: 0,
  rangeHi: 0,
  hidden: new Map(),
};

function getHidden(metricKey) {
  let s = state.hidden.get(metricKey);
  if (!s) {
    s = new Set();
    state.hidden.set(metricKey, s);
  }
  return s;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function fmt(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString("en-US");
}

function fmtTick(t) {
  if (t == null || Number.isNaN(t)) return "—";
  if (!Number.isInteger(t)) return t.toFixed(1);
  if (t >= 1e6) return (t / 1e6).toFixed(t % 1e6 ? 1 : 0) + "M";
  if (t >= 1e3) return (t / 1e3).toFixed(t % 1e3 ? 1 : 0) + "k";
  return fmt(t);
}

function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const exp = Math.floor(Math.log10(raw));
  const base = Math.pow(10, exp);
  for (const c of [1, 2, 2.5, 5, 10]) {
    if (raw <= c * base) return c * base;
  }
  return 10 * base;
}

function lttb(xs, ys, threshold) {
  const n = xs.length;
  if (n <= threshold) return { xs, ys };
  const t = Math.max(3, threshold);
  const sampledX = [xs[0]];
  const sampledY = [ys[0]];
  const every = (n - 2) / (t - 2);
  let a = 0;
  for (let i = 0; i < t - 2; i++) {
    const avgStart = Math.floor(i * every) + 1;
    const avgEnd = Math.min(Math.floor((i + 1) * every) + 1, n - 1);
    let avgX = 0;
    let avgY = 0;
    for (let j = avgStart; j < avgEnd; j++) {
      avgX += xs[j];
      avgY += ys[j];
    }
    const avgLen = Math.max(1, avgEnd - avgStart);
    avgX /= avgLen;
    avgY /= avgLen;
    const rangeStart = Math.floor((i + 1) * every) + 1;
    const rangeEnd = Math.min(Math.floor((i + 2) * every) + 1, n - 1);
    const ax = xs[a];
    const ay = ys[a];
    let bestArea = -1;
    let bestIdx = a;
    for (let j = rangeStart; j < rangeEnd; j++) {
      const area = Math.abs((ax - avgX) * (ys[j] - ay) - (ax - xs[j]) * (avgY - ay));
      if (area > bestArea) {
        bestArea = area;
        bestIdx = j;
      }
    }
    a = bestIdx;
    sampledX.push(xs[a]);
    sampledY.push(ys[a]);
  }
  sampledX.push(xs[n - 1]);
  sampledY.push(ys[n - 1]);
  return { xs: sampledX, ys: sampledY };
}

function sliceRange(ticks, ys) {
  const lo = state.rangeLo;
  const hi = state.rangeHi;
  if (hi <= 0) return { ticks, ys };
  const outT = [];
  const outY = [];
  for (let i = 0; i < ticks.length; i++) {
    const x = ticks[i];
    if (x >= lo && x <= hi) {
      outT.push(x);
      outY.push(ys[i]);
    }
  }
  return { ticks: outT, ys: outY };
}

function syncRangeControls(hi) {
  const rmin = document.getElementById("range-min");
  const rmax = document.getElementById("range-max");
  const loIn = document.getElementById("range-lo-in");
  const hiIn = document.getElementById("range-hi-in");
  const m = Math.max(1, hi);
  rmin.max = m;
  rmax.max = m;
  loIn.max = m;
  hiIn.max = m;
  if (state.rangeHi <= 0) {
    state.rangeLo = 0;
    state.rangeHi = m;
  }
  rmin.value = state.rangeLo;
  rmax.value = state.rangeHi;
  loIn.value = state.rangeLo;
  hiIn.value = state.rangeHi;
}

function brainInfo(w) {
  const kind = typeof w.brain.kind === "string" ? w.brain.kind : "unknown";
  const kindClass = KIND_CLASS[kind] || "kind-unknown";
  const label = kind === "unknown" ? "unknown brain" : kind;
  const name = w.brain.name;
  return { kind, kindClass, text: name ? `${label} · ${name}` : label, empty: !w.brain.kind && !w.brain.name };
}

function matchesSearch(w) {
  const q = state.search.trim().toLowerCase();
  if (!q) return true;
  const m = w.meta;
  const bio = brainInfo(m);
  const hay = [m.name, m.source || "", m.brain ? m.brain.name : "", m.brain ? m.brain.kind : "", bio.text]
    .filter(Boolean).join(" ").toLowerCase();
  return hay.includes(q);
}

function renderWorldList() {
  const list = document.getElementById("world-list");
  list.textContent = "";
  const worlds = [...state.worlds.values()].sort((a, b) => a.meta.name.localeCompare(b.meta.name, "en", { numeric: true }));
  for (const w of worlds) {
    const m = w.meta;
    const chartable = m.series_length > 0 && m.file;
    const li = el("li", "world" + (chartable ? "" : " disabled") + (matchesSearch(w) ? "" : " filter-hidden"));
    const label = el("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selected.has(m.id);
    cb.disabled = !chartable;
    cb.addEventListener("change", () => {
      if (cb.checked) state.selected.add(m.id);
      else state.selected.delete(m.id);
      updateSummary();
      rebuildCharts();
    });
    label.appendChild(cb);
    const info = el("div", "world-info");
    info.appendChild(el("span", "world-name", m.name));

    const bio = brainInfo(m);
    if (!bio.empty && m.brain) {
      info.appendChild(el("span", "brain-badge " + bio.kindClass, bio.text));
    }
    if (m.brain && bio.empty) {
      info.appendChild(el("span", "brain-badge kind-unknown", "unknown brain"));
    }

    const cfg = m.config || {};
    const dim = cfg.width && cfg.height ? `${cfg.width}×${cfg.height}` : "?×?";
    const seed = cfg.seed != null ? `seed ${cfg.seed}` : "";
    const start = cfg.starting_agent_count != null
      ? `${cfg.starting_agent_count}→${cfg.starting_energy_cell_count != null ? cfg.starting_energy_cell_count : "?"}`
      : "";
    const rate = m.settings.target_tick_rate != null ? `rate ${m.settings.target_tick_rate}` : "";
    const pop = m.final.agent_count != null ? `pop ${fmt(m.final.agent_count)}` : "";
    const ticks = m.series_length > 0 ? `${fmt(m.series_length)} ticks` : m.final.manifest_tick_count != null
      ? `${fmt(m.final.manifest_tick_count)} ticks (no data)` : "no data";
    const bits = [ticks, pop, dim, seed, start, rate].filter(Boolean);
    info.appendChild(el("span", "world-meta", bits.join(" · ")));
    if (m.flags && m.flags.length) {
      for (const flag of m.flags) info.appendChild(el("span", "flag-chip", flag.replace(/_/g, " ")));
    }
    label.appendChild(info);
    li.appendChild(label);
    list.appendChild(li);
  }
}

function renderMetricBar() {
  const bar = document.getElementById("metric-bar");
  bar.textContent = "";
  for (const metric of METRICS) {
    const btn = el("button", state.enabled.has(metric.key) ? "on" : "", metric.label);
    btn.type = "button";
    btn.addEventListener("click", () => {
      if (state.enabled.has(metric.key)) state.enabled.delete(metric.key);
      else state.enabled.add(metric.key);
      renderMetricBar();
      rebuildCharts();
    });
    bar.appendChild(btn);
  }
}

function updateSummary() {
  const total = state.index ? state.index.count : 0;
  const sel = state.selected.size;
  document.getElementById("summary").textContent = `${sel} / ${total} worlds selected`;
  document.getElementById("data-root").textContent = DATA_ROOT;
}

async function loadWorldData(w) {
  if (w.data) return w.data;
  const resp = await fetch(`${DATA_ROOT}/${w.meta.file}`);
  if (!resp.ok) throw new Error(`failed to load ${w.meta.file}`);
  w.data = await resp.json();
  return w.data;
}

function chartOpts(metricLabel, seriesCfg, xStep, minV) {
  const yScale = { auto: true };
  if (state.yFromZero && minV >= 0) yScale.min = 0;
  return {
    width: 760,
    height: 360,
    padding: [8, 8, 8, 8],
    legend: { show: false }, resize: false,
    cursor: {
      show: true,
      x: true,
      y: true,
      drag: { x: true, y: false },
      focus: { size: 5 },
    },
    scales: {
      x: { time: false, auto: true },
      y: yScale,
    },
    axes: [
      {
        stroke: cssVar("--axis-stroke", "#8b90a0"),
        grid: { stroke: cssVar("--grid-stroke", "#20242f"), width: 1 },
        size: 40,
        ticks: { stroke: cssVar("--tick-stroke", "#556077") },
        font: "11px ui-monospace, monospace",
        incrs: [xStep],
        values: (u, ticks) => ticks.map(fmtTick),
      },
      {
        stroke: cssVar("--axis-stroke", "#8b90a0"),
        grid: { stroke: cssVar("--grid-stroke", "#20242f"), width: 1 },
        size: 52,
        ticks: { stroke: cssVar("--tick-stroke", "#556077") },
        font: "11px ui-monospace, monospace",
      },
    ],
    series: seriesCfg,
    hooks: {
      setScale: [
        (u, key) => {
          if (key !== "x") return;
          if (state.syncing) return;
          const scale = u.scales.x;
          if (scale.min == null || scale.max == null) return;
          state.syncing = true;
          for (const c of state.charts) {
            if (c !== u) c.setScale("x", { min: scale.min, max: scale.max });
          }
          state.syncing = false;
        },
      ],
      dataIdx: [
        (u, idx) => {
          if (u.__leg) u.__leg.cb(u, idx);
        },
      ],
      ready: [
        (u) => {
          if (u.__leg && u.data[0].length) u.__leg.cb(u, u.data[0].length - 1);
        },
      ],
    },
  };
}

function makeLegend(card, chart, names, colors, metricKey, worldIds) {
  const leg = el("div", "chart-legend");
  const hdr = el("div", "leg-hdr", "tick —");
  leg.appendChild(hdr);
  const rows = [];
  const valEls = [];
  const hiddenSet = getHidden(metricKey);
  names.forEach((name, i) => {
    const row = el("div", "leg-row");
    const dot = el("span", "leg-dot");
    dot.style.background = colors[i];
    const nm = el("span", "leg-name", name);
    const val = el("span", "leg-val", "—");
    row.append(dot, nm, val);
    const si = i + 1;
    if (hiddenSet.has(worldIds[i])) row.classList.add("hidden");
    row.addEventListener("click", () => {
      const show = !chart.series[si].show;
      chart.setSeries(si, { show });
      row.classList.toggle("hidden", !show);
      if (show) hiddenSet.delete(worldIds[i]);
      else hiddenSet.add(worldIds[i]);
    });
    leg.appendChild(row);
    rows.push(row);
    valEls.push(val);
  });
  card.appendChild(leg);
  const L = {
    hdr,
    rows,
    valEls,
    cb(u, idx) {
      if (idx == null || idx < 0) {
        this.hdr.textContent = "tick —";
        for (const v of this.valEls) v.textContent = "—";
        return;
      }
      this.hdr.textContent = "tick " + fmt(u.data[0][idx]);
      for (let i = 0; i < this.valEls.length; i++) {
        const v = u.data[i + 1][idx];
        this.valEls[i].textContent = v == null ? "—" : fmt(Math.round(v * 100) / 100);
      }
    },
  };
  chart.__leg = L;
  return L;
}

function chartWidth() {
  const grid = document.getElementById("charts");
  const rect = grid.getBoundingClientRect();
  const avail = rect.width - 60;
  if (avail <= 0) return 760;
  return Math.max(320, Math.floor(avail));
}

async function renderChart(metric, seq, width) {
  const ids = [...state.selected];
  if (ids.length === 0) return;
  const grid = document.getElementById("charts");
  const card = el("div", "chart-card");
  card.appendChild(el("h3", null, metric.label));
  grid.appendChild(card);

  const ws = [];
  try {
    for (const id of ids) {
      const w = state.worlds.get(id);
      if (!w.meta.file) continue;
      await loadWorldData(w);
      if (seq !== state.buildSeq) return;
      ws.push(w);
    }
  } catch (err) {
    if (seq !== state.buildSeq) return;
    card.appendChild(el("div", "placeholder", err.message));
    return;
  }
  if (ws.length === 0) {
    if (seq !== state.buildSeq) return;
    card.appendChild(el("div", "placeholder", "no data"));
    return;
  }

  let maxTick = 0;
  for (const w of ws) {
    const t = w.data.series.tick;
    const last = t[t.length - 1];
    if (last > maxTick) maxTick = last;
  }
  syncRangeControls(maxTick);

  const sampled = ws.map((w) => {
    const ticks = w.data.series.tick;
    const ys = metric.y(w.data.series);
    const t = sliceRange(ticks, ys);
    return lttb(t.ticks, t.ys, MAX_POINTS);
  });
  const union = [...new Set(sampled.flatMap((p) => p.xs))].sort((a, b) => a - b);
  const xIndex = new Map(union.map((x, i) => [x, i]));
  const data = [union];
  const seriesCfg = [
    { label: "tick", stroke: "transparent", width: 0 },
  ];
  sampled.forEach((p, i) => {
    const vals = new Array(union.length).fill(null);
    for (let j = 0; j < p.xs.length; j++) vals[xIndex.get(p.xs[j])] = p.ys[j];
    data.push(vals);
    seriesCfg.push({
      label: ws[i].meta.name,
      stroke: WORLD_COLORS[i % WORLD_COLORS.length],
      width: 1.5,
      spanGaps: false,
      show: !getHidden(metric.key).has(ws[i].meta.id),
      points: state.points ? { show: true, size: 2 } : { show: false },
    });
  });

  let minV = Infinity;
  for (const p of sampled) {
    for (const v of p.ys) {
      if (v < minV) minV = v;
    }
  }
  if (!Number.isFinite(minV)) minV = 0;
  const span = (union[union.length - 1] - union[0]) || 1;
  const maxSteps = Math.max(2, Math.floor((width - 60) / 50));
  const desired = Math.min(Math.max(2, state.xTicks), maxSteps);
  const xStep = Math.max(1, niceStep(span / (desired - 1)));

  const opts = chartOpts(metric.label, seriesCfg, xStep, minV);
  opts.width = width;
  const chart = new uPlot(opts, data, card);
  makeLegend(card, chart, ws.map((w) => w.meta.name),
    sampled.map((_, i) => WORLD_COLORS[i % WORLD_COLORS.length]),
    metric.key, ws.map((w) => w.meta.id));
  if (chart.data[0].length) chart.__leg.cb(chart, chart.data[0].length - 1);
  card.addEventListener("mouseleave", () => {
    if (chart.__leg) chart.__leg.cb(chart, -1);
  });
  state.charts.push(chart);
}

function rebuildCharts() {
  for (const c of state.charts) c.destroy();
  state.charts = [];
  state.buildSeq += 1;
  const seq = state.buildSeq;
  const grid = document.getElementById("charts");
  grid.textContent = "";
  const hint = document.getElementById("empty-hint");
  if (state.selected.size === 0) {
    hint.textContent = "Select worlds from the left panel to compare series.";
    grid.style.display = "none";
    return;
  }
  grid.style.display = "";
  hint.textContent = "";
  const width = chartWidth();
  for (const metric of METRICS) {
    if (state.enabled.has(metric.key)) renderChart(metric, seq, width);
  }
}

function init() {
  document.getElementById("theme-toggle").addEventListener("click", () => {
    switchTheme(currentTheme() === "light" ? "dark" : "light");
  });
  document.getElementById("theme-toggle").textContent = currentTheme() === "light" ? "Dark" : "Light";

  document.getElementById("select-all").addEventListener("click", () => {
    for (const w of state.worlds.values()) {
      if (w.meta.series_length > 0 && w.meta.file) state.selected.add(w.meta.id);
    }
    renderWorldList();
    updateSummary();
    rebuildCharts();
  });
  document.getElementById("select-none").addEventListener("click", () => {
    state.selected.clear();
    renderWorldList();
    updateSummary();
    rebuildCharts();
  });

  renderMetricBar();

  const densIn = document.getElementById("x-ticks");
  const densOut = document.getElementById("x-ticks-out");
  densIn.addEventListener("input", () => {
    state.xTicks = +densIn.value;
    densOut.textContent = densIn.value;
    rebuildCharts();
  });
  document.getElementById("y-zero").addEventListener("change", (e) => {
    state.yFromZero = e.target.checked;
    rebuildCharts();
  });
  document.getElementById("points").addEventListener("change", (e) => {
    state.points = e.target.checked;
    rebuildCharts();
  });
  const searchIn = document.getElementById("world-search");
  searchIn.addEventListener("input", () => {
    state.search = searchIn.value;
    renderWorldList();
  });
  const rmin = document.getElementById("range-min");
  const rmax = document.getElementById("range-max");
  const loIn = document.getElementById("range-lo-in");
  const hiIn = document.getElementById("range-hi-in");
  const setRangeInputs = () => {
    rmin.value = state.rangeLo;
    rmax.value = state.rangeHi;
    loIn.value = state.rangeLo;
    hiIn.value = state.rangeHi;
  };
  const clampRange = () => {
    if (state.rangeHi <= 0) state.rangeHi = Math.max(1, +rmax.max);
    if (state.rangeLo >= state.rangeHi) state.rangeLo = Math.max(0, state.rangeHi - 1);
  };
  rmin.addEventListener("input", () => {
    state.rangeLo = Math.min(+rmin.value, state.rangeHi);
    clampRange();
    setRangeInputs();
    rebuildCharts();
  });
  rmax.addEventListener("input", () => {
    state.rangeHi = Math.max(+rmax.value, state.rangeLo);
    clampRange();
    setRangeInputs();
    rebuildCharts();
  });
  loIn.addEventListener("input", () => {
    loIn.value = loIn.value.replace(/[^0-9]/g, "");
    state.rangeLo = Math.min(+loIn.value || 0, state.rangeHi);
    clampRange();
    setRangeInputs();
    rebuildCharts();
  });
  hiIn.addEventListener("input", () => {
    hiIn.value = hiIn.value.replace(/[^0-9]/g, "");
    state.rangeHi = Math.max(+hiIn.value || 0, state.rangeLo);
    clampRange();
    setRangeInputs();
    rebuildCharts();
  });

  fetch(`${DATA_ROOT}/index.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`index.json not found at ${DATA_ROOT}/ (extract first)`);
      return r.json();
    })
    .then((data) => {
      state.index = data;
      for (const m of data.worlds) state.worlds.set(m.id, { meta: m, data: null });
      renderWorldList();
      updateSummary();
      rebuildCharts();
    })
    .catch((err) => {
      document.getElementById("empty-hint").textContent = err.message;
      document.getElementById("summary").textContent = "no data";
    });

  window.addEventListener("resize", () => {
    if (state.charts.length === 0) return;
    clearTimeout(window.__resizeT);
    window.__resizeT = setTimeout(rebuildCharts, 150);
  });
}

init();