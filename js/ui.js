// 共用的格式化、表格、卡片、下載與 Plotly 繪圖工具

export const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e9) return v.toExponential(3);
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a >= 1) return v.toLocaleString("en-US", { maximumFractionDigits: 3 });
  if (a >= 1e-3) return Number(v.toPrecision(4)).toString();
  return v.toExponential(2);
}

export function fmtP(p) {
  if (!Number.isFinite(p)) return "—";
  if (p < 1e-16) return "< 1e-16";
  if (p < 1e-4) return p.toExponential(2);
  return p.toFixed(4);
}

// 「p = 0.0123」或「p < 1e-16」
export const pEq = (p) => (fmtP(p).startsWith("<") ? `p ${fmtP(p)}` : `p = ${fmtP(p)}`);

export const pct =(v, d = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : "—");

export function table(head, rows, { num = [], cls = "", rowCls } = {}) {
  const numSet = new Set(num);
  const th = head.map((h, i) => `<th${numSet.has(i) ? ' class="num"' : ""}>${h}</th>`).join("");
  const body = rows
    .map((r, ri) => {
      const c = rowCls ? rowCls(r, ri) : "";
      return `<tr${c ? ` class="${c}"` : ""}>${r.map((v, i) => `<td${numSet.has(i) ? ' class="num"' : ""}>${v}</td>`).join("")}</tr>`;
    })
    .join("");
  return `<div class="tbl-wrap ${cls}"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}

export function card(title, body, { sub = "", cls = "" } = {}) {
  return `<section class="card ${cls}"><header class="card-h"><h3>${title}</h3>${sub ? `<p>${sub}</p>` : ""}</header>${body}</section>`;
}

export const chart = (id, cls = "") => `<div class="chart ${cls}" id="${id}"></div>`;

export function kpiRow(items) {
  return items
    .map(
      (k) => `<div class="kpi"><span class="kpi-label">${k.label}</span><span class="kpi-value">${k.value}</span>${
        k.sub ? `<span class="kpi-sub">${k.sub}</span>` : ""
      }</div>`,
    )
    .join("");
}

export const badge = (ok, yes = "通過", no = "注意") =>
  ok ? `<span class="badge ok">✓ ${yes}</span>` : `<span class="badge warn">! ${no}</span>`;

export function toCSV(head, rows) {
  const q = (v) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [head, ...rows].map((r) => r.map(q).join(",")).join("\n") + "\n";
}

export function download(filename, text, mime = "text/csv;charset=utf-8") {
  const blob = new Blob([mime.startsWith("text/csv") ? "﻿" + text : text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 0);
}

// ---------- 圖表 ----------
export function theme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    text: v("--text"), text2: v("--text-2"), muted: v("--muted"), grid: v("--grid"), axis: v("--axis"),
    surface: v("--surface"), mid: v("--mid"), s1: v("--s1"), s2: v("--s2"), s3: v("--s3"), neg: v("--neg"),
  };
}

export function rgba(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? [...h].map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export function band(x, lo, hi, color, alpha, name, extra = {}) {
  const xs = Array.from(x);
  return {
    type: "scatter",
    x: xs.concat(xs.slice().reverse()),
    y: Array.from(hi).concat(Array.from(lo).reverse()),
    fill: "toself",
    fillcolor: rgba(color, alpha),
    line: { width: 0 },
    hoverinfo: "skip",
    name,
    ...extra,
  };
}

export function plot(id, traces, layout = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  const t = theme();
  const axis = { gridcolor: t.grid, linecolor: t.axis, zeroline: false, automargin: true, tickfont: { color: t.muted }, title: { font: { color: t.text2 } } };
  const base = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: getComputedStyle(document.body).fontFamily, size: 12, color: t.text2 },
    margin: { l: 56, r: 16, t: 36, b: 48 },
    height: 360,
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", bgcolor: "rgba(0,0,0,0)", font: { color: t.text2 } },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.axis, font: { color: t.text } },
  };
  const merged = { ...base, ...layout, legend: { ...base.legend, ...layout.legend }, margin: { ...base.margin, ...layout.margin } };
  for (const ax of Object.keys(layout).filter((k) => /^[xy]axis\d*$/.test(k)).concat(["xaxis", "yaxis"])) {
    const user = layout[ax] || {};
    merged[ax] = { ...axis, showgrid: !ax.startsWith("x"), ...user, title: { ...axis.title, ...(typeof user.title === "string" ? { text: user.title } : user.title) } };
  }
  window.Plotly.react(el, traces, merged, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
  });
}
