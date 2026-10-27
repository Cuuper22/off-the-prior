/* Shared helpers for the Off the Prior figures: data loading and one SVG line chart.
   OTPData.get(name)  -> Promise of docs/data/<name>.json (cached)
   OTPChart.line(host, spec) draws a chart into host and returns { svg }.
     spec = { series: [{ name, color, points: [[x, y], ...], dash, band: [[x, lo, hi], ...], marker }],
              xlog, ylog, xLabel, yLabel, ref1 (dashed line at y = 1), xTicks: [[value, label], ...],
              yFmt, xFmt, height }
   Every chart has a hover/tap tooltip (nearest point of any series), a legend when there is more than one
   series, and a text table hidden for screen readers. No libraries. */
(function (root) {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var cache = {};
  var base = "";
  if (document.currentScript && document.currentScript.src) base = document.currentScript.src.replace(/figures\/[^/]*$/, "");

  root.OTPData = {
    base: base,
    get: function (name) {
      if (!cache[name]) {
        cache[name] = fetch(base + "data/" + name + ".json").then(function (r) {
          if (!r.ok) throw new Error(name + ": " + r.status);
          return r.json();
        });
      }
      return cache[name];
    },
  };

  // Ordered ramp for model size (light = small, dark = large), validated as an ordinal ramp on the light surface.
  var SIZE_COLORS = { "5M": "#86b6ef", "28M": "#3987e5", "123M": "#1c5cab", "209M": "#0d366b" };
  var SIZES = ["5M", "28M", "123M", "209M"];

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function h(tag, attrs, kids) {
    var n = document.createElement(tag);
    for (var k in attrs || {}) {
      if (k === "text") n.textContent = attrs[k];
      else if (k === "class") n.className = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return n;
  }
  function tickFmt(v) {
    if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
    return String(+v.toPrecision(2));
  }
  function fmtNum(v) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    var a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
  function niceLogTicks(lo, hi) {
    var out = [];
    var e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi));
    for (var e = e0; e <= e1; e++) [1, 2, 5].forEach(function (m) {
      var v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    });
    if (out.length > 7) out = out.filter(function (v) { var m = v / Math.pow(10, Math.floor(Math.log10(v))); return Math.abs(m - 1) < 1e-9 || Math.abs(m - 5) < 1e-9 || out.length < 10; });
    return out;
  }
  function niceLinTicks(lo, hi, n) {
    var span = hi - lo || 1, step = Math.pow(10, Math.floor(Math.log10(span / (n || 5))));
    var err = span / (n || 5) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // Drawing width in pixels: the host's own width, so text keeps its real size on phones (640 while the host is hidden).
  function widthOf(host) {
    var w = host.clientWidth;
    return w >= 200 ? Math.min(w, 860) : 640;
  }
  // Redraw a figure when its host changes width (window resize, window opened, phone turned).
  function watchWidth(host, redraw, w) {
    host.__otpRedraw = redraw;
    host.__otpW = w;
    if (host.__otpRO || !root.ResizeObserver) return;
    host.__otpRO = new root.ResizeObserver(function () {
      if (Math.abs(widthOf(host) - host.__otpW) > 24 && host.__otpRedraw) host.__otpRedraw();
    });
    host.__otpRO.observe(host);
  }
  // Keyboard and screen-reader reading of a chart: arrow keys step through points; the text goes to a live region.
  function live(host) {
    var n = host.querySelector(".otp-live");
    if (!n) { n = h("div", { class: "gsf-sr otp-live", "aria-live": "polite" }); host.appendChild(n); }
    return n;
  }

  // Cut an axis title that would not fit in `px` pixels (about 6.6 px per character at 13 px).
  function fit(text, px) {
    var max = Math.max(8, Math.floor(px / 6.6));
    return text.length > max ? text.slice(0, max - 1) + "…" : text;
  }

  function line(host, spec) {
    var W = widthOf(host), H = spec.height || (W < 480 ? 260 : 300), m = { l: W < 480 ? 54 : 58, r: 22, t: 14, b: 44 };
    host.textContent = "";
    watchWidth(host, function () { line(host, spec); }, W);
    var all = [];
    function okY(v) { return v !== null && isFinite(v) && (!spec.ylog || v > 0); }
    spec.series.forEach(function (s) {
      s.points = (s.points || []).filter(function (p) { return okY(p[1]); });
      if (s.band) s.band = s.band.filter(function (b) { return okY(b[1]) && okY(b[2]); });
      (s.points || []).forEach(function (p) { all.push(p); });
      (s.band || []).forEach(function (b) { all.push([b[0], b[1]]); all.push([b[0], b[2]]); });
    });
    if (!all.length) { host.appendChild(h("p", { class: "otp-empty", text: "No data for this choice." })); return {}; }
    var xs = all.map(function (p) { return p[0]; }), ys = all.map(function (p) { return p[1]; });
    if (spec.ref1) ys.push(1);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    if (spec.ylog) { y0 = Math.max(y0, 1e-6) / 1.15; y1 *= 1.15; } else { var pad = (y1 - y0) * 0.08 || 1; y0 -= pad; y1 += pad; }
    if (spec.yMin !== undefined) y0 = spec.yMin;
    if (spec.yMax !== undefined) y1 = spec.yMax;
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    var fx = spec.xlog ? function (v) { return Math.log10(v); } : function (v) { return v; };
    var fy = spec.ylog ? function (v) { return Math.log10(v); } : function (v) { return v; };
    var X = function (v) { return m.l + (fx(v) - fx(x0)) / (fx(x1) - fx(x0)) * (W - m.l - m.r); };
    var Y = function (v) { return H - m.b - (fy(v) - fy(y0)) / (fy(y1) - fy(y0)) * (H - m.t - m.b); };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "otp-chart", role: "group", "aria-label": (spec.label || spec.yLabel || "chart") + ". Arrow keys read the points one by one." });
    var g = el("g", {}, svg);
    var yt = spec.ylog ? niceLogTicks(y0, y1) : niceLinTicks(y0, y1, 5);
    yt.forEach(function (v) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "otp-grid" }, g);
      el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "otp-tick" }, g).textContent = (spec.yFmt || tickFmt)(v);
    });
    var xt = spec.xTicks || (spec.xlog ? niceLogTicks(x0, x1).map(function (v) { return [v, tickFmt(v)]; }) : niceLinTicks(x0, x1, 6).map(function (v) { return [v, (spec.xFmt || tickFmt)(v)]; }));
    xt.forEach(function (t) {
      el("text", { x: X(t[0]), y: H - m.b + 18, "text-anchor": "middle", class: "otp-tick" }, g).textContent = t[1];
    });
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "otp-axis" }, g);
    if (spec.xLabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "otp-label" }, g).textContent = fit(spec.xLabel, W - m.l - m.r);
    if (spec.yLabel) el("text", { x: 12, y: (m.t + H - m.b) / 2, transform: "rotate(-90 12 " + (m.t + H - m.b) / 2 + ")", "text-anchor": "middle", class: "otp-label" }, g).textContent = fit(spec.yLabel, H - m.t - m.b);
    if (spec.ref1 && 1 >= y0 && 1 <= y1) el("line", { x1: m.l, x2: W - m.r, y1: Y(1), y2: Y(1), class: "otp-ref" }, g);
    var pts = [], series_els = [];
    spec.series.forEach(function (s) {
      if (s.band && s.band.length) {
        var up = s.band.filter(function (b) { return isFinite(b[1]) && isFinite(b[2]); });
        if (up.length) {
          var d = up.map(function (b, i) { return (i ? "L" : "M") + X(b[0]) + " " + Y(b[2]); }).join(" ") + " " +
            up.slice().reverse().map(function (b) { return "L" + X(b[0]) + " " + Y(b[1]); }).join(" ") + " Z";
          s.bandEl = el("path", { d: d, fill: s.color, "fill-opacity": s.bandOpacity || 0.15, stroke: "none" }, g);
        }
      }
      var ok = (s.points || []).filter(function (p) { return p[1] !== null && isFinite(p[1]); });
      if (!ok.length) return;
      var pathEl = null;
      if (!s.scatter) pathEl = el("path", { d: ok.map(function (p, i) { return (i ? "L" : "M") + X(p[0]) + " " + Y(p[1]); }).join(" "), fill: "none", stroke: s.color,
        "stroke-width": s.width || 2, "stroke-dasharray": s.dash || null, "stroke-linejoin": "round" }, g);
      if (s.scatter) ok.forEach(function (p) {
        var r = s.radius || 5, cx = X(p[0]), cy = Y(p[1]);
        if (s.hollow) el("circle", { cx: cx, cy: cy, r: r, fill: "#f7f6f2", stroke: s.color, "stroke-width": 2 }, g);
        else el("circle", { cx: cx, cy: cy, r: r, fill: s.color, "fill-opacity": 0.85, stroke: "#f7f6f2", "stroke-width": 1.5 }, g);
      });
      else if (s.marker !== false && ok.length < 40) ok.forEach(function (p) {
        el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color, stroke: "#f7f6f2", "stroke-width": 2 }, g);
      });
      series_els.push({ s: s, path: pathEl });
      ok.forEach(function (p) { pts.push({ x: X(p[0]), y: Y(p[1]), s: s, p: p }); });
    });
    // tooltip: pointer, touch and keyboard (arrow keys step through the points, left to right)
    var tipG = el("g", { class: "otp-tip", visibility: "hidden" }, svg);
    var tipDot = el("circle", { r: 6, fill: "none", "stroke-width": 2 }, tipG);
    var tipBox = el("rect", { rx: 0, class: "otp-tipbox" }, tipG);
    var tipText = el("text", { class: "otp-tiptext" }, tipG);
    var hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    var liveEl = null;
    function labelOf(q) {
      return (q.s.name ? q.s.name + ": " : "") + (spec.yFmt || fmtNum)(q.p[1]) + "  at " + (q.p.label || (spec.xFmt || fmtNum)(q.p[0]));
    }
    function showTip(best, speak) {
      tipG.setAttribute("visibility", "visible");
      tipDot.setAttribute("cx", best.x); tipDot.setAttribute("cy", best.y); tipDot.setAttribute("stroke", best.s.color);
      var label = labelOf(best);
      tipText.textContent = label;
      var tw = Math.max(60, label.length * 7.2 + 14);
      var tx = Math.max(4, Math.min(best.x + 10, W - tw - 4)), ty = Math.max(best.y - 30, 4);
      tipBox.setAttribute("x", tx); tipBox.setAttribute("y", ty); tipBox.setAttribute("width", tw); tipBox.setAttribute("height", 22);
      tipText.setAttribute("x", tx + 7); tipText.setAttribute("y", ty + 15);
      if (speak) { liveEl = liveEl || live(host); liveEl.textContent = label; }
    }
    function move(ev) {
      var r = svg.getBoundingClientRect();
      var px = (ev.clientX - r.left) / r.width * W, py = (ev.clientY - r.top) / r.height * H;
      var best = null, bd = 1e9;
      pts.forEach(function (q) { var d = Math.hypot(q.x - px, (q.y - py) * 0.6); if (d < bd) { bd = d; best = q; } });
      if (best) showTip(best, false);
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", function () { tipG.setAttribute("visibility", "hidden"); });
    var order = pts.slice().sort(function (a, b) { return a.x - b.x || a.y - b.y; }), at = -1;
    svg.setAttribute("tabindex", "0");
    svg.addEventListener("keydown", function (ev) {
      if (!order.length) return;
      var k = ev.key;
      if (k === "ArrowRight" || k === "ArrowDown") at = Math.min(order.length - 1, at + 1);
      else if (k === "ArrowLeft" || k === "ArrowUp") at = Math.max(0, at < 0 ? 0 : at - 1);
      else if (k === "Home") at = 0;
      else if (k === "End") at = order.length - 1;
      else if (k === "Escape") { tipG.setAttribute("visibility", "hidden"); return; }
      else return;
      ev.preventDefault();
      showTip(order[at], true);
    });
    svg.addEventListener("blur", function () { tipG.setAttribute("visibility", "hidden"); at = -1; });
    host.appendChild(svg);
    var named = spec.series.filter(function (s) { return s.name && !s.noLegend; });
    if (named.length > 1) {
      var lg = h("div", { class: "otp-legend" });
      named.forEach(function (s) {
        var sw = h("span", { class: "otp-swatch" + (s.dash ? " is-dashed" : "") });
        sw.style.setProperty("--c", s.color);
        lg.appendChild(h("span", { class: "otp-legend-item" }, [sw, s.name]));
      });
      host.appendChild(lg);
    }
    var tbl = h("table", { class: "gsf-sr" });
    spec.series.forEach(function (s) {
      (s.points || []).forEach(function (p) {
        tbl.appendChild(h("tr", {}, [h("td", { text: s.name || "" }), h("td", { text: String(p[0]) }), h("td", { text: fmtNum(p[1]) })]));
      });
    });
    host.appendChild(tbl);
    var result = { svg: svg, X: X, Y: Y, series: series_els, W: W, H: H };
    if (spec.onDraw) spec.onDraw(result);
    return result;
  }


  /* dots: one row per item, a dot at its value with an optional interval (a "forest" plot). Rows can be grouped.
       spec = { rows: [{ label, group, v, lo, hi, color, tip }], ref (dashed vertical line), xlog, xLabel, xMin, xMax, label }
     Rows are keyboard stops (Tab or arrow keys); the row's text appears next to the dot and in a live region. */
  function dots(host, spec) {
    var W = widthOf(host), narrow = W < 480, LW = narrow ? 104 : 180, RH = 22, GH = 20, top = 8, bottom = 44;
    host.textContent = "";
    watchWidth(host, function () { dots(host, spec); }, W);
    var rows = spec.rows.filter(function (r) { return r.v !== null && isFinite(r.v); });
    if (!rows.length) { host.appendChild(h("p", { class: "otp-empty", text: "No data for this choice." })); return {}; }
    var vals = [];
    rows.forEach(function (r) { vals.push(r.v); if (r.lo !== undefined && r.lo !== null) vals.push(r.lo); if (r.hi !== undefined && r.hi !== null) vals.push(r.hi); });
    if (spec.ref !== undefined) vals.push(spec.ref);
    var v0 = spec.xMin !== undefined ? spec.xMin : Math.min.apply(null, vals), v1 = spec.xMax !== undefined ? spec.xMax : Math.max.apply(null, vals);
    if (v0 === v1) { v0 -= 1; v1 += 1; }
    if (spec.xMin === undefined) { var pd = (v1 - v0) * 0.06; v0 -= pd; v1 += pd; }
    if (spec.xMax === undefined && spec.xMin !== undefined) v1 += (v1 - v0) * 0.04;
    var f = spec.xlog ? Math.log10 : function (v) { return v; };
    var X = function (v) { return LW + (f(v) - f(v0)) / (f(v1) - f(v0)) * (W - LW - 14); };
    var y = top, layout = [], lastGroup = null;
    rows.forEach(function (r) {
      if (r.group !== undefined && r.group !== lastGroup) { layout.push({ head: r.group, y: y }); y += GH; lastGroup = r.group; }
      layout.push({ r: r, y: y }); y += RH;
    });
    var H = y + bottom;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "otp-chart", role: "group", "aria-label": (spec.label || "dot plot") + ". Tab or arrow keys read the rows one by one." });
    var g = el("g", {}, svg);
    var ticks = spec.xlog ? niceLogTicks(v0, v1) : niceLinTicks(v0, v1, narrow ? 4 : 6);
    ticks.forEach(function (t) {
      el("line", { x1: X(t), x2: X(t), y1: top, y2: H - bottom, class: "otp-grid" }, g);
      el("text", { x: X(t), y: H - bottom + 16, "text-anchor": "middle", class: "otp-tick" }, g).textContent = (spec.xFmt || tickFmt)(t);
    });
    el("line", { x1: LW, x2: W - 14, y1: H - bottom, y2: H - bottom, class: "otp-axis" }, g);
    if (spec.xLabel) el("text", { x: (LW + W - 14) / 2, y: H - 6, "text-anchor": "middle", class: "otp-label" }, g).textContent = fit(spec.xLabel, W - LW - 14);
    if (spec.ref !== undefined && spec.ref >= v0 && spec.ref <= v1) el("line", { x1: X(spec.ref), x2: X(spec.ref), y1: top, y2: H - bottom, class: "otp-ref" }, g);
    var liveEl = live(host), tipText = el("text", { class: "otp-tiptext" }, svg), tipBox = el("rect", { class: "otp-tipbox", visibility: "hidden" }, svg);
    svg.appendChild(tipText);
    var stops = [];
    function tipFor(item, speak) {
      var r = item.r, text = r.tip || (r.label + ": " + (spec.vFmt || fmtNum)(r.v));
      tipText.textContent = text;
      var tw = Math.max(60, text.length * 7.2 + 14), cx = X(r.v);
      var tx = Math.max(4, Math.min(cx + 10, W - tw - 4)), ty = item.y - 2;
      if (ty < 2) ty = 2;
      tipBox.setAttribute("visibility", "visible"); tipBox.setAttribute("x", tx); tipBox.setAttribute("y", ty); tipBox.setAttribute("width", tw); tipBox.setAttribute("height", 20);
      tipText.setAttribute("x", tx + 7); tipText.setAttribute("y", ty + 14); tipText.setAttribute("visibility", "visible");
      svg.appendChild(tipBox); svg.appendChild(tipText);
      if (speak) liveEl.textContent = text;
    }
    function hide() { tipBox.setAttribute("visibility", "hidden"); tipText.setAttribute("visibility", "hidden"); }
    layout.forEach(function (it) {
      if (it.head !== undefined) {
        el("text", { x: 4, y: it.y + 14, class: "otp-label", "font-weight": 600 }, g).textContent = it.head;
        return;
      }
      var r = it.r, cy = it.y + RH / 2;
      var row = el("g", { tabindex: "0", role: "img", "aria-label": r.tip || (r.label + ": " + (spec.vFmt || fmtNum)(r.v)), class: "otp-dotrow" }, g);
      el("rect", { x: 0, y: it.y, width: W, height: RH, fill: "transparent" }, row);
      var lab = r.label.length > (narrow ? 15 : 27) ? r.label.slice(0, narrow ? 14 : 26) + "…" : r.label;
      el("text", { x: LW - 8, y: cy + 4, "text-anchor": "end", class: "otp-tick" }, row).textContent = lab;
      if (r.lo !== undefined && r.lo !== null && r.hi !== undefined && r.hi !== null)
        el("line", { x1: X(r.lo), x2: X(r.hi), y1: cy, y2: cy, stroke: r.color || "#1c5cab", "stroke-width": 2 }, row);
      el("circle", { cx: X(r.v), cy: cy, r: 5, fill: r.hollow ? "#f7f6f2" : (r.color || "#1c5cab"), stroke: r.color || "#1c5cab", "stroke-width": 2 }, row);
      row.addEventListener("pointerenter", function () { tipFor(it, false); });
      row.addEventListener("pointerleave", hide);
      row.addEventListener("focus", function () { tipFor(it, true); });
      row.addEventListener("blur", hide);
      row.addEventListener("keydown", function (ev) {
        var i = stops.indexOf(row), j = ev.key === "ArrowDown" ? i + 1 : ev.key === "ArrowUp" ? i - 1 : -1;
        if (j >= 0 && j < stops.length) { ev.preventDefault(); stops[j].focus(); }
      });
      stops.push(row);
    });
    host.insertBefore(svg, liveEl);
    var named = [], seen = {};
    rows.forEach(function (r) { if (r.legend && !seen[r.legend]) { seen[r.legend] = 1; named.push([r.legend, r.color]); } });
    if (named.length > 1) {
      var lg = h("div", { class: "otp-legend" });
      named.forEach(function (n) { var sw = h("span", { class: "otp-swatch" }); sw.style.setProperty("--c", n[1]); lg.appendChild(h("span", { class: "otp-legend-item" }, [sw, n[0]])); });
      host.insertBefore(lg, liveEl);
    }
    return { svg: svg };
  }

  /* heat: a small grid of cells coloured by steps. spec = { rows: [label], cols: [label], cell(r, c) -> { v, text, step } or null,
     colors: [step colours], legend: [labels per step], rowTitle, colTitle, label }. Arrow keys move over the cells;
     the text of the active cell is printed under the grid. */
  function heat(host, spec) {
    var W = widthOf(host), narrow = W < 480, LW = narrow ? 70 : 90, TH = 22, nC = spec.cols.length, nR = spec.rows.length;
    host.textContent = "";
    watchWidth(host, function () { heat(host, spec); }, W);
    var cw = (W - LW - 8) / nC, ch = 34, H = TH + nR * ch + 22;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "otp-chart", role: "group", "aria-label": (spec.label || "grid") + ". Arrow keys move over the cells." });
    spec.cols.forEach(function (c, j) { el("text", { x: LW + j * cw + cw / 2, y: 14, "text-anchor": "middle", class: "otp-tick" }, svg).textContent = c; });
    var cells = [], info = h("p", { class: "otp-note otp-heatinfo", "aria-live": "polite", text: spec.hint || "" });
    spec.rows.forEach(function (rl, i) {
      el("text", { x: LW - 6, y: TH + i * ch + ch / 2 + 4, "text-anchor": "end", class: "otp-tick" }, svg).textContent = rl;
      spec.cols.forEach(function (cl, j) {
        var d = spec.cell(i, j);
        var rect = el("rect", { x: LW + j * cw + 1, y: TH + i * ch + 1, width: cw - 2, height: ch - 2,
          fill: d && d.step !== undefined ? spec.colors[d.step] : "none", stroke: d ? "#1c1d21" : "#8d9096", "stroke-width": 1, "stroke-dasharray": d ? null : "2 3", "data-cell": i + "," + j }, svg);
        cells.push({ rect: rect, i: i, j: j, text: d ? d.text : rl + ", " + cl + ": empty" });
      });
    });
    el("text", { x: LW + (W - LW) / 2, y: H - 6, "text-anchor": "middle", class: "otp-label" }, svg).textContent = spec.colTitle || "";
    var cur = 0;
    function select(k, speak) {
      cells.forEach(function (c) { c.rect.setAttribute("stroke-width", 1); });
      cur = k; var c = cells[k]; c.rect.setAttribute("stroke-width", 3);
      info.textContent = c.text;
    }
    svg.setAttribute("tabindex", "0");
    svg.addEventListener("keydown", function (ev) {
      var i = cells[cur].i, j = cells[cur].j;
      if (ev.key === "ArrowRight") j = Math.min(nC - 1, j + 1); else if (ev.key === "ArrowLeft") j = Math.max(0, j - 1);
      else if (ev.key === "ArrowDown") i = Math.min(nR - 1, i + 1); else if (ev.key === "ArrowUp") i = Math.max(0, i - 1); else return;
      ev.preventDefault(); select(i * nC + j, true);
    });
    svg.addEventListener("focus", function () { select(cur, true); });
    cells.forEach(function (c, k) {
      c.rect.addEventListener("pointerenter", function () { select(k, false); });
      c.rect.addEventListener("pointerdown", function () { select(k, false); });
    });
    host.appendChild(svg);
    if (spec.legend) {
      var lg = h("div", { class: "otp-legend" });
      spec.legend.forEach(function (t, k) { var sw = h("span", { class: "otp-swatch otp-swatch-box" }); sw.style.setProperty("--c", spec.colors[k]); lg.appendChild(h("span", { class: "otp-legend-item" }, [sw, t])); });
      host.appendChild(lg);
    }
    host.appendChild(info);
    return { svg: svg };
  }

  function pendingBox(host, d) {
    host.textContent = "";
    var box = h("div", { class: "otp-pending" }, [
      h("strong", { text: "Results not in yet." }),
      " This panel fills itself from ",
      h("code", { text: d && d.expect ? d.expect : "a result file" }),
      " when that file lands in the repository. Nothing here is a guess.",
    ]);
    host.appendChild(box);
  }

  function table(host, columns, rows, opts) {
    var t = h("table", { class: "otp-table" });
    var thead = h("thead", {}, [h("tr", {}, columns.map(function (c) { return h("th", { text: c }); }))]);
    t.appendChild(thead);
    var tb = h("tbody");
    rows.forEach(function (r) {
      tb.appendChild(h("tr", {}, r.map(function (v) { return h("td", { text: typeof v === "number" ? fmtNum(v) : (v === null ? "–" : String(v)) }); })));
    });
    t.appendChild(tb);
    var wrap = h("div", { class: "otp-table-wrap" }, [t]);
    host.appendChild(wrap);
    return t;
  }

  function segmented(host, options, value, onChange, label) {
    var wrap = h("div", { class: "otp-seg", role: "group", "aria-label": label || "choose" });
    options.forEach(function (o) {
      var b = h("button", { type: "button", class: "otp-seg-btn", "aria-pressed": String(o[0] === value), text: o[1] });
      b.addEventListener("click", function () {
        wrap.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onChange(o[0]);
      });
      wrap.appendChild(b);
    });
    host.appendChild(wrap);
    return wrap;
  }

  root.OTPChart = { line: line, dots: dots, heat: heat, width: widthOf, table: table, pending: pendingBox, segmented: segmented, fmt: fmtNum, h: h,
                    SIZE_COLORS: SIZE_COLORS, SIZES: SIZES };
})(window);
