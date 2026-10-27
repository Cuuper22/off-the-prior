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

  function line(host, spec) {
    host.textContent = "";
    var W = 640, H = spec.height || 300, m = { l: 56, r: 16, t: 14, b: 44 };
    var all = [];
    spec.series.forEach(function (s) {
      (s.points || []).forEach(function (p) { if (p[1] !== null && isFinite(p[1])) all.push(p); });
      (s.band || []).forEach(function (b) { if (isFinite(b[1])) all.push([b[0], b[1]]); if (isFinite(b[2])) all.push([b[0], b[2]]); });
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
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "otp-chart", role: "img", "aria-label": spec.label || spec.yLabel || "chart" });
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
    if (spec.xLabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "otp-label" }, g).textContent = spec.xLabel;
    if (spec.yLabel) el("text", { x: 14, y: (m.t + H - m.b) / 2, transform: "rotate(-90 14 " + (m.t + H - m.b) / 2 + ")", "text-anchor": "middle", class: "otp-label" }, g).textContent = spec.yLabel;
    if (spec.ref1 && 1 >= y0 && 1 <= y1) el("line", { x1: m.l, x2: W - m.r, y1: Y(1), y2: Y(1), class: "otp-ref" }, g);
    var pts = [];
    spec.series.forEach(function (s) {
      if (s.band && s.band.length) {
        var up = s.band.filter(function (b) { return isFinite(b[1]) && isFinite(b[2]); });
        if (up.length) {
          var d = up.map(function (b, i) { return (i ? "L" : "M") + X(b[0]) + " " + Y(b[2]); }).join(" ") + " " +
            up.slice().reverse().map(function (b) { return "L" + X(b[0]) + " " + Y(b[1]); }).join(" ") + " Z";
          el("path", { d: d, fill: s.color, "fill-opacity": s.bandOpacity || 0.15, stroke: "none" }, g);
        }
      }
      var ok = (s.points || []).filter(function (p) { return p[1] !== null && isFinite(p[1]); });
      if (!ok.length) return;
      el("path", { d: ok.map(function (p, i) { return (i ? "L" : "M") + X(p[0]) + " " + Y(p[1]); }).join(" "), fill: "none", stroke: s.color,
        "stroke-width": s.width || 2, "stroke-dasharray": s.dash || null, "stroke-linejoin": "round" }, g);
      if (s.marker !== false && ok.length < 40) ok.forEach(function (p) {
        el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: s.color, stroke: "#f7f6f2", "stroke-width": 2 }, g);
      });
      ok.forEach(function (p) { pts.push({ x: X(p[0]), y: Y(p[1]), s: s, p: p }); });
    });
    // tooltip
    var tipG = el("g", { class: "otp-tip", visibility: "hidden" }, svg);
    var tipDot = el("circle", { r: 6, fill: "none", "stroke-width": 2 }, tipG);
    var tipBox = el("rect", { rx: 0, class: "otp-tipbox" }, tipG);
    var tipText = el("text", { class: "otp-tiptext" }, tipG);
    var hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    function move(ev) {
      var r = svg.getBoundingClientRect();
      var px = (ev.clientX - r.left) / r.width * W, py = (ev.clientY - r.top) / r.height * H;
      var best = null, bd = 1e9;
      pts.forEach(function (q) { var d = Math.hypot(q.x - px, (q.y - py) * 0.6); if (d < bd) { bd = d; best = q; } });
      if (!best) return;
      tipG.setAttribute("visibility", "visible");
      tipDot.setAttribute("cx", best.x); tipDot.setAttribute("cy", best.y); tipDot.setAttribute("stroke", best.s.color);
      var label = (best.s.name ? best.s.name + ": " : "") + (spec.yFmt || fmtNum)(best.p[1]) + "  at " + (best.p.label || (spec.xFmt || fmtNum)(best.p[0]));
      tipText.textContent = label;
      var tw = Math.max(60, label.length * 7.2 + 14);
      var tx = Math.min(best.x + 10, W - tw - 4), ty = Math.max(best.y - 30, 4);
      tipBox.setAttribute("x", tx); tipBox.setAttribute("y", ty); tipBox.setAttribute("width", tw); tipBox.setAttribute("height", 22);
      tipText.setAttribute("x", tx + 7); tipText.setAttribute("y", ty + 15);
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", function () { tipG.setAttribute("visibility", "hidden"); });
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

  root.OTPChart = { line: line, table: table, pending: pendingBox, segmented: segmented, fmt: fmtNum, h: h,
                    SIZE_COLORS: SIZE_COLORS, SIZES: SIZES };
})(window);
