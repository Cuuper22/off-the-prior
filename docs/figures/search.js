/* search: three ways to search for failures at the same budget. Data: data/search.json (results/search/*).
   Views: best fitness so far against Timestral calls, the MAP-Elites grid per method, transfer to larger sizes. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var COLORS = { cmaes: "#0e7d84", map_elites: "#b4531c", gradient: "#6a4c93", gradient_r12: "#9b8bb8" };
  var STEPS = ["#eef0f3", "#bfe0e0", "#6bb8bb", "#0e7d84", "#08474c"];
  F.define("search", {
    title: "Letting a computer hunt for failures",
    label: "Charts comparing CMA-ES, MAP-Elites and gradient search: best failure found against Timestral calls, grid coverage, transfer to larger sizes",
    summary: "Each method gets the same number of Timestral calls. Fitness is Timestral's error divided by the referee's on the series the method built.",
    build: root.OTPSlot("search", function (ctx, d) {
      var bar = C.h("div", { class: "otp-controls" }), bar2 = C.h("div", { class: "otp-controls" }), plot = C.h("div"), note = C.h("p", { class: "otp-note" });
      ctx.el.appendChild(bar); ctx.el.appendChild(bar2); ctx.el.appendChild(plot); ctx.el.appendChild(note);
      var names = {}; d.methods.forEach(function (m) { names[m[0]] = m[1]; });
      var views = [["curve", "Best so far"]];
      if (d.grid) views.push(["grid", "Failure types found"]);
      if (d.transfer) views.push(["transfer", "Carry over to bigger sizes"]);
      var st = { view: "curve", method: d.methods[0][0], thr: "1.5" };
      function show() {
        bar2.textContent = "";
        if (st.view === "curve") {
          var series = [];
          d.methods.forEach(function (m) {
            var c = d.curves[m[0]];
            if (!c) return;
            series.push({ name: m[1], color: COLORS[m[0]], marker: false, points: c.calls.map(function (x, i) { return [x, c.median[i]]; }),
                          band: c.calls.map(function (x, i) { return [x, c.lo[i], c.hi[i]]; }), bandOpacity: 0.12 });
          });
          C.line(plot, { series: series, xlog: true, ylog: true, ref1: true, xLabel: "Timestral calls spent", yLabel: "best error ÷ referee's",
                         label: "Best failure found so far against Timestral calls, per search method" });
          note.textContent = "Line: median over search seeds; band: lowest to highest seed. Above 1 means Timestral is worse than the referee on the best series found.";
        } else if (st.view === "grid") {
          C.segmented(bar2, d.methods.filter(function (m) { return d.grid.methods[m[0]]; }).map(function (m) { return [m[0], m[1]]; }), st.method, function (v) { st.method = v; show(); }, "method");
          C.segmented(bar2, [["1.5", "cells at 1.5× or worse"], ["2.0", "2×"], ["3.0", "3×"]], st.thr, function (v) { st.thr = v; show(); }, "threshold");
          var g = d.grid.methods[st.method] || { cells: {}, runs: 0 }, t = parseFloat(st.thr), bins = d.grid.bins, filled = 0, hit = 0;
          var cols = []; for (var b = 0; b < bins; b++) cols.push(String(b + 1));
          C.heat(plot, { rows: d.grid.families, cols: cols, colors: STEPS, legend: ["empty", "below 1.5×", "1.5× to 2×", "2× to 3×", "3× or more"],
            colTitle: "spectral entropy of the series, low (smooth) to high (noisy)", label: names[st.method] + ": best failure found in each cell of the shared grid",
            hint: "Move over the grid to read a cell.", cell: function (i, j) {
              var c = g.cells[d.grid.families[i] + "|" + j];
              if (!c) return null;
              var v = c[0], step = v >= 3 ? 4 : v >= 2 ? 3 : v >= 1.5 ? 2 : 1;
              return { step: step, text: d.grid.families[i] + ", entropy bin " + (j + 1) + ": best error ÷ referee's error " + C.fmt(v) + ", reached in " + c[1] + " of " + g.runs + " runs" };
            } });
          Object.keys(g.cells).forEach(function (k) { filled++; if (g.cells[k][0] >= t) hit++; });
          note.textContent = names[st.method] + ": " + filled + " cells filled by at least one of its " + g.runs + " runs; " + hit + " of them hold a failure at " + t + "× or worse (best over seeds per cell). Families are the dominant piece in the series; each series is placed in one cell.";
        } else {
          var T = root.OTPRows(d.transfer), rows = [], sizes = [];
          T.forEach(function (r) { if (sizes.indexOf(r.size) < 0) sizes.push(r.size); });
          C.segmented(bar2, [["frac_gt_1.5", "still above 1.5×"], ["frac_gt_2", "still above 2×"]], st.thr === "2.0" ? "frac_gt_2" : "frac_gt_1.5", function (v) { st.thr = v === "frac_gt_2" ? "2.0" : "1.5"; show(); }, "threshold");
          var key = st.thr === "2.0" ? "frac_gt_2" : "frac_gt_1.5";
          sizes.forEach(function (s) {
            d.methods.forEach(function (m) {
              var r = T.filter(function (x) { return x.size === s && x.method === m[0]; })[0];
              if (r) rows.push({ group: "Timestral-" + s, label: m[1], v: r[key], color: COLORS[m[0]], legend: m[1], tip: m[1] + " on " + s + ": " + C.fmt(r[key] * 100) + "% of its best series still above " + st.thr + "× (" + r.n + " series)" });
            });
          });
          C.dots(plot, { rows: rows, xMin: 0, xMax: 1, xFmt: function (v) { return Math.round(v * 100) + "%"; }, vFmt: function (v) { return Math.round(v * 100) + "%"; },
                         xLabel: "share of the best series found on 5M that still fail on this size", label: "Transfer of failures found on 5M to larger sizes" });
          note.textContent = "Top series from each method, re-scored with fresh seeds on each size.";
        }
      }
      if (views.length > 1) C.segmented(bar, views, st.view, function (v) { st.view = v; show(); }, "view");
      show();
    }),
  });
})(window);
