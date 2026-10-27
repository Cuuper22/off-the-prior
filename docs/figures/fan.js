/* fan: what a probabilistic forecast looks like. Draws one precomputed example from data/toy.json: the end of the
   history, what really happened, Timestral's median and 80% band, and the referee's (oracle's) median and band. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var INK = "#1c1d21", TS = "#1c5cab", OR = "#b4531c";

  function draw(host, toy, item, size, onDraw) {
    var n = item.context.length, Hh = item.future.length;
    var hist = item.context.map(function (v, i) { return [i - n, v]; });
    var truth = [[-1, item.context[n - 1]]].concat(item.future.map(function (v, i) { return [i, v]; }));
    var q = item["q" + size], o = item.oracle;
    function band(qq) { return qq[1].map(function (_, i) { return [i, qq[0][i], qq[2][i]]; }); }
    function med(qq) { return qq[1].map(function (v, i) { return [i, v]; }); }
    return C.line(host, {
      series: [
        { name: "history", color: INK, points: hist, marker: false, width: 1.5 },
        { name: "what really happened", color: INK, points: truth, marker: false, width: 1.5, dash: "2 3" },
        { name: "Timestral-" + size + " (median, 80% band)", color: TS, points: med(q), band: band(q), marker: false },
        { name: "referee (median, 80% band)", color: OR, points: med(o), band: band(o), marker: false, dash: "6 4", bandOpacity: 0.1 },
      ],
      xLabel: "steps (0 = now)", yLabel: "value", height: 280,
      label: "History, truth, Timestral forecast and referee forecast", onDraw: onDraw,
    });
  }
  root.OTPFan = { draw: draw };

  var EXAMPLES = [
    ["s1.4|none|0|n0.1", "Home turf: season + noise"],
    ["s0.7|mackey_glass|1.4|n0.1", "Delay-feedback chaos"],
    ["s0.0|mean_jump|1.4|n0.01", "Level jumped 16 steps ago"],
  ];

  /* The animation: the history draws in, the truth appears, Timestral's median and its band open out from the median,
     then the referee's. Every frame is a function of time, so the last frame is the still picture used for reduced motion. */
  var T_HIST = 1.8, T_TRUTH = 2.4, T_BAND = 4.4, T_REF = 5.6;
  function clamp01(x) { return Math.max(0, Math.min(1, x)); }
  function ease(x) { x = clamp01(x); return x * x * (3 - 2 * x); }

  F.define("fan", {
    title: "What a forecast looks like",
    label: "A history, what really happened next, and two forecasts drawn as a median line with an 80% band",
    summary: "Pick an example. The solid black line is the history, the dotted line is what happened next, blue is Timestral's forecast and orange is the referee's.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" });
      var plot = C.h("div");
      var note = C.h("p", { class: "otp-note" });
      ctx.el.appendChild(bar); ctx.el.appendChild(plot); ctx.el.appendChild(note);
      var cur = { handles: null, item: null, bands: null, t: T_REF };
      function bandPath(res, band, k, med) {
        var up = band.filter(function (b) { return isFinite(b[1]) && isFinite(b[2]); });
        function lo(b, i) { return med[i][1] + (b[1] - med[i][1]) * k; }
        function hi(b, i) { return med[i][1] + (b[2] - med[i][1]) * k; }
        return up.map(function (b, i) { return (i ? "L" : "M") + res.X(b[0]) + " " + res.Y(hi(b, i)); }).join(" ") + " " +
          up.slice().reverse().map(function (b, i) { var j = up.length - 1 - i; return "L" + res.X(b[0]) + " " + res.Y(lo(b, j)); }).join(" ") + " Z";
      }
      function render(t) {
        cur.t = t;
        var res = cur.handles; if (!res || !res.series) return;
        var S = res.series;                       // [history, truth, timestral, referee]
        var hist = S[0].path, truth = S[1].path, ts = S[2].path, ref = S[3].path;
        hist.setAttribute("pathLength", "1"); hist.style.strokeDasharray = "1 1"; hist.style.strokeDashoffset = String(1 - ease(t / T_HIST));
        truth.style.opacity = String(ease((t - T_HIST) / (T_TRUTH - T_HIST)));
        var kts = ease((t - T_TRUTH) / (T_BAND - T_TRUTH));
        ts.style.opacity = String(clamp01((t - T_TRUTH) / 0.5));
        if (S[2].s.bandEl) { S[2].s.bandEl.setAttribute("d", bandPath(res, cur.bands.ts, kts, cur.bands.tsMed)); S[2].s.bandEl.style.opacity = String(clamp01((t - T_TRUTH) / 0.5)); }
        var kref = ease((t - T_BAND) / (T_REF - T_BAND));
        ref.style.opacity = String(clamp01((t - T_BAND) / 0.5));
        if (S[3].s.bandEl) { S[3].s.bandEl.setAttribute("d", bandPath(res, cur.bands.ref, kref, cur.bands.refMed)); S[3].s.bandEl.style.opacity = String(clamp01((t - T_BAND) / 0.5)); }
      }
      var toyData = null, by = {};
      function show(key, animate) {
        var it = by[key];
        cur.item = it;
        var q = it["q209M"], o = it.oracle;
        function band(qq) { return qq[1].map(function (_, i) { return [i, qq[0][i], qq[2][i]]; }); }
        function med(qq) { return qq[1].map(function (v, i) { return [i, v]; }); }
        cur.bands = { ts: band(q), tsMed: med(q), ref: band(o), refMed: med(o) };
        draw(plot, toyData, it, "209M", function (res) { cur.handles = res; render(cur.t); });
        note.textContent = it.ratio209M === null ? "" : "On this one, Timestral-209M's error is " + C.fmt(it.ratio209M) + "× the referee's.";
        var g = ctx.root.__gsf;
        if (animate && g && !ctx.reduced) g.replay(); else render(T_REF);
      }
      root.OTPData.get("toy").then(function (toy) {
        toyData = toy;
        toy.items.forEach(function (it) { by[it.key] = it; });
        C.segmented(bar, EXAMPLES, EXAMPLES[0][0], function (k) { show(k, true); }, "example");
        show(EXAMPLES[0][0], false);
        if (!ctx.reduced) render(ctx.root.__gsf && ctx.root.__gsf.visible ? cur.t : 0);
      });
      return { duration: T_REF + 0.6, still: T_REF + 0.6, render: render, autoplay: true };
    },
  });
})(window);
