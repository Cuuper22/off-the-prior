/* fan: what a probabilistic forecast looks like. Draws one precomputed example from data/toy.json: the end of the
   history, what really happened, Timestral's median and 80% band, and the referee's (oracle's) median and band. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var INK = "#1c1d21", TS = "#1c5cab", OR = "#b4531c";

  function draw(host, toy, item, size) {
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
      label: "History, truth, Timestral forecast and referee forecast",
    });
  }
  root.OTPFan = { draw: draw };

  var EXAMPLES = [
    ["s1.4|none|0|n0.1", "Home turf: season + noise"],
    ["s0.7|mackey_glass|1.4|n0.1", "Delay-feedback chaos"],
    ["s0.0|mean_jump|1.4|n0.01", "Level jumped 16 steps ago"],
  ];

  F.define("fan", {
    title: "What a forecast looks like",
    label: "A history, what really happened next, and two forecasts drawn as a median line with an 80% band",
    summary: "Pick an example. The solid black line is the history, the dotted line is what happened next, blue is Timestral's forecast and orange is the referee's.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" });
      var plot = C.h("div");
      var note = C.h("p", { class: "otp-note" });
      ctx.el.appendChild(bar); ctx.el.appendChild(plot); ctx.el.appendChild(note);
      root.OTPData.get("toy").then(function (toy) {
        var by = {};
        toy.items.forEach(function (it) { by[it.key] = it; });
        function show(key) {
          var it = by[key];
          draw(plot, toy, it, "209M");
          note.textContent = it.ratio209M === null ? "" :
            "On this one, Timestral-209M's error is " + C.fmt(it.ratio209M) + "× the referee's.";
        }
        C.segmented(bar, EXAMPLES, EXAMPLES[0][0], show, "example");
        show(EXAMPLES[0][0]);
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
