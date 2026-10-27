/* scaling: error ratio against model size, one line per generator. Data: data/scaling.json (results/map/chaos and regime summaries). */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var NAMES = { mackey_glass: "Mackey-Glass", rossler: "Rössler", kuramoto: "Kuramoto", lorenz63: "Lorenz", logistic_map: "Logistic map",
                mean_jump: "level jump", variance_jump: "noise jump", frequency_jump: "season changes length", markov_ar: "random switching" };
  var COLORS = ["#0e7d84", "#b4531c", "#6a4c93", "#3987e5", "#8a6d00"];
  F.define("scaling", {
    title: "Does a bigger model close the gap?",
    label: "Line chart of Timestral's error divided by the referee's against model size on a log scale, one line per generator",
    summary: "Pick a family, a horizon and whether to look at each generator's hardest setting or the average over its settings. Both axes are logarithmic.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" }), bar2 = C.h("div", { class: "otp-controls" }), plot = C.h("div"), note = C.h("p", { class: "otp-note" });
      ctx.el.appendChild(bar); ctx.el.appendChild(bar2); ctx.el.appendChild(plot); ctx.el.appendChild(note);
      var st = { fam: "chaos", H: 128, stat: "hardest" };
      root.OTPData.get("scaling").then(function (d) {
        var rows = root.OTPRows(d);
        function show() {
          var rr = rows.filter(function (r) { return r.family === st.fam && r.H === st.H && r.stat === st.stat; });
          var series = rr.map(function (r, i) {
            var pts = d.sizes.map(function (s, k) { var p = [d.params_millions[k], r["r" + s]]; p.label = s + " (" + r.config + ")"; return p; });
            return { name: NAMES[r.generator] || r.generator, color: COLORS[i % COLORS.length], points: pts };
          });
          C.line(plot, { series: series, xlog: true, ylog: true, ref1: true, xLabel: "model size (millions of parameters)", yLabel: "error ÷ referee's error",
                         xTicks: d.sizes.map(function (s, k) { return [d.params_millions[k], s]; }), label: "Error ratio against model size, one line per generator" });
          note.textContent = st.stat === "hardest" ?
            "Each line follows the one setting of that generator where the smallest model does worst. Hover a point to see which setting." :
            "Each line is the geometric mean over every setting of that generator in the map.";
        }
        C.segmented(bar, [["chaos", "Chaos (history 2,048)"], ["regime", "Regime switches (history 8,192)"]], st.fam, function (v) { st.fam = v; show(); }, "family");
        C.segmented(bar2, [[16, "16 steps ahead"], [128, "128"], [1024, "1,024"]], st.H, function (v) { st.H = v; show(); }, "horizon");
        C.segmented(bar2, [["hardest", "hardest setting"], ["average", "average over settings"]], st.stat, function (v) { st.stat = v; show(); }, "which setting");
        show();
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
