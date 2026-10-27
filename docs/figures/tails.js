/* tails: heavy tails, long memory, bursts and cascades stay close to the referee. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var GENS = [["stable_ar", "Wild jumps (α-stable)"], ["fbm", "Long memory (fBm)"], ["hawkes", "Bursts (Hawkes)"], ["msm_cascade", "Cascades (MSM)"]];
  F.define("tails", {
    title: "Wild outliers stay close",
    label: "Line chart of Timestral's error divided by the referee's for heavy-tailed and long-memory generators",
    summary: "Pick a generator. Most points sit near 1: Timestral is about as good as the referee here. History 2,048 steps, horizon 128.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" }), plot = C.h("div");
      ctx.el.appendChild(bar); ctx.el.appendChild(plot);
      root.OTPMap.rows().then(function (rows) {
        var t = rows.filter(function (r) { return r.family === "tails" && r.T === 2048 && r.H === 128; });
        function show(gen) {
          var rr = t.filter(function (r) { return r.generator === gen; });
          var keys = []; rr.forEach(function (r) { if (keys.indexOf(r.config_id) < 0) keys.push(r.config_id); });
          keys.sort();
          root.OTPMap.draw(plot, rr, { x: function (r) { return keys.indexOf(r.config_id); },
                                       xLabelOf: function (r) { return r.config_id.split("/").slice(2).join("/"); },
                                       xTicks: keys.map(function (k, i) { return [i, k.split("/").slice(2).join("/")]; }),
                                       xLabel: "setting" });
        }
        C.segmented(bar, GENS, GENS[0][0], show, "generator");
        show(GENS[0][0]);
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
