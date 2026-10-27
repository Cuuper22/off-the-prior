/* chaos-map: Timestral's error relative to the referee along each chaos knob, per model size. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var SYSTEMS = [
    ["mackey_glass", "Mackey-Glass", "delay τ"],
    ["rossler", "Rössler", "parameter c"],
    ["kuramoto", "Kuramoto", "coupling K"],
    ["lorenz63", "Lorenz", "ρ"],
    ["logistic_map", "Logistic map", "r"],
  ];
  F.define("chaos-map", {
    title: "Chaos: how far behind the referee",
    label: "Line chart of Timestral's error divided by the referee's, along each chaos system's knob, one line per model size",
    summary: "Pick a system and a horizon. Each line is one Timestral size (light = small, dark = big). 1 means as good as the referee; the shaded band is a 95% interval over 8 seeds.",
    build: function (ctx) {
      var bar = C.h("div", { class: "otp-controls" }), bar2 = C.h("div", { class: "otp-controls" }), plot = C.h("div");
      ctx.el.appendChild(bar); ctx.el.appendChild(bar2); ctx.el.appendChild(plot);
      var st = { gen: "mackey_glass", H: 128, vs: "oracle" };
      root.OTPMap.rows().then(function (rows) {
        var chaos = rows.filter(function (r) { return r.family === "chaos" && r.T === 2048 && r.config_id.indexOf("_oe") < 0; });
        function show() {
          var sys = SYSTEMS.filter(function (s) { return s[0] === st.gen; })[0];
          var rr = chaos.filter(function (r) { return r.generator === st.gen && r.H === st.H; });
          root.OTPMap.draw(plot, rr, { x: function (r) { return parseFloat(r.knob_value); }, xLabel: sys[2], vsClassical: st.vs === "classical",
                                       label: sys[1] + " at horizon " + st.H });
        }
        C.segmented(bar, SYSTEMS.map(function (s) { return [s[0], s[1]]; }), st.gen, function (v) { st.gen = v; show(); }, "system");
        C.segmented(bar2, [[16, "16 steps ahead"], [128, "128"], [1024, "1,024"]], st.H, function (v) { st.H = v; show(); }, "horizon");
        C.segmented(bar2, [["oracle", "vs referee"], ["classical", "vs old-school"]], st.vs, function (v) { st.vs = v; show(); }, "compared with");
        show();
      });
      return root.OTPStatic(ctx);
    },
  });
})(window);
