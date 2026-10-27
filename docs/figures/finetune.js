/* finetune: what cheap fine-tuning buys and costs. Data: data/finetune.json (results/finetune/<local|colab>/summary.csv and costs.csv).
   Views: change in error on held-out failure families, gain against cost, damage on normal series. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  var MC = { full: "#1c5cab", lora: "#b4531c" };
  function variant(r) { return (r.method === "lora" ? "LoRA" : "full") + (r.replay > 0 ? " + replay" : ""); }
  var FAMS = { chaos: "chaos", regime: "regime switches", tails: "heavy tails", mixture: "mixtures" };
  var PROTO = [["lofo", "Family left out"], ["knob", "Difficulty range left out"]];
  var VARIANTS = ["full", "full + replay", "LoRA", "LoRA + replay"];
  F.define("finetune", {
    title: "Teaching it the hard cases",
    label: "Charts of fine-tuning results: error change on held-out failure families, gain against cost, and change on normal series",
    summary: "Full fine-tuning and LoRA, each with and without replay of normal series, tested on families or difficulty ranges the training never saw.",
    build: root.OTPSlot("finetune", function (ctx, d) {
      var bar = C.h("div", { class: "otp-controls" }), bar2 = C.h("div", { class: "otp-controls" }), plot = C.h("div"), note = C.h("p", { class: "otp-note" });
      ctx.el.appendChild(bar); ctx.el.appendChild(bar2); ctx.el.appendChild(plot); ctx.el.appendChild(note);
      var T = root.OTPRows(d.tests), G = root.OTPRows(d.regression);
      var st = { view: "gain", size: d.sizes[d.sizes.length - 1], proto: "lofo", cost: "wall_sec", set: "in_prior" };
      function show() {
        bar2.textContent = "";
        if (d.sizes.length > 1) C.segmented(bar2, d.sizes.map(function (s) { return [s, "Timestral-" + s]; }), st.size, function (v) { st.size = v; show(); }, "model size");
        if (st.view !== "damage") C.segmented(bar2, PROTO, st.proto, function (v) { st.proto = v; show(); }, "protocol");
        if (st.view === "cost") C.segmented(bar2, [["wall_sec", "training time"], ["trainable_params", "trainable parameters"]], st.cost, function (v) { st.cost = v; show(); }, "cost measure");
        if (st.view === "damage") C.segmented(bar2, [["in_prior", "500 in-prior series"], ["real", "real windows"]], st.set, function (v) { st.set = v; show(); }, "regression set");
        var rows = T.filter(function (r) { return r.size === st.size && r.protocol === st.proto; });
        if (st.view === "gain") {
          var items = [];
          Object.keys(FAMS).forEach(function (f) {
            VARIANTS.forEach(function (v) {
              var r = rows.filter(function (x) { return x.holdout === f && variant(x) === v; })[0];
              if (r) items.push({ group: "Held out: " + FAMS[f], label: v, v: r.paired, lo: r.lo, hi: r.hi, color: MC[r.method], legend: r.method === "lora" ? "LoRA" : "full fine-tuning",
                                  tip: v + ", " + FAMS[f] + ": error × " + C.fmt(r.paired) + " (95% " + C.fmt(r.lo) + " to " + C.fmt(r.hi) + ")" });
            });
          });
          C.dots(plot, { rows: items, ref: 1, xFmt: function (v) { return v.toFixed(2); }, vFmt: function (v) { return v.toFixed(3); }, xLabel: "fine-tuned error ÷ original error (below 1 is better)",
                         label: "Change in error on held-out series for each fine-tuning variant" });
          note.textContent = "Each dot compares the fine-tuned model with the original on the same 32 held-out series per family; the line is a 95% interval. Replay means part of every training batch was normal in-prior series.";
        } else if (st.view === "cost") {
          var series = ["full", "lora"].map(function (m) {
            return { name: m === "lora" ? "LoRA" : "full fine-tuning", color: MC[m], scatter: true, points: rows.filter(function (r) { return r.method === m && r[st.cost] !== null && r.gap_mase !== null; }).map(function (r) {
              var p = [r[st.cost], r.gap_mase]; p.label = FAMS[r.holdout] + (r.replay > 0 ? ", replay" : "") + ", cost " + (st.cost === "wall_sec" ? C.fmt(r[st.cost]) + " s" : r[st.cost].toLocaleString("en-US") + " weights"); return p; }) };
          });
          C.line(plot, { series: series, xlog: true, xLabel: st.cost === "wall_sec" ? "training time (seconds)" : "trainable parameters", yLabel: "gap to the referee closed", yFmt: function (v) { return (v * 100).toFixed(0) + "%"; },
                         label: "Share of the gap to the referee closed against the cost of training" });
          note.textContent = "Gap closed = (original error − fine-tuned error) ÷ (original error − referee's error), on the held-out series. Zero means no change; negative means the fine-tuned model got worse.";
        } else {
          var gr = G.filter(function (r) { return r.size === st.size && r.set === st.set; }), items2 = [];
          VARIANTS.forEach(function (v) {
            gr.filter(function (r) { return variant(r) === v; }).forEach(function (r) {
              items2.push({ group: v, label: r.protocol + "/" + r.holdout, v: r.paired, lo: r.lo, hi: r.hi, color: MC[r.method], legend: r.method === "lora" ? "LoRA" : "full fine-tuning",
                            tip: r.run + ": error × " + C.fmt(r.paired) + (r.lo !== null ? " (95% " + C.fmt(r.lo) + " to " + C.fmt(r.hi) + ")" : "") });
            });
          });
          C.dots(plot, { rows: items2, ref: 1, xFmt: function (v) { return v.toFixed(2); }, vFmt: function (v) { return v.toFixed(3); }, xLabel: "fine-tuned error ÷ original error on normal series (above 1 is worse)",
                         label: "Change in error on normal series after fine-tuning, one dot per trained model" });
          note.textContent = "Normal series are outside the failure families. Each dot is one trained model: its group is the training variant, its row says which protocol and held-out family it was trained for.";
        }
      }
      C.segmented(bar, [["gain", "Held-out failures"], ["cost", "Gain against cost"], ["damage", "Normal series"]], st.view, function (v) { st.view = v; show(); }, "view");
      show();
    }),
  });
})(window);
