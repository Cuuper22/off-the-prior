/* atlas: the neuron atlas summary. Four views from data/neurons.json (built from results/neurons/*.csv):
   share of poles over the threshold, by depth, corpus dependence, and the steering test. */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  function pcf(v) { return (v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0) + "%"; }
  var CLASSES = [["one_job", "One job"], ["mixed", "Mixed"], ["unexplained", "Unexplained"]];
  var CORPUS = { main: "main (1,024-step history)", short: "short histories", long: "long histories", xlong: "very long histories" };
  var ROLE = { labeled: "labeled poles", complement: "the neuron's other pole", control: "random poles (control)" };

  F.define("atlas", {
    title: "Inside the neurons",
    label: "Charts of how many Timestral poles are explained by a single known variable, by threshold, depth, corpus, and the steering test",
    summary: "Every pole (one sign of one neuron) of every size, scored against variables known exactly for the generated series. Pick a view.",
    build: root.OTPSlot("neurons", function (ctx, d) {
      var bar = C.h("div", { class: "otp-controls" }), bar2 = C.h("div", { class: "otp-controls" }), plot = C.h("div"), note = C.h("p", { class: "otp-note", "aria-live": "polite" });
      ctx.el.insertBefore(note, ctx.el.firstChild);
      ctx.el.insertBefore(plot, note);
      ctx.el.insertBefore(bar2, plot);
      ctx.el.insertBefore(bar, bar2);
      var cur = rowsOf(d.curves), lay = rowsOf(d.layers), cor = rowsOf(d.corpora), ste = rowsOf(d.steering);
      var st = { view: "theta", cls: "one_job", floor: 0.1 };
      function sizeLines(rows, xf, yf, extra) {
        return C.SIZES.map(function (s) {
          var r = rows.filter(function (x) { return x.size === s; });
          return Object.assign({ name: s, color: C.SIZE_COLORS[s], points: r.map(function (x) { var p = [xf(x), yf(x)]; return p; }),
                                 band: extra ? r.map(function (x) { return [xf(x), extra.lo(x), extra.hi(x)]; }) : [] }, {});
        });
      }
      function show() {
        bar2.textContent = "";
        if (st.view === "theta") {
          C.segmented(bar2, CLASSES, st.cls, function (v) { st.cls = v; show(); }, "class of pole");
          C.segmented(bar2, d.floors_ui || [[0.05, "floor 0.05"], [0.1, "floor 0.1"], [0.2, "floor 0.2"]], st.floor, function (v) { st.floor = v; show(); }, "floor");
          var rows = cur.filter(function (x) { return Math.abs(x.floor - st.floor) < 1e-9; });
          var ext = st.cls === "one_job" ? { lo: function (x) { return x.one_job_lo; }, hi: function (x) { return x.one_job_hi; } } : null;
          C.line(plot, { series: sizeLines(rows, function (x) { return x.theta; }, function (x) { return x[st.cls]; }, ext), yFmt: pcf, xLabel: "threshold θ (stricter to the right)",
                         yLabel: "share of live poles", yMin: 0, xFmt: function (v) { return v.toFixed(1); },
                         label: "Share of poles by class against the threshold, one line per model size" });
          note.textContent = st.cls === "unexplained" ? "Unexplained poles do not depend on θ: they are the ones no combination of library variables explains above the floor." :
            "A pole has one job when its best single variable explains at least θ of what the best combination explains. Raising θ makes it harder. The floor is the least explainable variance a pole needs to count as explained at all.";
        } else if (st.view === "depth") {
          var ext2 = { lo: function (x) { return x.one_job_lo; }, hi: function (x) { return x.one_job_hi; } };
          C.line(plot, { series: sizeLines(lay, function (x) { return x.depth; }, function (x) { return x.one_job; }, ext2), yFmt: pcf, yMin: 0,
                         xLabel: "depth in the model (0 = first layer, 1 = last layer)", yLabel: "one-job share of live poles", xFmt: function (v) { return v.toFixed(1); },
                         label: "One-job share by layer depth, one line per model size" });
          note.textContent = "Each dot is one layer; the shaded band is the 95% bootstrap interval over poles. θ = " + d.theta + ", floor = " + d.floor + ".";
        } else if (st.view === "corpus") {
          var rows2 = [];
          Object.keys(CORPUS).forEach(function (c) {
            C.SIZES.forEach(function (s) {
              var r = cor.filter(function (x) { return x.size === s && x.corpus === c; })[0];
              if (r) rows2.push({ group: CORPUS[c], label: s, v: r.one_job, color: C.SIZE_COLORS[s], legend: s,
                                  tip: s + ", " + CORPUS[c] + ": " + pcf(r.one_job) + " one job; " + pcf(r.same_label_as_main) + " keep their main-corpus label" });
            });
          });
          C.dots(plot, { rows: rows2, xMin: 0, xFmt: pcf, vFmt: pcf, xLabel: "share of live poles with one job (θ = " + d.theta + ", floor = " + d.floor + ")", label: "One-job share on four corpora, per size" });
          note.textContent = "Same models, same generator recipe, different history lengths. The share is a property of the pair (neurons, corpus), not of the neurons alone: far fewer of a pole's labels survive a change of corpus.";
        } else {
          var rows3 = [];
          C.SIZES.forEach(function (s) {
            ["labeled", "complement", "control"].forEach(function (role) {
              var r = ste.filter(function (x) { return x.size === s && x.role === role; })[0];
              if (r) rows3.push({ group: "Timestral-" + s, label: ROLE[role], v: r.match_rate, lo: r.ci_lo, hi: r.ci_hi, color: C.SIZE_COLORS[s],
                                  tip: s + ", " + ROLE[role] + ": " + r.n_match + " of " + r.n + " matched (95% interval " + pcf(r.ci_lo) + " to " + pcf(r.ci_hi) + ")" });
            });
          });
          C.dots(plot, { rows: rows3, xMin: 0, xFmt: pcf, vFmt: pcf, xLabel: "share of steered poles whose forecast moved as predicted", label: "Steering test: hit rate with 95% interval, per size and kind of pole" });
          note.textContent = "Each pole was scaled up or down during the forecast and compared with a prediction written down beforehand. The moves were smaller than the pre-set size thresholds, so this test could not detect a match; a hit rate of zero is not evidence against the labels.";
        }
      }
      C.segmented(bar, [["theta", "By threshold"], ["depth", "By depth"], ["corpus", "By corpus"], ["steer", "Steering test"]], st.view, function (v) { st.view = v; show(); }, "view");
      show();
      var b = C.h("p", { class: "go-deeper" }, [C.h("a", { href: "#atlas", "data-open": "atlas", text: "Open the neuron browser" })]);
      ctx.el.appendChild(b);
    }),
  });
  function rowsOf(t) { return root.OTPRows(t); }
})(window);
