/* atlas: the neuron atlas summary (slot; fills from results/neurons/one_job_curves.csv and results/neurons/browser/). */
(function (root) {
  "use strict";
  var F = root.OTPFigures, C = root.OTPChart;
  F.define("atlas", {
    title: "Inside the neurons",
    label: "How many neurons have one clear job, per model size",
    summary: "Every neuron of every size, scored against things we know exactly about the series it reads.",
    build: root.OTPSlot("neurons", function (ctx, d) {
      if (d.one_job_curves) C.table(ctx.el, d.one_job_curves.columns, d.one_job_curves.rows.slice(0, 40));
      var b = C.h("p", { class: "go-deeper" }, [C.h("a", { href: "#atlas", "data-open": "atlas", text: "Open the neuron browser" })]);
      ctx.el.appendChild(b);
    }),
  });
})(window);
