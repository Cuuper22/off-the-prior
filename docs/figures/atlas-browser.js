/* Atlas window: the neuron browser. Until results/neurons/browser/*.json exist this says so; the site worker
   builds the browser itself on top of the files listed in data/neurons.json. */
(function (root) {
  "use strict";
  var C = root.OTPChart;
  root.OTPAtlas = {
    mount: function (host) {
      host.textContent = "";
      var wrap = C.h("div", { class: "toy" });
      host.appendChild(wrap);
      wrap.appendChild(C.h("h2", { text: "Neuron atlas" }));
      wrap.appendChild(C.h("p", { text: "Every neuron of every Timestral size, split into its two halves (it can fire positive or negative), each tagged with the thing it tracks best: local level, slope, noise, season phase, how long since a switch, and so on." }));
      var out = C.h("div"); wrap.appendChild(out);
      root.OTPData.get("neurons").then(function (d) {
        if (d.status !== "ready") return C.pending(out, d);
        out.appendChild(C.h("p", { class: "otp-note", text: "Browser data files: " + d.files.join(", ") }));
      }).catch(function () { C.pending(out, null); });
    },
  };
})(window);
