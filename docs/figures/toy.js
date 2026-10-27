/* "Beat it" window: mix a series from a few pieces and see how far Timestral falls behind the referee.
   Every combination was computed ahead of time by scripts/site_toy.py (data/toy.json); the page only looks it up. */
(function (root) {
  "use strict";
  var C = root.OTPChart;
  var WILD_NAMES = { none: "nothing", mackey_glass: "delay feedback (Mackey-Glass)", kuramoto: "coupled oscillators (Kuramoto)",
                     rossler: "smooth chaos (Rössler)", lorenz63: "butterfly chaos (Lorenz)", mean_jump: "level jump, 16 steps ago",
                     variance_jump: "noise jump, 16 steps ago", stable_ar: "wild jumps (α-stable)", hawkes: "bursts (Hawkes)" };
  var KEY = "otp-toy-best";

  function readBest() { try { return JSON.parse(root.localStorage.getItem(KEY) || "null"); } catch (e) { return null; } }
  function writeBest(v) { try { root.localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) { /* storage blocked: fine */ } }

  root.OTPToy = {
    mount: function (host) {
      host.textContent = "";
      var wrap = C.h("div", { class: "toy" });
      host.appendChild(wrap);
      wrap.appendChild(C.h("h2", { text: "Try to beat it" }));
      wrap.appendChild(C.h("p", { text: "Mix a series and see whether Timestral keeps up with the referee (a forecaster that knows the true equations). Your goal: make Timestral look bad. The score is Timestral's error divided by the referee's: 1 means it kept up, 5 means five times worse." }));
      var ctl = C.h("div", { class: "toy-controls" });
      var plot = C.h("div", { class: "toy-plot" });
      var score = C.h("div", { class: "toy-score", "aria-live": "polite" });
      wrap.appendChild(ctl); wrap.appendChild(score); wrap.appendChild(plot);
      root.OTPData.get("toy").then(function (toy) {
        var by = {};
        toy.items.forEach(function (it) { by[it.key] = it; });
        var st = { season: toy.controls.season[1], wild: "none", strength: toy.controls.strength[1], noise: toy.controls.noise[1], size: "209M" };
        function key() { return "s" + st.season + "|" + st.wild + "|" + (st.wild === "none" ? 0 : st.strength) + "|n" + st.noise; }
        function row(label, opts, k) {
          var r = C.h("div", { class: "toy-row" }, [C.h("span", { class: "toy-label", text: label })]);
          C.segmented(r, opts, st[k], function (v) { st[k] = v; show(); }, label);
          ctl.appendChild(r);
        }
        row("Season strength", toy.controls.season.map(function (v) { return [v, v === 0 ? "none" : String(v)]; }), "season");
        row("Wild piece", toy.controls.wild.map(function (v) { return [v, WILD_NAMES[v] || v]; }), "wild");
        row("Wild strength", toy.controls.strength.map(function (v) { return [v, String(v)]; }), "strength");
        row("Noise", toy.controls.noise.map(function (v) { return [v, String(v)]; }), "noise");
        row("Model", [["5M", "Timestral-5M"], ["209M", "Timestral-209M"]], "size");
        function show() {
          var it = by[key()];
          if (!it) { plot.textContent = "This combination was not computed."; return; }
          root.OTPFan.draw(plot, toy, it, st.size);
          var r = it["ratio" + st.size];
          var best = readBest();
          if (r !== null && (!best || r > best.ratio)) { best = { ratio: r, key: it.key, size: st.size }; writeBest(best); }
          score.textContent = "";
          score.appendChild(C.h("span", { class: "toy-big", text: r === null ? "–" : C.fmt(r) + "×" }));
          score.appendChild(C.h("span", { text: " Timestral-" + st.size + "'s error vs the referee's on this series." }));
          if (best) score.appendChild(C.h("span", { class: "toy-best", text: "  Your worst-case so far: " + C.fmt(best.ratio) + "× (" + best.size + ")." }));
          if (it.nonfinite_oracle_paths) score.appendChild(C.h("span", { text: " (" + it.nonfinite_oracle_paths + " referee paths blew up and were dropped.)" }));
        }
        show();
        var srch = C.h("div", { class: "toy-search" });
        wrap.appendChild(srch);
        root.OTPData.get("search").then(function (d) {
          if (d.status !== "ready") srch.appendChild(C.h("p", { class: "otp-note", text: "Coming: the worst series our search algorithms found, to compare with yours (fills from results/search/)." }));
          else srch.appendChild(C.h("p", { class: "otp-note", text: "Search results are in: see 'Letting a computer hunt for failures' in the Story." }));
        });
        wrap.appendChild(C.h("p", { class: "otp-note", text: "One fixed random seed per combination; history 512 steps (the last 256 shown), forecast 64 steps. Precomputed by scripts/site_toy.py." }));
      });
    },
  };
})(window);
