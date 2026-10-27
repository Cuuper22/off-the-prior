/* "Beat it" window: mix a series from a few pieces and see how far Timestral falls behind the referee.
   Two ways to run it, and the page says which one is on:
   - precomputed: every combination was computed ahead of time by scripts/site_toy.py (data/toy.json); the page only
     looks it up. Both Timestral-5M and -209M, one fixed random seed per combination.
   - live: a new random series each time, made by a JavaScript port of the generators (live/generators.js), forecast
     by Timestral-5M running in this browser (live/timestral.js, weights from scripts/export_site_weights.py), and
     scored against a referee that simulates 200 continuations from the exact state. It runs in a web worker
     (live/worker.js) and loads about 10 MB the first time it is switched on. */
(function (root) {
  "use strict";
  var C = root.OTPChart;
  var WILD_NAMES = { none: "nothing", mackey_glass: "delay feedback (Mackey-Glass)", kuramoto: "coupled oscillators (Kuramoto)",
                     rossler: "smooth chaos (Rössler)", lorenz63: "butterfly chaos (Lorenz)", mean_jump: "level jump, 16 steps ago",
                     variance_jump: "noise jump, 16 steps ago", stable_ar: "wild jumps (α-stable)", hawkes: "bursts (Hawkes)" };
  var KEY = "otp-toy-best";

  function readBest() { try { return JSON.parse(root.localStorage.getItem(KEY) || "null"); } catch (e) { return null; } }
  function writeBest(v) { try { root.localStorage.setItem(KEY, JSON.stringify(v)); } catch (e) { /* storage blocked: fine */ } }

  /* The worker, created on first use. Resolves with the info message once the weights are in. */
  function LiveEngine(onProgress) {
    var self = this, worker = null, readyInfo = null, waiting = [], jobs = {}, nextId = 1;
    this.available = typeof root.Worker === "function";
    this.start = function () {
      if (readyInfo) return Promise.resolve(readyInfo);
      return new Promise(function (resolve, reject) {
        waiting.push([resolve, reject]);
        if (worker) return;
        try { worker = new root.Worker("live/worker.js"); } catch (e) { waiting.splice(0).forEach(function (w) { w[1](e); }); return; }
        worker.onmessage = function (e) {
          var m = e.data;
          if (m.type === "progress") onProgress(m.got, m.total);
          else if (m.type === "ready") { readyInfo = m.info; waiting.splice(0).forEach(function (w) { w[0](m.info); }); }
          else if (m.type === "result" && jobs[m.id]) { jobs[m.id][0](m.item); delete jobs[m.id]; }
          else if (m.type === "error") {
            if (m.id && jobs[m.id]) { jobs[m.id][1](new Error(m.message)); delete jobs[m.id]; }
            else waiting.splice(0).forEach(function (w) { w[1](new Error(m.message)); });
          }
        };
        worker.onerror = function (e) {
          var err = new Error(e && e.message ? e.message : "the worker stopped");
          waiting.splice(0).forEach(function (w) { w[1](err); });
          Object.keys(jobs).forEach(function (k) { jobs[k][1](err); delete jobs[k]; });
        };
        worker.postMessage({ type: "load" });
      });
    };
    this.run = function (controls, seed) {
      return self.start().then(function () {
        return new Promise(function (resolve, reject) {
          var id = nextId++;
          jobs[id] = [resolve, reject];
          worker.postMessage({ type: "run", id: id, controls: controls, seed: seed });
        });
      });
    };
  }

  root.OTPToy = {
    mount: function (host) {
      host.textContent = "";
      var wrap = C.h("div", { class: "toy" });
      host.appendChild(wrap);
      wrap.appendChild(C.h("h2", { text: "Try to beat it" }));
      wrap.appendChild(C.h("p", { text: "Mix a series and see whether Timestral keeps up with the referee (a forecaster that knows the true equations). Your goal: make Timestral look bad. The score is Timestral's error divided by the referee's: 1 means it kept up, 5 means five times worse." }));
      var ctl = C.h("div", { class: "toy-controls" });
      var modeNote = C.h("p", { class: "otp-note toy-mode", "aria-live": "polite" });
      var plot = C.h("div", { class: "toy-plot" });
      var score = C.h("div", { class: "toy-score", "aria-live": "polite" });
      var foot = C.h("p", { class: "otp-note" });
      wrap.appendChild(ctl); wrap.appendChild(modeNote); wrap.appendChild(score); wrap.appendChild(plot);
      root.OTPData.get("toy").then(function (toy) {
        var by = {};
        toy.items.forEach(function (it) { by[it.key] = it; });
        var st = { season: toy.controls.season[1], wild: "none", strength: toy.controls.strength[1], noise: toy.controls.noise[1], size: "209M",
                   mode: "pre", seed: 1, token: 0 };
        var live = { engine: null, info: null, failed: null, loading: false };
        var sizeRow, drawRow, drawBtn, seedText;
        function key() { return "s" + st.season + "|" + st.wild + "|" + (st.wild === "none" ? 0 : st.strength) + "|n" + st.noise; }
        function row(label, opts, k) {
          var r = C.h("div", { class: "toy-row" }, [C.h("span", { class: "toy-label", text: label })]);
          C.segmented(r, opts, st[k], function (v) { st[k] = v; show(); }, label);
          ctl.appendChild(r);
          return r;
        }
        var modeRow = C.h("div", { class: "toy-row" }, [C.h("span", { class: "toy-label", text: "Forecast made" })]);
        ctl.appendChild(modeRow);
        row("Season strength", toy.controls.season.map(function (v) { return [v, v === 0 ? "none" : String(v)]; }), "season");
        row("Wild piece", toy.controls.wild.map(function (v) { return [v, WILD_NAMES[v] || v]; }), "wild");
        row("Wild strength", toy.controls.strength.map(function (v) { return [v, String(v)]; }), "strength");
        row("Noise", toy.controls.noise.map(function (v) { return [v, String(v)]; }), "noise");
        sizeRow = row("Model", [["5M", "Timestral-5M"], ["209M", "Timestral-209M"]], "size");
        drawRow = C.h("div", { class: "toy-row", hidden: "" }, [C.h("span", { class: "toy-label", text: "Random draw" })]);
        drawBtn = C.h("button", { type: "button", class: "otp-btn", text: "Draw a new series" });
        seedText = C.h("span", { class: "toy-seed" });
        drawBtn.addEventListener("click", function () { st.seed += 1; show(); });
        drawRow.appendChild(drawBtn); drawRow.appendChild(seedText);
        ctl.appendChild(drawRow);

        var liveOk = typeof root.Worker === "function";
        var modeOpts = [["pre", "Precomputed (5M and 209M)"]];
        if (liveOk) modeOpts.push(["live", "Live: Timestral-5M in this browser"]);
        C.segmented(modeRow, modeOpts, st.mode, function (v) { st.mode = v; show(); }, "How the forecast is made");

        function mb(bytes) { return (Math.round(bytes / 1e5) / 10).toFixed(1) + " MB"; }

        function paintScore(it, size, label) {
          var r = it["ratio" + size];
          var best = readBest();
          if (r !== null && (!best || r > best.ratio)) { best = { ratio: r, key: it.key, size: size + (label || "") }; writeBest(best); }
          score.textContent = "";
          score.appendChild(C.h("span", { class: "toy-big", text: r === null ? "–" : C.fmt(r) + "×" }));
          score.appendChild(C.h("span", { text: " Timestral-" + size + "'s error vs the referee's on this series." }));
          if (best) score.appendChild(C.h("span", { class: "toy-best", text: "  Your worst-case so far: " + C.fmt(best.ratio) + "× (" + best.size + ")." }));
          if (it.nonfinite_oracle_paths) score.appendChild(C.h("span", { text: " (" + it.nonfinite_oracle_paths + " referee paths blew up and were dropped.)" }));
        }

        function showPrecomputed() {
          modeNote.textContent = "Showing: precomputed. Every combination was run in Python ahead of time and the page looks it up.";
          foot.textContent = "One fixed random seed per combination; history " + toy.T + " steps (the last " + toy.context_shown + " shown), forecast " + toy.H + " steps. Precomputed by scripts/site_toy.py.";
          sizeRow.hidden = false; drawRow.hidden = true;
          var it = by[key()];
          if (!it) { plot.textContent = "This combination was not computed."; return; }
          root.OTPFan.draw(plot, toy, it, st.size);
          paintScore(it, st.size);
        }

        function showLive() {
          sizeRow.hidden = true; drawRow.hidden = false;
          seedText.textContent = " seed " + st.seed;
          if (live.failed) {
            modeNote.textContent = "Live mode did not start (" + live.failed + "). Showing the precomputed version instead.";
            st.mode = "pre"; showPrecomputed();
            return;
          }
          if (!live.engine) {
            live.engine = new LiveEngine(function (got, total) {
              if (st.mode === "live" && !live.info) modeNote.textContent = "Loading Timestral-5M into this browser: " + mb(got) + (total ? " of " + mb(total) : "") + ".";
            });
          }
          var token = ++st.token;
          if (!live.info) modeNote.textContent = "Loading Timestral-5M into this browser. This downloads its weights once.";
          else modeNote.textContent = "Running live: making a series and forecasting it in this browser…";
          live.engine.run({ season: st.season, wild: st.wild, strength: st.strength, noise: st.noise }, st.seed).then(function (item) {
            if (token !== st.token || st.mode !== "live") return;
            if (!live.info) live.info = true;
            modeNote.textContent = "Showing: live. This series was made in your browser (live/generators.js), forecast by Timestral-5M running in your browser (live/timestral.js), and scored against a referee that simulated " +
              item.live.paths + " continuations. Series " + item.live.ms_series + " ms, forecast " + item.live.ms_model + " ms.";
            foot.textContent = "Live draws use a different random stream than the precomputed ones, so the same controls give a different series. History " + toy.T +
              " steps (the last " + toy.context_shown + " shown), forecast " + toy.H + " steps. The referee's numbers carry Monte Carlo noise from the finite number of paths. The JavaScript model and generators are checked against the Python ones by tests/test_site_parity.py.";
            root.OTPFan.draw(plot, toy, item, "5M");
            paintScore(item, "5M", " live");
          }).catch(function (err) {
            if (token !== st.token) return;
            if (!live.info) { live.failed = err.message; showLive(); return; }
            modeNote.textContent = "This draw failed (" + err.message + "). Try a new one.";
          });
        }

        function show() { if (st.mode === "live") showLive(); else showPrecomputed(); }
        wrap.appendChild(foot);
        show();
        var srch = C.h("div", { class: "toy-search" });
        wrap.appendChild(srch);
        root.OTPData.get("search").then(function (d) {
          if (d.status !== "ready") srch.appendChild(C.h("p", { class: "otp-note", text: "Coming: the worst series our search algorithms found, to compare with yours (fills from results/search/)." }));
          else srch.appendChild(C.h("p", { class: "otp-note", text: "Search results are in: see 'Letting a computer hunt for failures' in the Story." }));
        });
      });
    },
  };
})(window);
