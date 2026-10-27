/* Atlas window: a browser for every pole of every Timestral size.
   Data: data/neurons/index.json (sizes, label lists, how the files were cut down) and one shard per size and layer,
   data/neurons/<size>/layer_<k>.json, loaded the first time a layer is opened (written by scripts/site_data.py from
   results/neurons/browser/). Names in plain words: data/label_names.json.
   A pole is one sign of one neuron: z = gate * silu(up) can be positive or negative; each sign is its own entry. */
(function (root) {
  "use strict";
  var C = root.OTPChart, h = C.h, NS = "http://www.w3.org/2000/svg";
  var CLASS_NAME = { o: "one job", m: "mixed", u: "unexplained", d: "never fires" };
  var CLASS_ABOUT = {
    o: function (ix) { return "One job: this pole's best single variable explains at least " + pc(ix.theta) + " of what any combination of the library's variables can explain."; },
    m: function (ix) { return "Mixed: its best single variable explains less than " + pc(ix.theta) + " of what combinations of variables explain, so more than one variable matters."; },
    u: function (ix) { return "Unexplained: even combinations of the library's variables explain less than " + pc(ix.floor) + " of its variation."; },
    d: function () { return "It almost never fires on this corpus, so nothing could be scored."; },
  };
  var QUANTITY = { level: "the level of the forecast", slope: "the slope of the forecast", spread: "the spread of the forecast", amp: "the size of the seasonal swing", phase: "the phase of the season" };
  var PAGE = 40;
  var CONTEXT_PATCHES = 32;   // tokens 0..31 are history patches, the rest forecast patches (browser/index.json "token")

  function pc(x) { return Math.round(x * 100) + "%"; }
  function fx(x, d) { return x === null || x === undefined || !isFinite(x) ? "–" : x.toFixed(d === undefined ? 3 : d); }
  function svgEl(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function get(url) { return fetch(root.OTPData.base + url).then(function (r) { if (!r.ok) throw new Error(url + ": " + r.status); return r.json(); }); }

  function sparkline(snip) {
    var W = 150, H = 52, hex = snip[2], pts = [];
    for (var i = 0; i < hex.length; i++) {
      var v = parseInt(hex.charAt(i), 16);
      pts.push((4 + i * (W - 8) / (hex.length - 1)).toFixed(1) + "," + (H - 6 - v * (H - 12) / 15).toFixed(1));
    }
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, class: "atlas-spark", role: "img",
      "aria-label": "Patch of " + hex.length + " values scaled to its own range, from " + snip[3] + " to " + snip[4] });
    svgEl("line", { x1: 4, x2: W - 4, y1: H - 3, y2: H - 3, class: "otp-axis" }, svg);
    svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: "#1c5cab", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    return svg;
  }

  root.OTPAtlas = {
    mount: function (host) {
      host.textContent = "";
      var wrap = h("div", { class: "toy atlas" });
      host.appendChild(wrap);
      wrap.appendChild(h("h2", { text: "Neuron atlas" }));
      wrap.appendChild(h("p", { text: "Every MLP neuron of every Timestral size, split into its two poles (the neuron can fire positive or negative, and each sign is its own entry). Each pole is scored against variables we know exactly because we generated the series: local level, slope, noise, season phase, steps since a switch, which generator made the series, and so on. Pick a size and a layer, filter by label, and open a pole to see the series patches it likes best." }));
      wrap.appendChild(h("p", { class: "otp-note", text: "Read the scores as a statement about these neurons and this variable library together. A pole listed as mixed is not explained by one of these variables; it may be explained by something that is not in the library. The corpus matters as much as the neuron: scored on a different corpus, the same poles often get a different best label (see the story)." }));
      var status = h("p", { class: "otp-note", "aria-live": "polite", text: "Loading the atlas index…" });
      wrap.appendChild(status);
      var body = h("div", { class: "atlas-body" });
      wrap.appendChild(body);
      Promise.all([get("data/neurons/index.json"), get("data/label_names.json")]).then(function (res) {
        status.textContent = "";
        start(body, status, res[0], res[1]);
      }).catch(function () {
        status.textContent = "";
        root.OTPData.get("neurons").then(function (d) { C.pending(body, d); }).catch(function () { C.pending(body, null); });
      });
    },
  };

  function start(body, status, ix, nm) {
    var sizes = Object.keys(ix.sizes);
    var st = { size: sizes[0], layer: "all", cls: "all", sign: "both", q: "", sort: "r2", shown: PAGE, open: null };
    var cache = {};      // size -> { layer -> promise of shard }
    var view = null;     // { size, layer, rows, order } for the current choice

    function name(id) {
      if (id.indexOf("|") >= 0) return id.split("|").map(name).join(" + ");
      var e = nm.names[id];
      return e ? e.name : id;
    }
    function about(id) { var e = nm.names[id]; return e ? e.about : ""; }
    function groupOf(id) {
      for (var k in nm.groups) if (id.indexOf(k) === 0) return nm.groups[k];
      return "";
    }

    // ---- controls
    var bar1 = h("div", { class: "otp-controls" }), bar2 = h("div", { class: "otp-controls atlas-filters" });
    body.appendChild(bar1); body.appendChild(bar2);
    C.segmented(bar1, sizes.map(function (s) { return [s, "Timestral-" + s]; }), st.size, function (v) { st.size = v; st.layer = "all"; fillLayers(); st.open = null; refresh(); }, "model size");
    var layerSel = h("select", { id: "atlas-layer", class: "atlas-select" });
    var layerLab = h("label", { class: "atlas-field", for: "atlas-layer" }, ["Layer ", layerSel]);
    var cls = C.segmented(bar1, [["all", "All poles"], ["o", "One job"], ["m", "Mixed"], ["u", "Unexplained"]], "all", function (v) { st.cls = v; st.shown = PAGE; render(); }, "class of pole");
    bar1.insertBefore(layerLab, cls);
    var signSel = h("select", { id: "atlas-sign", class: "atlas-select" }, [h("option", { value: "both", text: "both signs" }), h("option", { value: "0", text: "positive pole" }), h("option", { value: "1", text: "negative pole" })]);
    var sortSel = h("select", { id: "atlas-sort", class: "atlas-select" }, [h("option", { value: "r2", text: "R² (best first)" }), h("option", { value: "ratio", text: "share of explainable (best first)" }), h("option", { value: "neuron", text: "neuron number" })]);
    var qIn = h("input", { type: "search", id: "atlas-q", class: "atlas-input", list: "atlas-labels", placeholder: "label, e.g. season length", autocomplete: "off" });
    var dl = h("datalist", { id: "atlas-labels" });
    bar2.appendChild(h("label", { class: "atlas-field", for: "atlas-q" }, ["Label ", qIn, dl]));
    bar2.appendChild(h("label", { class: "atlas-field", for: "atlas-sign" }, ["Sign ", signSel]));
    bar2.appendChild(h("label", { class: "atlas-field", for: "atlas-sort" }, ["Sort ", sortSel]));
    var count = h("p", { class: "otp-note atlas-count", "aria-live": "polite" });
    body.appendChild(count);
    var grid = h("div", { class: "atlas-grid" });
    var listBox = h("div", { class: "atlas-listbox" }), detail = h("section", { class: "atlas-detail", "aria-label": "Pole details", tabindex: "-1" });
    grid.appendChild(listBox); grid.appendChild(detail);
    body.appendChild(grid);
    var foot = h("p", { class: "otp-note", text: ix.kept ? "What the page keeps: all poles with their label, R² and class; " + ix.kept + ". Full files: results/neurons/browser/ in the repository." : "" });
    body.appendChild(foot);

    function fillLayers() {
      layerSel.textContent = "";
      layerSel.appendChild(h("option", { value: "all", text: "all layers" }));
      var n = ix.sizes[st.size].layers;
      for (var k = 0; k < n; k++) layerSel.appendChild(h("option", { value: String(k), text: k + (k === 0 ? " (first)" : k === n - 1 ? " (last)" : "") }));
      layerSel.value = st.layer;
      dl.textContent = "";
      var seen = {};
      ix.sizes[st.size].labels.forEach(function (l) {
        if (l.indexOf("|") >= 0 || l === "dead" || seen[l]) return;
        seen[l] = 1;
        dl.appendChild(h("option", { value: name(l) }));
      });
    }
    layerSel.addEventListener("change", function () { st.layer = layerSel.value; st.shown = PAGE; st.open = null; refresh(); });
    signSel.addEventListener("change", function () { st.sign = signSel.value; st.shown = PAGE; render(); });
    sortSel.addEventListener("change", function () { st.sort = sortSel.value; st.shown = PAGE; render(); });
    qIn.addEventListener("input", function () { st.q = qIn.value.trim().toLowerCase(); st.shown = PAGE; render(); });

    function loadLayer(size, k) {
      cache[size] = cache[size] || {};
      if (!cache[size][k]) cache[size][k] = get("data/neurons/" + size + "/layer_" + k + ".json");
      return cache[size][k];
    }
    function refresh() {
      var size = st.size, want = st.layer;
      var layers = want === "all" ? Array.apply(null, { length: ix.sizes[size].layers }).map(function (_, i) { return i; }) : [+want];
      status.textContent = "Loading " + size + (want === "all" ? ", all " + layers.length + " layers" : ", layer " + want) + "…";
      Promise.all(layers.map(function (k) { return loadLayer(size, k); })).then(function (shards) {
        if (size !== st.size || want !== st.layer) return;
        status.textContent = "";
        var rows = [];
        shards.forEach(function (d) {
          for (var i = 0; i < d.n; i++) rows.push({ d: d, i: i, layer: d.layer, id: d.id[i], label: d.label[i], r2: d.r2[i], ratio: d.ratio[i], cls: d.cls.charAt(i) });
        });
        view = { rows: rows };
        render();
      }).catch(function () { status.textContent = "This layer could not be loaded."; });
    }

    function matches() {
      var labels = ix.sizes[st.size].labels, ok = null;
      if (st.q) {
        ok = {};
        labels.forEach(function (l, i) {
          if (l.indexOf("|") >= 0) return;
          if (l.toLowerCase().indexOf(st.q) >= 0 || name(l).toLowerCase().indexOf(st.q) >= 0) ok[i] = 1;
        });
      }
      return view.rows.filter(function (r) {
        if (st.cls !== "all" && r.cls !== st.cls) return false;
        if (st.sign !== "both" && (r.id & 1) !== +st.sign) return false;
        if (ok && !ok[r.label]) return false;
        return true;
      });
    }
    function render() {
      if (!view) return;
      var rows = matches();
      if (st.sort === "r2") rows.sort(function (a, b) { return b.r2 - a.r2 || a.layer - b.layer || a.id - b.id; });
      else if (st.sort === "ratio") rows.sort(function (a, b) { return b.ratio - a.ratio || b.r2 - a.r2; });
      else rows.sort(function (a, b) { return a.layer - b.layer || (a.id >> 1) - (b.id >> 1) || (a.id & 1) - (b.id & 1); });
      var live = view.rows.filter(function (r) { return r.cls !== "d"; }).length;
      count.textContent = rows.length.toLocaleString("en-US") + " poles match (of " + view.rows.length.toLocaleString("en-US") + (live < view.rows.length ? ", " + live.toLocaleString("en-US") + " fire at all" : "") + "). Showing the first " + Math.min(st.shown, rows.length) + ".";
      listBox.textContent = "";
      if (!rows.length) { listBox.appendChild(h("p", { class: "otp-empty", text: "No pole matches. Loosen a filter." })); return; }
      var head = h("div", { class: "atlas-row atlas-head", "aria-hidden": "true" }, [h("span", { text: "pole" }), h("span", { text: "best label" }), h("span", { class: "atlas-num", text: "R²" }), h("span", { class: "atlas-num atlas-hide", text: "share" }), h("span", { text: "class" })]);
      listBox.appendChild(head);
      var ul = h("ul", { class: "atlas-list" });
      rows.slice(0, st.shown).forEach(function (r) {
        var labelId = ix.sizes[st.size].labels[r.label];
        var b = h("button", { type: "button", class: "atlas-row", "aria-pressed": String(st.open && st.open.d === r.d && st.open.i === r.i) }, [
          h("span", { class: "atlas-id", text: "L" + r.layer + "." + (r.id >> 1) + ((r.id & 1) ? "−" : "+") }),
          h("span", { class: "atlas-lab", text: name(labelId) }),
          h("span", { class: "atlas-num", text: r.r2 < 0 ? "–" : fx(r.r2 / 10000) }),
          h("span", { class: "atlas-num atlas-hide", text: r.ratio < 0 ? "–" : pc(r.ratio / 1000) }),
          h("span", { class: "atlas-cls atlas-cls-" + r.cls, text: CLASS_NAME[r.cls] }),
        ]);
        b.setAttribute("aria-label", "Layer " + r.layer + " neuron " + (r.id >> 1) + (r.id & 1 ? " negative" : " positive") + " pole, best label " + name(labelId) + ", " + CLASS_NAME[r.cls]);
        b.addEventListener("click", function () { open(r, true, b); });
        ul.appendChild(h("li", {}, [b]));
      });
      listBox.appendChild(ul);
      if (rows.length > st.shown) {
        var more = h("button", { type: "button", class: "otp-seg-btn atlas-more", text: "Show " + Math.min(PAGE, rows.length - st.shown) + " more" });
        more.addEventListener("click", function () { st.shown += PAGE; render(); });
        listBox.appendChild(more);
      }
    }

    function open(r, focus, btn) {
      st.open = r;
      listBox.querySelectorAll(".atlas-row[aria-pressed]").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
      var labelId = ix.sizes[st.size].labels[r.label];
      var d = r.d.detail[String(r.i)];
      detail.textContent = "";
      detail.classList.add("is-open");
      var back = h("button", { type: "button", class: "otp-seg-btn atlas-back", text: "Back to the list" });
      back.addEventListener("click", function () { detail.classList.remove("is-open"); var cur = listBox.querySelector("button.atlas-row"); if (cur) cur.focus(); });
      detail.appendChild(back);
      var title = h("h3", { text: "Timestral-" + st.size + ", layer " + r.layer + ", neuron " + (r.id >> 1) + ", " + ((r.id & 1) ? "negative" : "positive") + " pole", tabindex: "-1" });
      detail.appendChild(title);
      detail.appendChild(h("p", {}, [h("span", { class: "atlas-cls atlas-cls-" + r.cls, text: CLASS_NAME[r.cls] }), " " + CLASS_ABOUT[r.cls](ix)]));
      if (r.cls === "d") {
        detail.appendChild(h("p", { text: "Nothing more to show: " + about("dead") }));
      } else {
        var dl2 = h("dl", { class: "atlas-facts" });
        function fact(k, v) { dl2.appendChild(h("dt", { text: k })); dl2.appendChild(h("dd", { text: v })); }
        fact("Best label", name(labelId) + " (" + groupOf(labelId) + ")");
        if (about(labelId)) fact("What that is", about(labelId));
        fact("R² of the label alone", r.r2 < 0 ? "–" : fx(r.r2 / 10000) + " (cross-validated, by series)");
        if (d) {
          fact("Best any combination of library variables", fx(d[0]));
          fact("Share of that this pole gets", r.ratio < 0 ? "–" : pc(r.ratio / 1000));
          fact("R² on held-out series", fx(d[1]));
          if (d[2]) fact("Direction", "fires more when " + name(labelId) + " is " + (d[2] > 0 ? "higher" : "lower"));
          if (d[3] !== null && d[3] !== undefined) fact("Best pair of variables", name(ix.sizes[st.size].labels[d[3]]));
          if (d[4]) fact("Gate routes", "share of its activation through the two sign routes of the gate: " + d[4].map(function (x) { return fx(x, 2); }).join(" and "));
          if (d[5] !== null && d[5] !== undefined) fact("On real series", "the label's fit from made-up series correlates " + fx(d[5], 2) + " with this pole's activity on real windows");
        }
        detail.appendChild(dl2);
        if (d && d[6]) {
          var q = d[6];
          detail.appendChild(h("p", { class: "atlas-steer", text: "Steering test: we wrote down that turning this pole up or down would change " + (QUANTITY[q[0]] || q[0]) + ", then scaled it during the forecast. Did the forecast move as predicted? " + (q[1] ? "Yes." : "No.") + " The test could only detect effects larger than its pre-set thresholds, and the measured moves were smaller than that, so a no here does not show the label is wrong." }));
        }
        if (d) {
          var ex = h("div", { class: "atlas-ex" });
          [[d[7], "Strongest on made-up series"], [d[8], "Mid-strength"], [d[9], "Strongest on real series"]].forEach(function (grp) {
            if (!grp[0] || !grp[0].length) return;
            var sec = h("div", { class: "atlas-exsec" }, [h("h4", { text: grp[1] })]);
            var row = h("div", { class: "atlas-exrow" });
            grp[0].forEach(function (s) {
              var fig = h("figure", { class: "atlas-fig" });
              fig.appendChild(sparkline(s));
              var where = s[0] < CONTEXT_PATCHES ? "history patch " + (s[0] + 1) : "forecast patch " + (s[0] - CONTEXT_PATCHES + 1);
              fig.appendChild(h("figcaption", { text: where + ", pole value " + fx(s[1], 2) + "; patch range " + s[3] + " to " + s[4] }));
              row.appendChild(fig);
            });
            sec.appendChild(row);
            ex.appendChild(sec);
          });
          detail.appendChild(ex);
          detail.appendChild(h("p", { class: "otp-note", text: "Each small plot is one 32-step patch of a series, drawn from its own lowest to its own highest value." }));
        } else {
          detail.appendChild(h("p", { class: "otp-note", text: "Exemplars are kept only for one-job poles, steered poles and the best-scoring poles of each layer, to keep the page small. This pole's score is shown above." }));
        }
      }
      if (focus) {
        title.focus({ preventScroll: true });
        if (detail.scrollIntoView) detail.scrollIntoView({ block: "nearest", behavior: "auto" });
      }
      listBox.querySelectorAll(".atlas-row[aria-pressed]").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
      if (btn) btn.setAttribute("aria-pressed", "true");
    }

    fillLayers();
    detail.appendChild(h("p", { class: "otp-note", text: "Open a pole from the list to see its details." }));
    refresh();
  }
})(window);
