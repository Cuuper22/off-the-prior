/* Web worker for the live toy: loads the Timestral-5M weights once, then makes a series, runs the referee and the
   model, and sends back one item shaped like an entry of data/toy.json (so the same chart code draws it).
   Messages in:  { type: "load" }                          fetch weights + generator spec (relative to this file)
                 { type: "run", id, controls, seed }       one series
   Messages out: { type: "progress", got, total } | { type: "ready", info } | { type: "result", id, item } | { type: "error", message } */
/* global importScripts, OTPTimestral, OTPLiveGen */
importScripts("timestral.js", "generators.js");

var model = null, spec = null;
var QI = [2, 10, 18];            // quantile indices for 0.1, 0.5, 0.9
var SHOWN = 256;                 // context steps shown in the plot, as in data/toy.json
var N_PATHS = 200;               // referee continuations

function round4(x) { return Math.round(x * 1e4) / 1e4; }
function cols(q, H) {
  return QI.map(function (j) {
    var a = [];
    for (var t = 0; t < H; t++) a.push(round4(q[t * 21 + j]));
    return a;
  });
}

function fetchBytes(url, onProgress) {
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error(url + ": " + r.status);
    var total = +r.headers.get("content-length") || 0;
    if (!r.body || !r.body.getReader) return r.arrayBuffer();
    var reader = r.body.getReader(), chunks = [], got = 0;
    function pump() {
      return reader.read().then(function (s) {
        if (s.done) {
          var out = new Uint8Array(got), o = 0;
          chunks.forEach(function (c) { out.set(c, o); o += c.length; });
          return out.buffer;
        }
        chunks.push(s.value); got += s.value.length;
        onProgress(got, total);
        return pump();
      });
    }
    return pump();
  });
}

function load() {
  return fetch("timestral5m.json").then(function (r) { if (!r.ok) throw new Error("timestral5m.json: " + r.status); return r.json(); })
    .then(function (table) {
      return Promise.all([
        fetchBytes(table.file, function (got, total) { postMessage({ type: "progress", got: got, total: total }); }),
        fetch("live_spec.json").then(function (r) { if (!r.ok) throw new Error("live_spec.json: " + r.status); return r.json(); }),
      ]).then(function (res) {
        model = OTPTimestral.load(table, res[0]);
        spec = res[1];
        postMessage({ type: "ready", info: { name: table.model, dtype: table.dtype, elements: table.elements, bytes: res[0].byteLength,
                                             T: spec.T, H: spec.H, nPaths: N_PATHS } });
      });
    });
}

function run(msg) {
  var t0 = Date.now();
  var s = OTPLiveGen.makeSample(spec, msg.controls, msg.seed, { nPaths: N_PATHS });
  var t1 = Date.now();
  var f = OTPTimestral.forecast(model, s.context, s.H);
  var t2 = Date.now();
  var n = s.context.length, shown = Array.prototype.slice.call(s.context, n - SHOWN).map(round4);
  var mTs = OTPLiveGen.mase(s.context, s.future, f.q, s.period);
  var mOr = OTPLiveGen.mase(s.context, s.future, s.oracle, s.period);
  var item = {
    key: "live#" + msg.seed, period: s.period, context: shown,
    future: Array.prototype.slice.call(s.future).map(round4),
    oracle: cols(s.oracle, s.H), oracle_mase: round4(mOr),
    nonfinite_oracle_paths: s.nonfiniteOracle,
    q5M: cols(f.q, s.H), mase5M: round4(mTs),
    ratio5M: isFinite(mTs) && isFinite(mOr) && mOr > 0 ? round4(mTs / mOr) : null,
    live: { seed: msg.seed, ms_series: t1 - t0, ms_model: t2 - t1, paths: N_PATHS },
  };
  postMessage({ type: "result", id: msg.id, item: item });
}

onmessage = function (e) {
  var m = e.data;
  try {
    if (m.type === "load") load().catch(function (err) { postMessage({ type: "error", message: String(err && err.message || err) }); });
    else if (m.type === "run") run(m);
  } catch (err) {
    postMessage({ type: "error", id: m.id, message: String(err && err.message || err) });
  }
};
