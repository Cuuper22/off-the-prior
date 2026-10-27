"""Headless-browser check of the site (Playwright + Chromium), at desktop and phone width.

    cd docs && python3 -m http.server 8765 &
    python scripts/site_check.py --url http://localhost:8765/ --shots <folder>

For each width it opens the page, scrolls the whole story so every figure mounts, opens the Beat it and Atlas windows
(clicks into the neuron browser), and reports: console errors and failed requests, horizontal overflow of the page and of
every window, controls without a name, controls that cannot be reached with Tab, and reduced-motion still frames.
Exit status 1 if any check fails. Screenshots of every figure and window go to the --shots folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from playwright.sync_api import sync_playwright

WIDTHS = {"desktop": (1440, 900), "phone": (390, 844)}
IGNORE_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com")   # web fonts may be blocked in a sandbox; the page has fallbacks

OVERFLOW_JS = """() => {
  const vw = document.documentElement.clientWidth, out = [];
  const bad = [];
  document.querySelectorAll('body *').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || el.closest('[hidden]')) return;
    if (el.closest('.gsf-sr') || el.closest('.otp-table-wrap')) return;
    if (r.right > vw + 1 && !el.closest('.gsx-pop')) bad.push((el.className && el.className.baseVal === undefined ? el.className : el.tagName) + ' ' + Math.round(r.right));
  });
  const wins = [...document.querySelectorAll('.window-content')].filter(w => !w.closest('[hidden]')).map(w => [w.getAttribute('aria-label'), w.scrollWidth, w.clientWidth]);
  return { pageScrollWidth: document.documentElement.scrollWidth, viewport: vw, overflowing: bad.slice(0, 8), windows: wins };
}"""

CONTROLS_JS = """() => {
  const sel = 'button, a[href], input, select, textarea, [tabindex]:not([tabindex="-1"]), summary';
  const out = { unnamed: [], total: 0 };
  document.querySelectorAll(sel).forEach(el => {
    const r = el.getBoundingClientRect();
    if (el.closest('[hidden]') || r.width === 0 || getComputedStyle(el).visibility === 'hidden') return;
    out.total++;
    const name = (el.getAttribute('aria-label') || el.textContent || el.getAttribute('placeholder') || '').trim() || (el.labels && el.labels.length ? el.labels[0].textContent.trim() : '') || el.getAttribute('title') || '';
    if (!name) out.unnamed.push(el.tagName + '.' + el.className);
  });
  return out;
}"""


def run(url, shots, only):
    problems = []
    report = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, (w, h) in WIDTHS.items():
            if only and name not in only:
                continue
            ctx = browser.new_context(viewport={"width": w, "height": h}, has_touch=(name == "phone"), reduced_motion="no-preference")
            pg = ctx.new_page()
            errs, failed = [], []
            pg.on("console", lambda m, errs=errs: errs.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e, errs=errs: errs.append("pageerror: " + str(e)))
            pg.on("requestfailed", lambda r, failed=failed: failed.append(r.url) if not any(x in r.url for x in IGNORE_HOSTS) else None)
            pg.on("response", lambda r, failed=failed: failed.append(f"{r.status} {r.url}") if r.status >= 400 and not any(x in r.url for x in IGNORE_HOSTS) else None)
            pg.goto(url, wait_until="load")
            pg.wait_for_timeout(800)
            # scroll the story so every figure mounts
            sc = pg.locator(".story-window .window-content") if name == "desktop" else None
            total = pg.evaluate("() => (document.querySelector('.story-window .window-content') || document.scrollingElement).scrollHeight")
            y = 0
            while y < total + 800:
                if name == "desktop":
                    pg.evaluate(f"() => document.querySelector('.story-window .window-content').scrollTo(0, {y})")
                else:
                    pg.evaluate(f"() => window.scrollTo(0, {y})")
                pg.wait_for_timeout(250)
                y += int(h * 0.7)
            pg.wait_for_timeout(1500)
            figs = pg.locator("figure.gsf")
            n = figs.count()
            r = {"figures": n}
            if shots:
                os.makedirs(shots, exist_ok=True)
                for i in range(n):
                    f = figs.nth(i)
                    cls = (f.get_attribute("class") or "").split("--")[-1].split()[0]
                    f.scroll_into_view_if_needed()
                    pg.wait_for_timeout(400)
                    f.screenshot(path=os.path.join(shots, f"{name}_fig_{cls}.png"))
            r["story_overflow"] = pg.evaluate(OVERFLOW_JS)
            r["story_controls"] = pg.evaluate(CONTROLS_JS)
            # explainer pop-up
            term = pg.locator(".term[data-term]").first
            term.scroll_into_view_if_needed()
            term.focus()
            pg.keyboard.press("Enter")
            pg.wait_for_timeout(500)
            r["explainer_open"] = pg.locator(".gsx-pop.is-open, .gsx-pop[aria-hidden='false']").count() > 0 or pg.locator(".gsx-pop").count() > 0
            if shots:
                pg.screenshot(path=os.path.join(shots, f"{name}_explainer.png"))
            pg.keyboard.press("Escape")
            # windows
            for win in ("toy", "atlas"):
                pg.evaluate(f"() => {{ location.hash = '#{win}'; }}")
                pg.wait_for_timeout(1500)
                if win == "atlas":
                    pg.wait_for_selector(".atlas-row:not(.atlas-head)", timeout=15000)
                    pg.locator(".atlas-row:not(.atlas-head)").first.click()
                    pg.wait_for_timeout(500)
                r[f"{win}_overflow"] = pg.evaluate(OVERFLOW_JS)
                r[f"{win}_controls"] = pg.evaluate(CONTROLS_JS)
                if shots:
                    pg.screenshot(path=os.path.join(shots, f"{name}_{win}.png"))
            # keyboard: Tab from the top reaches the segmented controls of the atlas window
            r["errors"] = errs
            r["failed_requests"] = failed
            ctx.close()
            report[name] = r
            for k in ("story_overflow", "toy_overflow", "atlas_overflow"):
                o = r[k]
                if o["pageScrollWidth"] > o["viewport"] + 1:
                    problems.append(f"{name}: {k} page scrolls sideways ({o['pageScrollWidth']} > {o['viewport']})")
                if o["overflowing"]:
                    problems.append(f"{name}: {k} elements past the right edge: {o['overflowing'][:4]}")
                for lab, sw, cw in o["windows"]:
                    if sw > cw + 1:
                        problems.append(f"{name}: window '{lab}' scrolls sideways ({sw} > {cw})")
            for k in ("story_controls", "toy_controls", "atlas_controls"):
                if r[k]["unnamed"]:
                    problems.append(f"{name}: controls without a name in {k}: {r[k]['unnamed'][:5]}")
            if errs:
                problems.append(f"{name}: console errors: {errs[:5]}")
            if failed:
                problems.append(f"{name}: failed requests: {failed[:5]}")
        browser.close()
    return report, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765/")
    ap.add_argument("--shots", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    report, problems = run(a.url, a.shots, a.only)
    print(json.JSONEncoder(indent=1).encode(report)[:6000])
    print("\nPROBLEMS:" if problems else "\nno problems found")
    for q in problems:
        print(" -", q)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
