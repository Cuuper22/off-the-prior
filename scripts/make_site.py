"""Fill docs/src/index.src.html with numbers from paper/numbers.json and write docs/index.html, then copy the
paper PDF next to it. Every {{num:KEY}} must exist in paper/numbers.json (run paper/build.py first); a missing key
stops the build. Run scripts/site_data.py for the chart data.

    python scripts/make_site.py
"""

from __future__ import annotations

import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    nums = json.load(open(os.path.join(ROOT, "paper", "numbers.json")))
    src = open(os.path.join(ROOT, "docs", "src", "index.src.html")).read()
    missing = []

    def sub(m):
        k = m.group(1)
        if k not in nums or "text" not in nums[k]:
            missing.append(k)
            return m.group(0)
        v = nums[k]
        return f'<span class="num" title="{v["src"]}">{v["text"]}</span>'

    out = re.sub(r"\{\{num:([A-Za-z0-9_]+)\}\}", sub, src)
    if missing:
        raise SystemExit(f"undefined numbers: {sorted(set(missing))}")
    open(os.path.join(ROOT, "docs", "index.html"), "w").write(out)
    pdf = os.path.join(ROOT, "paper", "off_the_prior.pdf")
    if os.path.exists(pdf):
        os.makedirs(os.path.join(ROOT, "docs", "paper"), exist_ok=True)
        shutil.copy(pdf, os.path.join(ROOT, "docs", "paper", "off_the_prior.pdf"))
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
