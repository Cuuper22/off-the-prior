"""Fill docs/src/index.src.html and write docs/index.html, then copy the paper PDF next to it.

Placeholders in the source:
  {{num:KEY}}         a number from paper/numbers.json (run paper/build.py first); a missing key stops the build
  {{site:KEY}}        a number from docs/data/site_numbers.json (run scripts/site_data.py first)
  {{text:NAME|FALLBACK}}  a generated passage from docs/data/site_numbers.json "text" (search, finetune); FALLBACK is used
                      while the results are not in the repository
  {{if:FLAG}}...{{else}}...{{endif}}   keep the first part when site_numbers.json "ready" has FLAG set, else the second

    python scripts/make_site.py

OTP_DOCS=<folder> reads and writes a copy of docs/ instead (preview builds).
"""

from __future__ import annotations

import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.environ.get("OTP_DOCS") or os.path.join(ROOT, "docs")   # OTP_DOCS: a copy of docs/ for a preview build


def main():
    nums = json.load(open(os.path.join(ROOT, "paper", "numbers.json")))
    sn_path = os.path.join(DOCS, "data", "site_numbers.json")
    site = json.load(open(sn_path)) if os.path.exists(sn_path) else {"ready": {}, "numbers": {}, "text": {}}
    src = open(os.path.join(DOCS, "src", "index.src.html")).read()
    missing = []

    def span(v):
        return f'<span class="num" title="{v["src"]}">{v["text"]}</span>'

    def cond(m):
        flag, yes, no = m.group(1), m.group(2), m.group(3) or ""
        return yes if site["ready"].get(flag) else no

    out = re.sub(r"\{\{if:([a-z_]+)\}\}(.*?)(?:\{\{else\}\}(.*?))?\{\{endif\}\}", cond, src, flags=re.S)

    def text(m):
        name, fallback = m.group(1), m.group(2)
        return site["text"].get(name) or fallback

    out = re.sub(r"\{\{text:([a-z_]+)\|([^}]*)\}\}", text, out)

    def sub_num(m):
        k = m.group(1)
        if k not in nums or "text" not in nums[k]:
            missing.append(k)
            return m.group(0)
        return span(nums[k])

    def sub_site(m):
        k = m.group(1)
        if k not in site["numbers"]:
            missing.append("site:" + k)
            return m.group(0)
        return span(site["numbers"][k])

    out = re.sub(r"\{\{num:([A-Za-z0-9_]+)\}\}", sub_num, out)
    out = re.sub(r"\{\{site:([A-Za-z0-9_]+)\}\}", sub_site, out)
    if missing:
        raise SystemExit(f"undefined numbers: {sorted(set(missing))}")
    open(os.path.join(DOCS, "index.html"), "w").write(out)
    pdf = os.path.join(ROOT, "paper", "off_the_prior.pdf")
    if os.path.exists(pdf):
        os.makedirs(os.path.join(DOCS, "paper"), exist_ok=True)
        shutil.copy(pdf, os.path.join(DOCS, "paper", "off_the_prior.pdf"))
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
