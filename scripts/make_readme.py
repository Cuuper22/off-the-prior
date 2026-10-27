"""Fill README.src.md placeholders ({{num:KEY}}) from paper/numbers.json (built by paper/build.py).

python paper/build.py && python scripts/make_readme.py
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    nums = json.load(open(os.path.join(ROOT, "paper", "numbers.json")))
    src = open(os.path.join(ROOT, "README.src.md")).read()
    missing = []

    def sub(m):
        k = m.group(1)
        if k not in nums or "text" not in nums[k]:
            missing.append(k)
            return m.group(0)
        return nums[k]["text"]

    out = re.sub(r"\{\{num:([A-Za-z0-9_]+)\}\}", sub, src)
    if missing:
        sys.exit(f"undefined: {sorted(set(missing))}")
    open(os.path.join(ROOT, "README.md"), "w").write(out)
    print("wrote README.md")


if __name__ == "__main__":
    main()
