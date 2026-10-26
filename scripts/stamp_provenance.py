"""Fill the empty `git` field of provenance files written by runs from code bundles without .git.

Past Colab/laptop runs recorded git = "". For each run we know when it was launched and from which
branch its code bundle was archived (HEAD at launch), so the commit is recovered from the branch
history and written with a note that it was stamped afterwards.

python scripts/stamp_provenance.py --file results/map/tails_big/provenance.json --commit abc1234 --how "..."
python scripts/stamp_provenance.py --infer <provenance.json> --branch master --before 2026-10-26T21:01:21Z
"""
import argparse
import json
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--branch", default=None)
    ap.add_argument("--before", default=None, help="UTC launch time; the last commit on --branch before it")
    ap.add_argument("--how", default="")
    a = ap.parse_args()
    commit = a.commit
    if commit is None:
        commit = subprocess.run(["git", "log", "-1", "--first-parent", "--format=%h", f"--before={a.before}", a.branch],
                                capture_output=True, text=True, check=True).stdout.strip()
    d = json.load(open(a.file))
    if d.get("git"):
        print(f"{a.file}: git already set ({d['git']}); left unchanged")
        return
    d["git"] = commit
    d["git_stamped_after_run"] = a.how or f"HEAD of {a.branch} at launch time {a.before}"
    with open(a.file, "w") as f:
        f.write(json.JSONEncoder(indent=2).encode(d) + "\n")
    print(f"{a.file}: git = {commit}")


if __name__ == "__main__":
    main()
