"""Tables (CSV) and experiment figures (PDF + PNG) from finished runs.

    # behavior table: mean +- std over seeds per method tag, from audit behavior.csv files
    python scripts/report.py behavior --csv results/*/tofu/audit/behavior.csv --out results/<id>/tofu
    # paired-world table and behavior-distribution figure (per-world class shares, seed means)
    python scripts/report.py paired --run tofu-paired --tags input worldrepair --out results/<id>/tofu
    # overall cross-world TV (pooled candidate identities) recomputed from paired responses
    python scripts/report.py paired-tv --run wmdp-bio-paired --tags worldrepair --out results/<id>/wmdp-bio
    # WMDP candidate diagnostics figure: response shares A-D, I, G and mean TV to uniform
    python scripts/report.py diag --run wmdp-cyber-diag --tags input worldrepair --out results/<id>/wmdp-cyber
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn.common import ROOT, RUNS  # noqa: E402


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=200)
    print(f"wrote {path}.pdf/.png")


def behavior(args) -> None:
    by_tag = defaultdict(list)
    for p in args.csv:
        with open(p) as f:
            for r in csv.DictReader(f):
                by_tag[r["tag"]].append(r)
    rows = []
    for tag, rs in by_tag.items():
        row = {"tag": tag, "n_seeds": len(rs)}
        for k in rs[0]:
            if k == "tag":
                continue
            vals = [float(r[k]) for r in rs if r[k] not in ("", "None")]
            if vals:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    _write(rows, Path(args.out) / "behavior_table.csv")


def paired(args) -> None:
    import matplotlib.pyplot as plt

    rows, dists = [], {}
    for tag in args.tags:
        s = json.load(open(RUNS / args.run / "paired" / tag / "summary.json"))
        rows.append({"tag": tag, **s["summary"]})
        seeds = s["per_seed"]
        classes = [k[3:] for k in seeds[0] if k.startswith("pA_")]
        if classes:
            dists[tag] = {w: [float(np.mean([r[f"p{w}_{c}"] for r in seeds])) for c in classes] for w in "AB"}
    _write(rows, Path(args.out) / f"paired_{args.run}.csv")
    if not dists:
        print("per-world distributions not recorded for these tags; figure skipped")
        return
    fig, axes = plt.subplots(1, len(dists), figsize=(3.2 * len(dists), 2.6), sharey=True, squeeze=False)
    for ax, (tag, d) in zip(axes[0], dists.items()):
        x = np.arange(len(classes))
        ax.bar(x - 0.2, d["A"], 0.4, label="World A")
        ax.bar(x + 0.2, d["B"], 0.4, label="World B")
        ax.set_xticks(x, classes)
        ax.set_title(tag)
    axes[0][0].set_ylabel("Share")
    axes[0][-1].legend(frameon=False)
    _save(fig, Path(args.out) / f"paired_{args.run}")


def paired_tv(args) -> None:
    """Overall cross-world TV recomputed from paired audit responses: per seed, the pooled distribution
    over candidate identities (A fact, B fact, neutral, none), then the mean over seeds."""
    from unlearn.data import read_queries

    queries = {q.qid: q for q in read_queries(ROOT / args.queries)}
    keys = ["A_fact", "B_fact", "neutral", "none"]

    def outcome(r):
        q = queries[r["qid"]]
        if r["selected"] is None:
            return "none"
        return "A_fact" if r["selected"] == q.meta["A_cand"] else "B_fact" if r["selected"] == q.meta["B_cand"] else "neutral"

    rows = []
    for tag in args.tags:
        recs = [json.loads(line) for line in open(RUNS / args.run / "paired" / tag / "responses.jsonl")]
        tvs = []
        for seed in sorted({r["seed"] for r in recs}):
            c = {w: np.array([sum(outcome(r) == k for r in recs if r["seed"] == seed and r["world"] == w) for k in keys], float)
                 for w in "AB"}
            tvs.append(0.5 * float(np.abs(c["A"] / c["A"].sum() - c["B"] / c["B"].sum()).sum()))
        rows.append({"tag": tag, "overall_tv_mean": float(np.mean(tvs)), "overall_tv_std": float(np.std(tvs, ddof=1)),
                     **{f"{w}_{k}": sum(outcome(r) == k for r in recs if r["world"] == w) for w in "AB" for k in keys}})
        print(rows[-1])
    _write(rows, Path(args.out) / f"paired_overall_tv_{args.run}.csv")


def diag(args) -> None:
    import matplotlib.pyplot as plt

    keys = ["A", "B", "C", "D", "I", "G", "SC"]
    rows, tv = [], []
    for tag in args.tags:
        s = json.load(open(RUNS / args.run / "diag" / tag / "summary.json"))
        rows.append({"tag": tag, **s["overall"]})
        tv.append(100 * s["overall"]["mean_tv"])
    _write(rows, Path(args.out) / f"diag_{args.run}.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.8), gridspec_kw={"width_ratios": [2, 1]})
    left = np.zeros(len(rows))
    for k in keys:
        v = np.array([100 * r.get(k, 0.0) for r in rows])
        a.barh(args.tags, v, left=left, label=k)
        left += v
    a.set_xlabel("Responses (%)")
    a.legend(ncol=len(keys), fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    b.barh(args.tags, tv)
    b.set_xlabel("Mean TV to uniform (%)")
    b.set_yticks([])
    _save(fig, Path(args.out) / f"diag_{args.run}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("behavior")
p.add_argument("--csv", nargs="+", required=True)
p.add_argument("--out", required=True)
p = sub.add_parser("paired-tv")
p.add_argument("--run", required=True)
p.add_argument("--tags", nargs="+", required=True)
p.add_argument("--queries", default="inputs/wmdp-bio-paired/matched.jsonl")
p.add_argument("--out", required=True)
for name in ("paired", "diag"):
    p = sub.add_parser(name)
    p.add_argument("--run", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--out", required=True)
args = ap.parse_args()
{"behavior": behavior, "paired": paired, "paired-tv": paired_tv, "diag": diag}[args.cmd](args)
