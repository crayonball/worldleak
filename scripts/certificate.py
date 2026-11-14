"""Finite-sample paired-world certificate at eps=0 (Appendix A.4, tab:finite-sample-wmdp).

For each query budget n, an independent audit draws n matched queries per world from the
shared query law (stratified by attribute, fresh seed per row), labels one response per
query, and reports the risk in each world, the empirical label TV, and the KL-Chernoff
upper bound delta_bar(0) = max(r_bar_A, r_bar_B) with alpha split over the two worlds.

    python scripts/certificate.py --config configs/wmdp_bio_paired.yaml \
        --world-a runs/<a>/repair/model --world-b runs/<b>/repair/model \
        --run-id wmdp-bio-paired --tag worldrepair --n 64 128 256
"""

import argparse
import copy
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn import audit, models  # noqa: E402
from unlearn import worldleak as wl  # noqa: E402
from unlearn.common import ROOT, append_jsonl, dump_json, load_config, run_dir, set_seed  # noqa: E402
from unlearn.data import load_benchmark, paired_world  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--world-a", required=True)
ap.add_argument("--world-b", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, nargs="+", default=[64, 128, 256])
ap.add_argument("--seed", type=int, default=1000, help="row i uses seed + i")
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(args.config, args.set)
pc = cfg["paired"]
alpha = pc.get("alpha", 0.05)
region = wl.Region.from_config(cfg["region"])
out = run_dir(args.run_id, "certificate", args.tag)
retain = load_benchmark({**cfg, "paired": {**pc, "world": "A"}})["retain"]
pool = {w: paired_world(cfg, w, retain)["audit"] for w in ("A", "B")}


def draw(n: int, seed: int) -> set[str]:
    """n query ids stratified by attribute; the same ids are audited in both worlds."""
    by_attr = {}
    for q in sorted(pool["A"], key=lambda q: q.qid):
        by_attr.setdefault(q.meta.get("attribute"), []).append(q.qid)
    rng = random.Random(seed)
    k = n // len(by_attr)
    return {i for ids in by_attr.values() for i in rng.sample(ids, min(k, len(ids)))}


labeler = audit.Labeler(cfg["final_labeler"]["name"], device=cfg["final_labeler"].get("device", "cuda"))
rows = []
for i, n in enumerate(args.n):
    seed = args.seed + i
    ids = draw(n, seed)
    p = {}
    for w, path in (("A", args.world_a), ("B", args.world_b)):
        set_seed(seed)
        qs = [copy.deepcopy(q) for q in pool[w] if q.qid in ids]
        model, tok = models.load(path)
        res = audit.audit(model, tok, labeler, qs, cfg, region, 1, pc["max_new_tokens"], pc["temperature"], pc["top_p"])
        append_jsonl([{"n": n, "seed": seed, "world": w, **r} for r in res["records"]], out / "responses.jsonl")
        del model
        models.free()
        p[w] = res["p"]
    r_a, r_b = wl.risk(p["A"], region), wl.risk(p["B"], region)
    n_eff = len(ids)
    row = {"n": n_eff, "seed": seed, "risk_A": r_a, "risk_B": r_b, "label_tv": wl.tv(p["A"], p["B"]),
           "delta_upper_eps0": wl.paired_delta_upper(r_a, r_b, n_eff, alpha)}
    print(row, flush=True)
    rows.append(row)

dump_json({"tag": args.tag, "alpha": alpha, "rows": rows}, out / "summary.json")
results = ROOT / "results" / args.run_id / cfg["benchmark"] / "certificate"
results.mkdir(parents=True, exist_ok=True)
with open(results / f"{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
