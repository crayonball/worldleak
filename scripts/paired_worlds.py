"""Paired-world audit (Section 5.2, Appendix A.4 / A.5.3).

Two repaired models for worlds A and B are audited on the same matched
candidate queries. Per seed: cross-world TV of the string-matched selection
distributions (max over queries), leakage C+I (max over worlds), safe
nonselection counts, retain KL from each world's input model, and the
KL-Chernoff upper bound on delta at eps=0 (SC-only region) plus the
two-component certificate.

Matched queries come from <paired.dir>/matched.jsonl (built by scripts/build_inputs.py),
filtered to paired.subject; each world's labeler reference is meta.reference_<world>.

    python scripts/paired_worlds.py --config configs/tofu_paired.yaml \
        --world-a runs/<a>/repair/model --world-b runs/<b>/repair/model \
        --input-a <ckpt_a> --input-b <ckpt_b> --run-id tofu-paired --tag worldrepair
"""

import argparse
import copy
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn import audit, models  # noqa: E402
from unlearn import worldleak as wl  # noqa: E402
from unlearn.common import ROOT, append_jsonl, dump_json, load_config, run_dir, set_seed  # noqa: E402
from unlearn.data import load_benchmark, paired_world  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--world-a", required=True)
ap.add_argument("--world-b", required=True)
ap.add_argument("--input-a", required=True, help="world A model before repair (retain KL reference)")
ap.add_argument("--input-b", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(args.config, args.set)
pc = cfg["paired"]
region = wl.Region.from_config(cfg["region"])
out = run_dir(args.run_id, "paired", args.tag)
retain = load_benchmark({**cfg, "paired": {**pc, "world": "A"}})["retain"]
per_world_queries = {w: paired_world(cfg, w, retain)["audit"] for w in ("A", "B")}
labeler = audit.Labeler(cfg["final_labeler"]["name"], device=cfg["final_labeler"].get("device", "cuda"))
idx = region.classes.index
leak_cls = [c for c in ("C", "I") if c in region.classes]


def world_queries(w: str):
    return copy.deepcopy(per_world_queries[w])


def seed_queries(qs, seed: int):
    """Per-seed audit subset: paired.queries_per_seed queries, stratified by attribute (same subset in both worlds)."""
    n = pc.get("queries_per_seed")
    if not n:
        return qs
    by_attr = {}
    for q in sorted(qs, key=lambda q: q.qid):
        by_attr.setdefault(q.meta.get("attribute"), []).append(q)
    rng = random.Random(seed)
    k = n // len(by_attr)
    keep = {q.qid for group in by_attr.values() for q in rng.sample(group, min(k, len(group)))}
    return [q for q in qs if q.qid in keep]


def outcome_identity(q, selected) -> str:
    """Candidate identity aligned across worlds: world-A fact, world-B fact, neutral option, or none."""
    if selected is None:
        return "none"
    return "A_fact" if selected == q.meta["A_cand"] else "B_fact" if selected == q.meta["B_cand"] else "neutral"


def selection_counts(qs, recs) -> dict:
    """Selection counts per TV group: per matched query (default), pooled per attribute by option text
    (paired.tv_group=attribute), or pooled over all queries by candidate identity (paired.tv_group=overall);
    nonselection is its own outcome."""
    if pc.get("tv_group") == "overall":
        c = {}
        for q in qs:
            for r in recs[q.qid]:
                o = outcome_identity(q, r["selected"])
                c[o] = c.get(o, 0) + 1
        return {"overall": c}
    groups = {}
    for q in qs:
        key = q.meta["attribute"] if pc.get("tv_group") == "attribute" else q.block
        c = groups.setdefault(key, {})
        for r in recs[q.qid]:
            o = None if r["selected"] is None else q.candidates[r["selected"]]
            c[o] = c.get(o, 0) + 1
    return groups


def group_tv(a: dict, b: dict) -> float:
    keys = sorted(set(a) | set(b), key=str)
    pa = np.array([a.get(k, 0) for k in keys], float)
    pb = np.array([b.get(k, 0) for k in keys], float)
    return wl.tv(pa / pa.sum(), pb / pb.sum())


rows = []
for seed in pc["seeds"]:
    per_world = {}
    for w, path, inp in (("A", args.world_a, args.input_a), ("B", args.world_b, args.input_b)):
        set_seed(seed)
        qs = seed_queries(world_queries(w), seed)
        model, tok = models.load(path)
        res = audit.audit(model, tok, labeler, qs, cfg, region, pc["n_per_query"], pc["max_new_tokens"],
                          pc["temperature"], pc["top_p"])
        append_jsonl([{"seed": seed, "world": w, **r} for r in res["records"]], out / "responses.jsonl")
        sample = random.Random(seed).sample(retain, pc["retain_prompts"])
        ref, _ = models.load(inp)
        with torch.no_grad():
            kl = models.retain_kl(model, ref, tok, [models.format_prompt(tok, r.prompt) for r in sample],
                                  [r.completion for r in sample]).item()
        del model, ref
        models.free()
        recs = {q.qid: [r for r in res["records"] if r["qid"] == q.qid] for q in qs}
        per_world[w] = {
            "p": res["p"], "kl": kl, "n": res["n_queries"],
            "sel": selection_counts(qs, recs),
            "safe_nonselection": sum(r["selected"] is None and r["label"] == "SC" for r in res["records"]),
            "n_responses": len(res["records"]),
        }
    A, B = per_world["A"], per_world["B"]
    tvs = [group_tv(A["sel"][g], B["sel"][g]) for g in A["sel"]]
    leak = {w: float(sum(per_world[w]["p"][idx(c)] for c in leak_cls)) for w in per_world}
    r_a, r_b = wl.risk(A["p"], region), wl.risk(B["p"], region)
    row = {
        "seed": seed, "tv_max": max(tvs), "tv_mean": float(np.mean(tvs)), "leak_max": max(leak.values()),
        "risk_A": r_a, "risk_B": r_b, "label_tv": wl.tv(A["p"], B["p"]),
        "safe_nonselection": A["safe_nonselection"] + B["safe_nonselection"],
        "n_responses_pair": A["n_responses"] + B["n_responses"], "retain_kl_max": max(A["kl"], B["kl"]),
        "two_component_eps": wl.two_component_certificate(A["p"], B["p"], region)[0],
        **{f"pA_{c}": float(A["p"][idx(c)]) for c in region.classes},
        **{f"pB_{c}": float(B["p"][idx(c)]) for c in region.classes},
    }
    if region.admissible == ("SC",):
        row["delta_upper_eps0"] = wl.paired_delta_upper(r_a, r_b, min(A["n"], B["n"]), pc.get("alpha", 0.05))
    print(row)
    rows.append(row)

summary = {
    "tag": args.tag, "seeds": pc["seeds"], "tv_max_mean": float(np.mean([r["tv_max"] for r in rows])),
    "leak_max_mean": float(np.mean([r["leak_max"] for r in rows])),
    "safe_nonselection_pooled": f"{sum(r['safe_nonselection'] for r in rows)}/{sum(r['n_responses_pair'] for r in rows)}",
    "retain_kl_max_mean": float(np.mean([r["retain_kl_max"] for r in rows])),
}
dump_json({"summary": summary, "per_seed": rows}, out / "summary.json")
results = ROOT / "results" / args.run_id / cfg["benchmark"] / "paired"
results.mkdir(parents=True, exist_ok=True)
with open(results / f"{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(summary)
