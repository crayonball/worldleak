"""Open-ended responses in paired worlds (tab:tofu-hsiao-open-ended and related tables).

Matched questions are asked without options (one per attribute, or the given attributes),
greedy decoding, for each method's world-A and world-B model. These are illustrations only;
TV values come from scripts/paired_worlds.py.

    python scripts/open_ended.py --config configs/tofu_paired.yaml --run-id tofu-paired \
        --method input=<ckptA>,<ckptB> --method worldrepair=<repA>,<repB> --attributes city year_decade
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn import models  # noqa: E402
from unlearn.common import ROOT, append_jsonl, load_config, run_dir  # noqa: E402
from unlearn.data import load_benchmark, paired_world  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--method", action="append", required=True, help="name=<world A model>,<world B model>")
ap.add_argument("--attributes", nargs="*", default=None, help="default: every attribute")
ap.add_argument("--max-new-tokens", type=int, default=128)
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(args.config, args.set)
pc = cfg["paired"]
retain = load_benchmark({**cfg, "paired": {**pc, "world": "A"}})["retain"]
queries = paired_world(cfg, "A", retain)["audit"]
first = {}
for q in sorted(queries, key=lambda q: q.qid):
    a = q.meta.get("attribute")
    if (args.attributes is None or a in args.attributes) and a not in first:
        first[a] = q
questions = list(first.values())
out = run_dir(args.run_id, "open_ended", pc.get("subject", "all").replace(" ", "_"))

rows = []
for spec in args.method:
    name, paths = spec.split("=", 1)
    for w, path in zip("AB", paths.split(",")):
        model, tok = models.load(path)
        outs = models.generate(model, tok, [models.format_prompt(tok, q.prompt) for q in questions], 1,
                               args.max_new_tokens, do_sample=False)
        del model
        models.free()
        for q, o in zip(questions, outs):
            rows.append({"method": name, "world": w, "attribute": q.meta.get("attribute"), "question": q.prompt,
                         "reference": q.meta[f"reference_{w}"], "response": o[0].strip()})
append_jsonl(rows, out / "responses.jsonl")
results = ROOT / "results" / args.run_id / cfg["benchmark"] / "open_ended"
results.mkdir(parents=True, exist_ok=True)
with open(results / f"{pc.get('subject', 'all').replace(' ', '_')}.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
for r in rows:
    print(f"[{r['method']} / {r['world']} / {r['attribute']}] {r['question']}\n  ref={r['reference']}\n  {r['response'][:200]}")
