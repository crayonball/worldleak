"""Final behavioral audit of one checkpoint (Full, a baseline, or a WorldRepair model).

Fresh generations (J responses per audit query, new seed) labeled by the final
labeler; reports the class distribution, risk, B, WorldLeak, and the
Theorem 3.2 certificate at the configured tau, plus retain KL from model.ref
(theta_in) on a fixed retain sample as a utility-drift check.

    python scripts/audit.py --config configs/tofu.yaml --model runs/<id>/repair/model \
        --run-id tofu --tag worldrepair
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn import audit, models  # noqa: E402
from unlearn import worldleak as wl  # noqa: E402
from unlearn.common import ROOT, append_jsonl, dump_json, load_config, run_dir, set_seed  # noqa: E402
from unlearn.data import TOFU_ICU, load_benchmark  # noqa: E402

import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--model", required=True, help="HF id or local checkpoint")
ap.add_argument("--run-id", required=True)
ap.add_argument("--tag", required=True, help="method name, e.g. full / npo / worldrepair")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--retain-kl", type=int, default=128, help="retain pairs for KL from model.ref (0 to skip)")
ap.add_argument("--prefix", default="", help="instruction prepended to every prompt (in-context unlearning); {} = target")
ap.add_argument("--icu", action="store_true", help="in-context unlearning with the benchmark's instruction (TOFU)")
args = ap.parse_args()

cfg = load_config(args.config, args.set)
ac = cfg["audit"]
set_seed(ac["seed"])
region = wl.Region.from_config(cfg["region"])
tau = cfg["repair"]["tau"]
out = run_dir(args.run_id, "audit", args.tag)

data = load_benchmark(cfg)
queries = data["audit"]
if args.icu:
    args.prefix = TOFU_ICU
for q in queries:
    if args.prefix:
        q.meta["prefix"] = args.prefix.format((cfg.get("data") or {}).get("target"))
# model.dtype=float32 for checkpoints whose update is below bf16 resolution; the reference uses the same precision
dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
model, tok = models.load(args.model, tokenizer=cfg["model"].get("tokenizer"), dtype=dtype)
labeler = audit.Labeler(cfg["final_labeler"]["name"], device=cfg["final_labeler"].get("device", "cuda"))
res = audit.audit(model, tok, labeler, queries, cfg, region, ac["J"], ac["max_new_tokens"],
                  ac["temperature"], ac["top_p"])
(out / "responses.jsonl").unlink(missing_ok=True)  # a rerun with UNL_RESUME replaces the earlier responses
append_jsonl(res["records"], out / "responses.jsonl")
labeler.close()

retain_kl = None
if args.retain_kl:
    ref, _ = models.load(cfg["model"].get("ref") or cfg["model"]["name"], tokenizer=cfg["model"].get("tokenizer"), dtype=dtype)
    sample = random.Random(0).sample(data["retain"], args.retain_kl)
    kls = []
    with torch.no_grad():
        for i in range(0, len(sample), 8):
            b = sample[i:i + 8]
            pre = args.prefix.format((cfg.get("data") or {}).get("target"))  # in-context unlearning: policy side only
            kls.append(models.retain_kl(model, ref, tok, [models.prompt_of(tok, pre + r.prompt, r.raw) for r in b],
                                        [r.completion for r in b],
                                        ref_prompts=[models.prompt_of(tok, r.prompt, r.raw) for r in b] if pre else None
                                        ).item() * len(b))
    retain_kl = sum(kls) / len(sample)
    del ref
    models.free()

p = res["p"]
eta = wl.worldleak(p, region, tau)
eps, delta = wl.certificate(eta, tau, region)
summary = {
    "model": args.model, "tag": args.tag, "n_queries": res["n_queries"], "n_responses": len(res["records"]),
    "unlabeled": res["unlabeled"], "p": dict(zip(region.classes, p.round(6).tolist())),
    "risk": wl.risk(p, region), "B": wl.admissible_deviation(p, region), "worldleak": eta, "tau": tau,
    "theorem_eps": eps, "theorem_delta": delta, "retain_kl": retain_kl, "label_prompt": audit.LABEL_PROMPT_VERSION,
}
dump_json(summary, out / "summary.json")

results = ROOT / "results" / args.run_id / cfg["benchmark"] / "audit"
results.mkdir(parents=True, exist_ok=True)
row = {"tag": args.tag, **summary["p"], **{k: summary[k] for k in ("risk", "B", "worldleak", "retain_kl", "n_responses", "unlabeled")}}
path = results / "behavior.csv"
new = not path.exists()
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)

print(f"[{args.tag}] p={summary['p']} risk={summary['risk']:.4f} B={summary['B']:.4f} "
      f"WorldLeak={eta:.3f} (eps={eps:.2f}, delta={delta:.3f}) retain_kl={retain_kl}; unlabeled={res['unlabeled']}")
