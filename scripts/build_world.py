"""Construct one protected world by SFT of the shared base checkpoint on that world's facts.

Both worlds start from paired.base_model and share the retained background: each step
mixes a world batch with a retain batch, so only the protected facts differ.

    python scripts/build_world.py --config configs/tofu_paired.yaml --world A --run-id tofu-worlds
"""

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn import models  # noqa: E402
from unlearn.common import append_jsonl, dump_json, load_config, run_dir, set_seed  # noqa: E402
from unlearn.data import load_benchmark  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--world", required=True, choices=["A", "B"])
ap.add_argument("--run-id", required=True)
ap.add_argument("--set", nargs="*", default=[])
args = ap.parse_args()

cfg = load_config(args.config, args.set)
pc, sc = cfg["paired"], cfg["paired"]["world_sft"]
set_seed(cfg["seed"])
rng = random.Random(cfg["seed"])
out = run_dir(args.run_id, f"world_{args.world}")
data = load_benchmark({**cfg, "paired": {**pc, "world": args.world}})
world, retain = data["forget"], data["retain"]

model, tok = models.load(pc["base_model"], trainable=True)
opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=0.01)
steps = sc["steps"]
dump_json({"config": cfg, "world": args.world, "n_world": len(world), "steps": steps}, out / "setup.json")


def nll(batch):
    prompts = [models.format_prompt(tok, p.prompt) for p in batch]
    return -models.mean_token_logprob(model, tok, prompts, [p.completion for p in batch]).mean()


for step in range(steps):
    w_loss = nll(rng.sample(world, min(sc["batch"], len(world))))
    r_loss = nll(rng.sample(retain, sc["retain_batch"]))
    (w_loss + sc["retain_weight"] * r_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    append_jsonl([{"step": step, "world_nll": w_loss.item(), "retain_nll": r_loss.item()}], out / "steps.jsonl")
    if step % 20 == 0:
        print(f"step {step}/{steps} world_nll={w_loss.item():.3f} retain_nll={r_loss.item():.3f}")

model.save_pretrained(out / "model", safe_serialization=True)
tok.save_pretrained(out / "model")
print(f"saved {out / 'model'}")
