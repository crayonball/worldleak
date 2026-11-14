"""Parameter-update baselines on one RWKU target, ported from RWKU's LLaMA-Factory scripts (its pinned stack does not
run on current torch / transformers):

  npo  scripts/full/run_npo.sh, train/npo/trainer.py: synthetic forget corpus, beta 0.2, batch 16.
  rt   scripts/full/run_rt.sh (rejection tuning): SFT on the target's refusal QA (train_refusal_phi3), loss on the
       answer tokens of the chat-formatted pair, batch 8.

Both: cutoff 512, AdamW, 20-step warmup, cosine schedule, 3 epochs. LR as in PURGE Table 4 for Phi-3
(NPO 2e-6, RT 4e-7).

    python scripts/rwku_train.py --method npo --target "Stephen King" --run-id <id>
    python scripts/rwku_train.py --method rt --target "Stephen King" --run-id <id>
"""

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn import models  # noqa: E402
from unlearn.common import RUNS, set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["npo", "rt"], default="npo")
ap.add_argument("--target", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
ap.add_argument("--lr", type=float, default=None, help="default: NPO 2e-6, RT 4e-7")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--beta", type=float, default=0.2)
ap.add_argument("--batch", type=int, default=None, help="default: NPO 16, RT 8")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dtype", default="float32", help="weight precision during training")
ap.add_argument("--save-dtype", default="bfloat16", help="float32 keeps small updates that bf16 rounds away")
ap.add_argument("--tag", default=None, help="default: the method name")
args = ap.parse_args()
args.lr = args.lr or {"npo": 2e-6, "rt": 4e-7}[args.method]
args.batch = args.batch or {"npo": 16, "rt": 8}[args.method]
args.tag = args.tag or args.method

set_seed(args.seed)
slug = args.target.replace(" ", "_").replace(",", "").replace(".", "")
out = RUNS / args.run_id / args.tag / slug
model, tok = models.load(args.model, trainable=True, dtype=getattr(torch, args.dtype))
ref = models.load(args.model)[0] if args.method == "npo" else None
tok.padding_side = "right"
if args.method == "npo":
    ds = load_dataset("jinzhuoran/RWKU", "train_positive_phi3", split="train")
    texts = [(None, r["text"] + tok.eos_token) for r in ds if r["subject"] == args.target]
else:
    ds = load_dataset("jinzhuoran/RWKU", "train_refusal_phi3", split="train")
    texts = []
    for r in ds:
        if r["subject"] != args.target:
            continue
        q = (r["instruction"] + ("\n" + r["input"] if r["input"] else "")).strip()
        prompt = models.format_prompt(tok, q)
        texts.append((prompt, r["output"].strip() + tok.eos_token))
print(f"{args.method}: {len(texts)} training texts for {args.target}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
total = math.ceil(len(texts) / args.batch) * args.epochs
sched = get_cosine_schedule_with_warmup(opt, 20, total)


def batch_nll(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().transpose(-1, -2), labels[:, 1:], ignore_index=-100, reduction="none").sum(-1)


rng = random.Random(args.seed)
for ep in range(args.epochs):
    rng.shuffle(texts)
    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i + args.batch]
        enc = tok([(p or "") + c for p, c in chunk], return_tensors="pt", padding=True, truncation=True, max_length=512,
                  add_special_tokens=False).to(model.device)
        labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
        if args.method == "npo":
            cur = batch_nll(model(**enc).logits, labels)
            with torch.no_grad():
                base = batch_nll(ref(**enc).logits, labels)
            loss = -F.logsigmoid(args.beta * (cur - base)).mean() * 2 / args.beta
        else:  # SFT on the refusal only: prompt tokens are masked
            for j, (p, _) in enumerate(chunk):
                labels[j, :len(tok(p, add_special_tokens=False).input_ids)] = -100
            loss = batch_nll(model(**enc).logits, labels).sum() / (labels[:, 1:] != -100).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    print(f"epoch {ep} loss {loss.item():.3f}", flush=True)
model.to(getattr(torch, args.save_dtype)).save_pretrained(out)
tok.save_pretrained(out)
print("saved", out, flush=True)
