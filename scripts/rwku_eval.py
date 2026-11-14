"""RWKU evaluation of one checkpoint for one target (ported from RWKU's eval_forget / eval_neighbor / eval_mia / eval_mmlu).

ROUGE-L recall on Forget FB/QA/AA and Neighbor FB/QA (greedy, 30 new tokens, first line), per-token NLL and
Min-K%++ (20%) on FM and RM, 5-shot MMLU on a fixed subset of RWKU's utility_general. RWKU's own `loss` field stores log-likelihood;
here NLL = -log-likelihood, so higher FM NLL means the model gives the forget text lower probability.

    python scripts/rwku_eval.py --model <ckpt> --target "Stephen King" --run-id <id> --tag npo [--icu] [--skip-mmlu]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rouge import Rouge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn import models  # noqa: E402
from unlearn.common import ROOT, RUNS, append_jsonl  # noqa: E402
from unlearn.data import RWKU_ICU, rwku_probes, rwku_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", default=None, help="fallback tokenizer for checkpoints without tokenizer files")
ap.add_argument("--dtype", default="bfloat16", help="float32 for checkpoints whose update is below bf16 resolution")
ap.add_argument("--target", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--icu", action="store_true", help="prepend the in-context unlearning instruction")
ap.add_argument("--skip-mmlu", action="store_true")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--mmlu-n", type=int, default=2000, help="fixed question subset, the same for all methods (RWKU's split has 34200)")
args = ap.parse_args()

model, tok = models.load(args.model, tokenizer=args.tokenizer, dtype=getattr(torch, args.dtype))
model.eval()
prefix = RWKU_ICU.format(args.target) if args.icu else ""
slug = args.target.replace(" ", "_").replace(",", "").replace(".", "")
(RUNS / args.run_id / "rwku_eval").mkdir(parents=True, exist_ok=True)


def chat(text: str, answer: bool) -> str:
    return models.format_prompt(tok, prefix + text) + ("Answer:" if answer else "")


def recall(config: str) -> float:
    rows = rwku_probes(config, args.target)
    prompts = [chat(*rwku_prompt(r)) for r in rows]
    outs = models.generate(model, tok, prompts, 1, 30, batch_size=args.batch, do_sample=False)
    preds = [(o[0].strip().split("\n")[0].strip() or "NOANSWER") for o in outs]
    append_jsonl([{"config": config, "query": r["query"], "answer": r["answer"], "prediction": p} for r, p in zip(rows, preds)],
                 RUNS / args.run_id / "rwku_eval" / f"{args.tag}_{slug}.jsonl")
    return Rouge().get_scores(hyps=preds, refs=[r["answer"] for r in rows], avg=True)["rouge-l"]["r"]


@torch.no_grad()
def mia(config: str) -> tuple[float, float]:
    nll, minkpp = [], []
    for r in rwku_probes(config, args.target):
        ids = tok(prefix + r["text"], return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
        logits = model(ids).logits[0, :-1].float()
        lp = F.log_softmax(logits, -1)
        tok_lp = lp.gather(-1, ids[0, 1:, None]).squeeze(-1)
        mu = (lp.exp() * lp).sum(-1)
        sigma = (lp.exp() * lp.square()).sum(-1) - mu.square()
        z = ((tok_lp - mu) / sigma.sqrt()).sort().values
        nll.append(-tok_lp.mean().item())
        minkpp.append(z[:max(int(len(z) * 0.2), 1)].mean().item())
    return float(np.mean(nll)), float(np.mean(minkpp))


@torch.no_grad()
def mmlu() -> float:
    ds = load_dataset("jinzhuoran/RWKU", "utility_general", split="test").shuffle(seed=0).select(range(args.mmlu_n))
    ds = sorted(ds, key=lambda s: len(s["question"]) + sum(len(e["question"]) for e in s["examples"]))  # similar lengths per batch
    letters = [tok.encode(" " + c, add_special_tokens=False)[-1] for c in "ABCD"]
    tok.padding_side = "left"
    correct = 0
    for i in range(0, len(ds), 8):
        batch = ds[i:i + 8]
        prompts = []
        for s in batch:
            def fmt(e, ans):
                t = "Question: " + e["question"] + "".join(f"\n{c}. {x}" for c, x in zip("ABCD", e["choices"]))
                return t + (f"\nAnswer: {'ABCD'[e['answer']]}\n\n" if ans else "")
            head = f"The following are multiple choice questions (with answers) about {s['task'].replace('_', ' ')}.\n\n"
            prompts.append(chat(head + "".join(fmt(e, True) for e in s["examples"])
                                + "Please following the previous examples and answer the given question.\n" + fmt(s, False), True))
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        last = model(**enc).logits[:, -1, letters].float()
        correct += sum(int(p == s["answer"]) for p, s in zip(last.argmax(-1).tolist(), batch))
    return correct / len(ds)


row = {"tag": args.tag, "target": args.target,
       "forget_fb": recall("forget_level1"), "forget_qa": recall("forget_level2"), "forget_aa": recall("forget_level3"),
       "neighbor_fb": recall("neighbor_level1"), "neighbor_qa": recall("neighbor_level2")}
row["fm_nll"], row["fm_minkpp"] = mia("mia_forget")
row["rm_nll"], row["rm_minkpp"] = mia("mia_retain")
if not args.skip_mmlu:
    row["mmlu"] = mmlu()
dst = ROOT / "results" / args.run_id / "rwku" / "eval"
dst.mkdir(parents=True, exist_ok=True)
with open(dst / f"{args.tag}_{slug}.json", "w") as f:
    json.dump(row, f, indent=1)
print("RWKU_EVAL", json.dumps(row), flush=True)
