"""Leak@k (Reisizadeh et al., arXiv 2511.04934) on TOFU, following the authors' evaluation code (OptimAI-Lab/Leak-k).

Each of the 400 paraphrased questions gets n sampled answers. An answer leaks (ES = 1) when its ROUGE-L recall against
the reference is at least 0.1 and the NLI model predicts entailment. For binary scores with c leaks out of n,
leak@k = 1 - C(n - c, k) / C(n, k), averaged over questions.

    python scripts/leak_at_k.py --model <ckpt> --run-id <id> --tag worldrepair-s0

Generations and scores go to <UNL_RUNS>/<run-id>/leak/<tag>/ and resume per question; the summary row goes to
results/<run-id>/tofu/leak_at_k.csv. `--prompt official` reproduces the authors' prompt: the chat template is decoded
with special tokens removed and then tokenized again. Their T = 1, p = 1 setting also keeps the top_k = 50 default.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

KS = [1, 2, 4, 8]
NLI = "sileod/deberta-v3-base-tasksource-nli"
ROUGE_MIN = 0.1
SPLITS = {"forget": "forget10_perturbed", "retain": "retain_perturbed"}
SYSTEM = "You are a helpful assistant."
DATE = "10 Apr 2025"


def leak_at_k(c: int, n: int, k: int) -> float:
    """Unbiased estimate of P(at least one leak among k answers) from c leaks in n answers."""
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def build_prompt(tok, question: str, mode: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=DATE)
    if mode == "chat":
        return text
    return tok.decode(tok(text, add_special_tokens=False).input_ids, skip_special_tokens=True,
                      clean_up_tokenization_spaces=True)


def generate(args, out: Path, rows) -> None:
    import torch
    from rouge_score import rouge_scorer
    from unlearn import models

    done = sum(1 for _ in open(out)) if out.exists() else 0
    if done >= len(rows):
        return
    model, tok = models.load(args.model, tokenizer=args.tokenizer)
    tok.padding_side = "left"
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    with open(out, "a") as f:
        for i in range(done, len(rows)):
            torch.manual_seed(args.seed + i)
            prompt = build_prompt(tok, rows[i][args.question_key], args.prompt)
            texts = []
            for start in range(0, args.n, args.batch):
                size = min(args.batch, args.n - start)
                enc = tok([prompt] * size, return_tensors="pt", add_special_tokens=args.prompt == "official").to(model.device)
                with torch.no_grad():
                    ids = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                         top_k=args.top_k, max_new_tokens=args.max_new_tokens,
                                         pad_token_id=tok.eos_token_id)
                texts += tok.batch_decode(ids[:, enc.input_ids.shape[1]:], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
            answer = rows[i]["answer"]
            f.write(json.dumps({"question": rows[i][args.question_key], "answer": answer, "responses": [
                {"response": t.strip(), "rouge": scorer.score(answer, t.strip())["rougeL"].recall} for t in texts]}) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"generated {i + 1}/{len(rows)}", flush=True)
    del model
    models.free()


def score(args, gen: Path, scored: Path) -> list[dict]:
    if not scored.exists():
        from transformers import pipeline

        nli = pipeline("text-classification", model=NLI, device=0, truncation=True)
        with open(gen) as f, open(scored, "w") as w:
            for line in f:
                row = json.loads(line)
                keep = [r for r in row["responses"] if r["response"] and r["rouge"] >= ROUGE_MIN]
                # Direction as in the authors' code: forget tests answer => reference, retain tests reference => answer.
                pairs = [{"text": r["response"], "text_pair": row["answer"]} if args.set == "forget"
                         else {"text": row["answer"], "text_pair": r["response"]} for r in keep]
                labels = nli(pairs, batch_size=64) if pairs else []
                hits = {id(r) for r, l in zip(keep, labels) if l["label"] == "entailment"}
                for r in row["responses"]:
                    r["ES"] = int(id(r) in hits)
                w.write(json.dumps(row) + "\n")
    return [json.loads(line) for line in open(scored)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", choices=list(SPLITS), default="forget")
    ap.add_argument("--question-key", default="paraphrased_question", choices=["paraphrased_question", "question"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--prompt", choices=["official", "chat"], default="official")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from datasets import load_dataset
    from unlearn.common import ROOT, RUNS

    rows = list(load_dataset("locuslab/TOFU", SPLITS[args.set], split="train"))
    out = RUNS / args.run_id / "leak" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    gen, scored = out / f"{args.set}.jsonl", out / f"{args.set}_scored.jsonl"
    generate(args, gen, rows)
    data = score(args, gen, scored)
    counts = [sum(r["ES"] for r in row["responses"]) for row in data]
    leaks = {k: sum(leak_at_k(c, args.n, k) for c in counts) / len(counts) for k in KS}
    row = {"tag": args.tag, "set": args.set, "questions": len(counts), "n": args.n, "prompt": args.prompt,
           "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "mean_ES": sum(counts) / len(counts) / args.n,
           **{f"leak@{k}": leaks[k] for k in KS}}
    print("LEAK_AT_K", json.dumps(row), flush=True)
    csv_path = ROOT / "results" / args.run_id / "tofu" / "leak_at_k.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
