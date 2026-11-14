"""RWKU single-target experiment: every method starts from the input model and forgets one person.

Per person: Base, in-context unlearning (ICU), NPO, rejection tuning (RT), PURGE (the authors' released checkpoint,
evaluated only), WorldRepair (forget stage), WorldRepair+N (a short second stage with neighbor retention, chained from
the forget stage). Each gets the RWKU evaluation and the WorldLeak audit.
Finished steps are skipped on restart (marker files), failed steps are retried; training steps hold the GPU alone
(file lock), and the evaluation of a model whose training failed is skipped.

    python scripts/rwku_run.py --run-id rwku --persons 0 2 4 6 8
    python scripts/rwku_run.py --run-id rwku --persons 0 1 2 3 --methods rt purge
"""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn.common import RUNS  # noqa: E402
from unlearn.data import RWKU_ICU  # noqa: E402

PHI3 = "microsoft/Phi-3-mini-4k-instruct"
ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--persons", type=int, nargs="+", required=True, help="indices into RWKU's target list")
ap.add_argument("--npo-lr", default="1e-5")
METHODS = ["base", "icu", "npo", "rt", "purge", "wr", "wrn"]
ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
args = ap.parse_args()
names = load_dataset("jinzhuoran/RWKU", "forget_target", split="train")["target"]
marks = RUNS / args.run_id / "done"
marks.mkdir(parents=True, exist_ok=True)


def step(key: str, cmd: list[str], after: str | None = None) -> None:
    mark = marks / key
    if key.split(".")[1] not in args.methods or mark.exists():
        return
    if after and not (marks / after).exists():
        print(f"== SKIPPED {key}: {after} did not finish", flush=True)
        return
    # float32 training needs most of the GPU memory: it runs alone, other steps may share the GPU.
    with open(marks.parent / "gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if key.endswith((".repair", ".train")) else fcntl.LOCK_SH)
        for attempt in range(3):
            print(f"== {key} attempt {attempt} {time.strftime('%H:%M:%S')}", flush=True)
            if subprocess.run(cmd, env={**os.environ, "UNL_RESUME": "1"}).returncode == 0:
                mark.touch()
                return
            time.sleep(60)
    print(f"== FAILED {key}", flush=True)


py = sys.executable
for i in args.persons:
    t = names[i]
    slug = t.replace(" ", "_").replace(",", "").replace(".", "")
    tgt = f"data.target={t}"
    run = args.run_id
    audit = [py, "scripts/audit.py", "--config", "configs/rwku.yaml", "--run-id", run]
    evl = [py, "scripts/rwku_eval.py", "--target", t, "--run-id", run]
    step(f"{slug}.base.eval", evl + ["--model", PHI3, "--tag", "base"])
    step(f"{slug}.base.audit", audit + ["--set", tgt, "--model", PHI3, "--tag", f"base-{slug}"])
    step(f"{slug}.icu.eval", evl + ["--model", PHI3, "--tag", "icu", "--icu", "--skip-mmlu"])
    step(f"{slug}.icu.audit", audit + ["--set", tgt, "--prefix", RWKU_ICU, "--model", PHI3, "--tag", f"icu-{slug}"])
    npo = str(RUNS / run / "npo" / slug)
    step(f"{slug}.npo.train", [py, "scripts/rwku_train.py", "--method", "npo", "--target", t, "--run-id", run, "--lr", args.npo_lr])
    step(f"{slug}.npo.eval", evl + ["--model", npo, "--tag", "npo"], after=f"{slug}.npo.train")
    step(f"{slug}.npo.audit", audit + ["--set", tgt, "--model", npo, "--tag", f"npo-{slug}"], after=f"{slug}.npo.train")
    rt = str(RUNS / run / "rt" / slug)
    # RT and PURGE change the weights by less than bf16 resolution (relative ~1e-4 to 1e-3): keep and run them in float32
    f32 = ["--dtype", "float32"]
    step(f"{slug}.rt.train", [py, "scripts/rwku_train.py", "--method", "rt", "--target", t, "--run-id", run, "--save-dtype", "float32"])
    step(f"{slug}.rt.eval", evl + f32 + ["--model", rt, "--tag", "rt"], after=f"{slug}.rt.train")
    step(f"{slug}.rt.audit", audit + ["--set", tgt, "model.dtype=float32", "--model", rt, "--tag", f"rt-{slug}"], after=f"{slug}.rt.train")
    # PURGE: released per target in RWKU order; the model sits in the repo's only folder, checkpoint-1000
    repo = f"strzara/Phi-3-mini-4k-instruct-{i + 1}_{t.replace(' ', '_')}_PURGE"
    local = RUNS / run / "purge" / slug
    purge = str(local / "checkpoint-1000")
    # The saved config has use_cache=null (training with gradient checkpointing), which current transformers rejects;
    # it only switches the inference KV cache, so it is set back to true. Weights are untouched.
    step(f"{slug}.purge.fetch", [py, "-c", "import json; from huggingface_hub import snapshot_download as s; "
                                 f"s({repo!r}, local_dir={str(local)!r}, allow_patterns=['checkpoint-1000/*'], "
                                 "ignore_patterns=['*optimizer.pt', '*rng_state.pth', '*scheduler.pt', '*training_args.bin']); "
                                 f"p = {purge + '/config.json'!r}; c = json.load(open(p)); "
                                 "c['use_cache'] = True if c.get('use_cache') is None else c['use_cache']; "
                                 "json.dump(c, open(p, 'w'), indent=2)"])
    step(f"{slug}.purge.eval", evl + f32 + ["--model", purge, "--tokenizer", PHI3, "--tag", "purge"], after=f"{slug}.purge.fetch")
    step(f"{slug}.purge.audit", audit + ["--set", tgt, f"model.tokenizer={PHI3}", "model.dtype=float32", "--model", purge,
                                         "--tag", f"purge-{slug}"],
         after=f"{slug}.purge.fetch")
    wr = f"{run}-wr-{slug}"
    wr_model = str(RUNS / wr / "repair" / "model")
    step(f"{slug}.wr.repair", [py, "scripts/repair.py", "--config", "configs/rwku.yaml", "--run-id", wr, "--set", tgt])
    step(f"{slug}.wr.eval", evl + ["--model", wr_model, "--tag", "wr"], after=f"{slug}.wr.repair")
    step(f"{slug}.wr.audit", audit + ["--set", tgt, "--model", wr_model, "--tag", f"wr-{slug}"], after=f"{slug}.wr.repair")
    wn = f"{run}-wrn-{slug}"
    wn_model = str(RUNS / wn / "repair" / "model")
    step(f"{slug}.wrn.repair", [py, "scripts/repair.py", "--config", "configs/rwku.yaml", "--run-id", wn, "--set", tgt,
                                f"model.name={wr_model}", f"model.ref={PHI3}", "seed=1", "repair.blocks=3", "optim.lr=2.0e-6",
                                "repair.neighbor_retain=40", "repair.neighbor_skip=20", "repair.qa_batch=16", "repair.delta_Q=0.02"], after=f"{slug}.wr.repair")
    step(f"{slug}.wrn.eval", evl + ["--model", wn_model, "--tag", "wrn"], after=f"{slug}.wrn.repair")
    step(f"{slug}.wrn.audit", audit + ["--set", tgt, f"model.ref={PHI3}", "--model", wn_model, "--tag", f"wrn-{slug}"], after=f"{slug}.wrn.repair")
print("== ALL DONE", flush=True)
