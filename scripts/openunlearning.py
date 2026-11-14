"""Baselines and native metrics through OpenUnlearning (baselines are not reimplemented).

    python scripts/openunlearning.py setup                      # clone pinned OpenUnlearning, install, fetch eval logs + WMDP corpora
    python scripts/openunlearning.py train --bench tofu --trainer NPO --run-id <id>
    python scripts/openunlearning.py train --bench tofu --trainer MEGD --run-id <id> trainer.args.num_train_epochs=5
    python scripts/openunlearning.py train --bench tofu-paired --world A --model <world ckpt> --trainer NPO --run-id <id>
    python scripts/openunlearning.py eval --bench tofu --model <ckpt> --run-id <id> --tag npo
    python scripts/openunlearning.py eval --bench tofu --model <full ckpt> --run-id <id> --tag icu --icu

Benchmarks: tofu (forget10/retain90), muse (News), wmdp-bio / wmdp-cyber (lm-eval: WMDP + MMLU),
tofu-paired / wmdp-bio-paired (forget = that world's facts, retain = the shared retain set).
Checkpoints go to <UNL_RUNS>/<run-id>/baseline/<trainer>/; eval summaries to results/<run-id>/<bench>/native/.
Add --dry-run to print the composed Hydra config without running.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn.common import ROOT, RUNS, load_config  # noqa: E402
from unlearn.data import TOFU_ICU  # noqa: E402

OU = Path(os.environ.get("UNL_OU_DIR", str(ROOT.parent / "open-unlearning")))
OU_COMMIT = "2d98027"
TOKENIZER = {"muse": "NousResearch/Llama-2-7b-hf"}  # MUSE target checkpoints ship without tokenizer files
PY = OU / ".venv" / "bin" / "python"  # separate env on top of the system torch/transformers, so the repair stack is untouched
BASE = {  # OpenUnlearning model config and default checkpoint
    "tofu": ("Llama-3.2-1B-Instruct", "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"),
    "muse": ("Llama-2-7b-hf", "muse-bench/MUSE-News_target"),  # tokenizer: see TOKENIZER
    "wmdp-bio": ("zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-beta"),
    "wmdp-cyber": ("zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-beta"),
    "tofu-paired": ("Llama-3.2-1B-Instruct", None),
    "wmdp-bio-paired": ("zephyr-7b-beta", None),
}


def run(cmd: list[str], dry: bool) -> None:
    cmd = cmd + ["model.model_args.attn_implementation=sdpa"]  # flash-attn is not installed
    if dry:
        cmd = cmd + ["--cfg", "job"]
    print(" ".join(cmd), flush=True)
    # OpenUnlearning imports its modules from src/; keep any existing entries (site dirs added via PYTHONPATH)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(OU / "src"), os.environ.get("PYTHONPATH", "")]))}
    env.pop("PYTHONSAFEPATH", None)
    env["HF_DATASETS_CACHE"] = str(OU / ".hf_datasets")  # its pinned `datasets` cannot read caches written by newer versions
    subprocess.run(cmd, cwd=OU, check=True, env=env)


PLUGINS = ROOT / "scripts" / "ou_plugins"


def install_plugins() -> None:
    """Methods missing from OpenUnlearning (ME+GD, IDK+AP) as trainer plugins; idempotent."""
    shutil.copy(PLUGINS / "closer_look.py", OU / "src" / "trainer" / "unlearn" / "closer_look.py")
    for name in ("MEGD", "IDKAP"):
        shutil.copy(PLUGINS / f"{name}.yaml", OU / "configs" / "trainer" / f"{name}.yaml")
    init = OU / "src" / "trainer" / "__init__.py"
    text = init.read_text()
    if "closer_look" not in text:
        init.write_text(text + "\nfrom trainer.unlearn.closer_look import MEGD, IDKAP  # noqa: E402\n"
                        "_register_trainer(MEGD)\n_register_trainer(IDKAP)\n")


def setup(args) -> None:
    if not OU.exists():
        subprocess.run(["git", "clone", "https://github.com/locuslab/open-unlearning.git", str(OU)], check=True)
    subprocess.run(["git", "checkout", "-q", OU_COMMIT], cwd=OU, check=True)
    if not PY.exists():
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(OU / ".venv")], check=True)
    reqs = [r.strip() for r in open(OU / "requirements.txt") if r.strip() and not r.startswith("#")
            and not r.split("=")[0].split(">")[0].split("<")[0].strip().lower() in ("torch", "transformers", "accelerate")]
    subprocess.run([str(PY), "-m", "pip", "install", "-q", *reqs, "lm-eval"], cwd=OU, check=True)
    subprocess.run([str(PY), "-m", "pip", "install", "-q", "--no-deps", "-e", "."], cwd=OU, check=True)
    subprocess.run([str(PY), "setup_data.py", "--eval", "--wmdp", "--idk"], cwd=OU, check=True)


def world_files(bench: str, world: str) -> Path:
    """World facts as an OpenUnlearning QA json file (question / answer)."""
    cfg = load_config(ROOT / "configs" / f"{bench.replace('-', '_')}.yaml")
    src = ROOT / cfg["paired"]["dir"] / f"world_{world}.jsonl"
    dst = OU / "data" / "unl_paired" / bench / f"world_{world}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(src) if line.strip()]
    json.dump([{"question": r["prompt"], "answer": r["completion"]} for r in rows], open(dst, "w"))
    return dst


def train(args) -> None:
    install_plugins()
    model_cfg, default_ckpt = BASE[args.bench]
    ckpt = args.model or default_ckpt
    out = RUNS / args.run_id / "baseline" / args.trainer
    common = [f"trainer={args.trainer}", f"model={model_cfg}", f"model.model_args.pretrained_model_name_or_path={ckpt}",
              f"model.tokenizer_args.pretrained_model_name_or_path={TOKENIZER.get(args.bench, ckpt)}",
              f"task_name={args.run_id}-{args.trainer}", f"paths.output_dir={out}",
              "trainer.args.do_eval=false", "trainer.args.eval_strategy=no", "trainer.args.eval_on_start=false"]
    if args.bench == "tofu" and args.trainer == "IDKAP":  # forget: question -> IDK; retain: answer vs IDK pairs
        exp = ["experiment=unlearn/tofu/idk", "forget_split=forget10", "retain_split=retain90",
               "data.retain.TOFU_QA_retain.handler=QAwithIdkDataset",
               "+data.retain.TOFU_QA_retain.args.idk_path=./data/idk.jsonl",
               "+data.retain.TOFU_QA_retain.args.return_original=true"]
    elif args.bench == "tofu":
        exp = ["experiment=unlearn/tofu/default", "forget_split=forget10", "retain_split=retain90"]
    elif args.bench == "muse":
        exp = ["experiment=unlearn/muse/default", "data_split=News"]
    elif args.bench.startswith("wmdp") and not args.bench.endswith("paired"):
        exp = ["experiment=unlearn/wmdp/default", f"data_split={args.bench.split('-')[1]}"]
    else:  # paired worlds: forget = world facts, retain = shared retain set
        f = world_files(args.bench, args.world)
        forget = ["data.forget.TOFU_QA_forget.args.hf_args.path=json",
                  f"+data.forget.TOFU_QA_forget.args.hf_args.data_files={f}",
                  "~data.forget.TOFU_QA_forget.args.hf_args.name"]
        if args.bench == "tofu-paired":
            exp = ["experiment=unlearn/tofu/default", "retain_split=retain90", *forget]
        else:
            exp = ["experiment=unlearn/tofu/default", "data/datasets@data.retain=WMDP_retain",
                   "data.retain.WMDP_retain.args.hf_args.data_files=data/wmdp/wmdp-corpora/bio-retain-corpus.jsonl",
                   *forget]
    run([str(PY), "src/train.py", "--config-name=unlearn.yaml", *exp, *common, *args.overrides], args.dry_run)
    if not args.dry_run:
        print(f"saved {out}")


def evaluate(args) -> None:
    model_cfg, _ = BASE[args.bench]
    task = f"{args.run_id}-{args.tag}-eval"
    out = OU / "saves" / "eval" / task
    common = [f"model={model_cfg}", f"model.model_args.pretrained_model_name_or_path={args.model}",
              f"model.tokenizer_args.pretrained_model_name_or_path={args.tokenizer or TOKENIZER.get(args.bench, args.model)}",
              f"task_name={task}", f"paths.output_dir={out}"]
    if args.bench == "tofu":
        # float32: probability metrics call .numpy() on logits, which recent transformers keep in bf16
        exp = ["experiment=eval/tofu/default", "forget_split=forget10", "holdout_split=holdout10",
               "model.model_args.torch_dtype=float32",
               f"retain_logs_path=saves/eval/tofu_{model_cfg}_retain90/TOFU_EVAL.json"]
    elif args.bench == "muse":
        exp = ["experiment=eval/muse/default", "data_split=News",
               f"retain_logs_path=saves/eval/muse_{model_cfg}_News_retrain/MUSE_EVAL.json"]
    elif args.bench.startswith("wmdp"):
        exp = ["experiment=eval/wmdp/default", f"data_split={args.bench.split('-')[1]}"]
    else:
        raise ValueError(f"no native metrics for {args.bench}")
    entry = ["src/eval.py"]
    if args.icu:  # instruction prepended to every user prompt, for generation and probabilities alike
        os.environ["UNL_ICU_PREFIX"] = TOFU_ICU
        entry = [str(PLUGINS / "icu_eval.py")]
    run([str(PY), *entry, "--config-name=eval.yaml", *exp, *common, *args.overrides], args.dry_run)
    if args.dry_run:
        return
    summaries = sorted(out.glob("*SUMMARY*.json")) or sorted(out.glob("*.json"))
    dst = ROOT / "results" / args.run_id / args.bench / "native"
    dst.mkdir(parents=True, exist_ok=True)
    for s in summaries:
        shutil.copy(s, dst / f"{args.tag}_{s.name}")
        print(f"{s.name}: {json.load(open(s))}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("setup")
for name in ("train", "eval"):
    p = sub.add_parser(name)
    p.add_argument("--bench", required=True, choices=list(BASE))
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--tokenizer", default=None, help="tokenizer repo when the checkpoint's own files cannot be read")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("overrides", nargs="*", help="extra Hydra overrides")
    if name == "train":
        p.add_argument("--trainer", required=True, choices=["GradDiff", "NPO", "SimNPO", "RMU", "MEGD", "IDKAP"])
        p.add_argument("--world", choices=["A", "B"])
    else:
        p.add_argument("--tag", required=True)
        p.add_argument("--icu", action="store_true", help="in-context unlearning (TOFU instruction on the base model)")
args = ap.parse_args()
{"setup": setup, "train": train, "eval": evaluate}[args.cmd](args)
