"""Run WorldRepair on one benchmark.

    python scripts/repair.py --config configs/tofu.yaml --run-id tofu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn.common import load_config, run_dir  # noqa: E402
from unlearn.worldrepair import repair  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. repair.eta=0.8")
args = ap.parse_args()

cfg = load_config(args.config, args.set)
repair(cfg, run_dir(args.run_id, "repair"))
