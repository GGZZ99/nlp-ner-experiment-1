"""步骤3：运行四组对比实验。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.runner import (
    exp1_char_vs_word,
    exp2_hidden_layers,
    exp3_hidden_units,
    exp4_entity_types,
    run_all_experiments,
)
from src.utils import ensure_dirs, load_config, resolve_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="有色金属NER对比实验")
    parser.add_argument("--exp", choices=["all", "1", "2", "3", "4"], default="all")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    result_dir = resolve_path(cfg["paths"]["result_dir"])

    if args.exp == "all":
        run_all_experiments(cfg)
        return
    if args.exp == "1":
        save_json(exp1_char_vs_word(cfg), result_dir / "exp1_char_vs_word.json")
    elif args.exp == "2":
        save_json(exp2_hidden_layers(cfg), result_dir / "exp2_hidden_layers.json")
    elif args.exp == "3":
        save_json(exp3_hidden_units(cfg), result_dir / "exp3_hidden_units.json")
    elif args.exp == "4":
        rows, overall = exp4_entity_types(cfg)
        save_json({"by_type": rows, "overall": overall}, result_dir / "exp4_entity_types.json")


if __name__ == "__main__":
    main()
