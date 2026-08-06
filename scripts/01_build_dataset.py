"""步骤1：数据标注与划分（实验一起点）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.pipeline import build_dataset
from src.utils import ensure_dirs, load_config


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    build_dataset(cfg)


if __name__ == "__main__":
    main()
