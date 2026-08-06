"""用训练好的权重对单句话做实体识别。

示例:
  python scripts/04_predict.py --text "江西铜业提升电解铜产量"
  python scripts/04_predict.py --text "德兴铜矿储量丰富" --model outputs/models/exp2_layers_1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import format_prediction, predict_sentence
from src.utils import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="单句NER预测")
    parser.add_argument("--text", required=True, help="待识别的中文句子")
    parser.add_argument(
        "--model",
        default="outputs/models/exp1_char.pt",
        help="权重文件路径，默认用字粒度模型",
    )
    args = parser.parse_args()

    cfg = load_config()
    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型: {model_path}\n请先运行 scripts/03_run_experiments.py 生成权重。"
        )

    result = predict_sentence(args.text, model_path, cfg=cfg)
    print(format_prediction(result))


if __name__ == "__main__":
    main()
