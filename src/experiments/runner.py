"""四组对比实验执行与结果可视化."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.train.trainer import train_and_evaluate
from src.utils import resolve_path, save_json

# 中文字体兼容（Windows常见字体）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ENTITY_CN = {
    "PROD": "有色金属产品",
    "ORG": "有色金属组织机构",
    "MINE": "有色金属矿产",
    "LOC": "有色金属矿产地名",
}


def _save_table(rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    df.to_excel(path.with_suffix(".xlsx"), index=False)
    return df


def exp1_char_vs_word(cfg: dict) -> list[dict]:
    """对比实验(1)：字粒度 vs 词粒度 Word2Vec + 全连接网络."""
    rows = []
    for g in cfg["experiments"]["exp1_granularity"]:
        result = train_and_evaluate(
            cfg,
            granularity=g,
            run_name=f"exp1_{g}",
        )
        rows.append(
            {
                "experiment": "exp1_char_vs_word",
                "granularity": g,
                "accuracy": result["test"]["accuracy"],
                "precision": result["test"]["precision"],
                "recall": result["test"]["recall"],
                "f1": result["test"]["f1"],
            }
        )
    return rows


def exp2_hidden_layers(cfg: dict) -> list[dict]:
    """对比实验(2)：不同隐层数（字粒度）."""
    rows = []
    for n in cfg["experiments"]["exp2_hidden_layers"]:
        result = train_and_evaluate(
            cfg,
            granularity="char",
            num_hidden_layers=n,
            run_name=f"exp2_layers_{n}",
        )
        rows.append(
            {
                "experiment": "exp2_hidden_layers",
                "num_hidden_layers": n,
                "accuracy": result["test"]["accuracy"],
                "precision": result["test"]["precision"],
                "recall": result["test"]["recall"],
                "f1": result["test"]["f1"],
            }
        )
    return rows


def exp3_hidden_units(cfg: dict) -> list[dict]:
    """对比实验(3)：不同隐层节点数（字粒度）."""
    rows = []
    for h in cfg["experiments"]["exp3_hidden_units"]:
        result = train_and_evaluate(
            cfg,
            granularity="char",
            hidden_units=h,
            run_name=f"exp3_units_{h}",
        )
        rows.append(
            {
                "experiment": "exp3_hidden_units",
                "hidden_units": h,
                "accuracy": result["test"]["accuracy"],
                "precision": result["test"]["precision"],
                "recall": result["test"]["recall"],
                "f1": result["test"]["f1"],
            }
        )
    return rows


def exp4_entity_types(cfg: dict) -> tuple[list[dict], dict]:
    """对比实验(4)：各类实体识别结果（字粒度默认模型）."""
    result = train_and_evaluate(
        cfg,
        granularity="char",
        run_name="exp4_entity_types",
    )
    rows = []
    for et in cfg["experiments"]["exp4_entity_types"]:
        m = result["test"]["by_type"].get(et, {})
        rows.append(
            {
                "experiment": "exp4_entity_types",
                "entity_type": et,
                "entity_name": ENTITY_CN.get(et, et),
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1", 0.0),
                "support": m.get("support", 0),
            }
        )
    overall = {
        "accuracy": result["test"]["accuracy"],
        "precision": result["test"]["precision"],
        "recall": result["test"]["recall"],
        "f1": result["test"]["f1"],
    }
    return rows, overall


def _plot_exp1(df: pd.DataFrame, fig_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    ax = df.set_index("granularity")[metrics].plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title("实验1：字粒度 vs 词粒度 Word2Vec")
    ax.set_ylabel("分数")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "exp1_char_vs_word.png", dpi=150)
    plt.close()


def _plot_line(df: pd.DataFrame, x: str, title: str, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    for m in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(df[x], df[m], marker="o", label=m)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("分数")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_exp4(df: pd.DataFrame, fig_dir: Path) -> None:
    metrics = ["precision", "recall", "f1"]
    ax = df.set_index("entity_name")[metrics].plot(kind="bar", figsize=(9, 5), rot=20)
    ax.set_title("实验4：各类有色金属实体识别结果")
    ax.set_ylabel("分数")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_dir / "exp4_entity_types.png", dpi=150)
    plt.close()


def run_all_experiments(cfg: dict) -> dict:
    result_dir = resolve_path(cfg["paths"]["result_dir"])
    fig_dir = resolve_path(cfg["paths"]["figure_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("实验1：字/词粒度对比")
    print("=" * 60)
    rows1 = exp1_char_vs_word(cfg)
    df1 = _save_table(rows1, result_dir / "exp1_char_vs_word.csv")
    _plot_exp1(df1, fig_dir)

    print("=" * 60)
    print("实验2：隐层数对比")
    print("=" * 60)
    rows2 = exp2_hidden_layers(cfg)
    df2 = _save_table(rows2, result_dir / "exp2_hidden_layers.csv")
    _plot_line(df2, "num_hidden_layers", "实验2：不同隐层数对比", fig_dir / "exp2_hidden_layers.png")

    print("=" * 60)
    print("实验3：隐层节点数对比")
    print("=" * 60)
    rows3 = exp3_hidden_units(cfg)
    df3 = _save_table(rows3, result_dir / "exp3_hidden_units.csv")
    _plot_line(df3, "hidden_units", "实验3：不同隐层节点数对比", fig_dir / "exp3_hidden_units.png")

    print("=" * 60)
    print("实验4：各类实体识别结果")
    print("=" * 60)
    rows4, overall4 = exp4_entity_types(cfg)
    df4 = _save_table(rows4, result_dir / "exp4_entity_types.csv")
    _plot_exp4(df4, fig_dir)
    save_json(overall4, result_dir / "exp4_overall.json")

    summary = {
        "exp1": rows1,
        "exp2": rows2,
        "exp3": rows3,
        "exp4": rows4,
        "exp4_overall": overall4,
    }
    save_json(summary, result_dir / "summary.json")
    print("[experiments] 全部完成，结果目录:", result_dir)
    return summary