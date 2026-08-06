"""数据标注流水线：整理语料 -> BIO标注 -> 划分."""

from __future__ import annotations

import random
from pathlib import Path

from src.preprocess.annotate import annotate_corpus, save_bio_file
from src.preprocess.clean import clean_sentences
from src.utils import resolve_path, save_json, write_lines


def split_samples(
    samples: list,
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    data = list(samples)
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = data[:n_train]
    dev = data[n_train : n_train + n_dev]
    test = data[n_train + n_dev :]
    return train, dev, test


def build_dataset(cfg: dict) -> dict:
    raw_path = resolve_path(Path(cfg["paths"]["raw_dir"]) / "corpus.txt")
    if not raw_path.exists():
        raise FileNotFoundError(f"未找到语料文件: {raw_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    ds_cfg = cfg["dataset"]
    cleaned, clean_stats = clean_sentences(
        sentences,
        min_len=ds_cfg["min_sentence_len"],
        max_len=ds_cfg["max_sentence_len"],
    )
    print("[dataset] 语料整理完成:", clean_stats)

    samples = annotate_corpus(
        cleaned,
        cfg["paths"]["dict_dir"],
        min_len=ds_cfg["min_sentence_len"],
        max_len=ds_cfg["max_sentence_len"],
    )
    if len(samples) < 50:
        raise RuntimeError(f"有效标注样本过少({len(samples)})，请检查词典与语料。")

    train, dev, test = split_samples(
        samples,
        train_ratio=ds_cfg["train_ratio"],
        dev_ratio=ds_cfg["dev_ratio"],
        seed=cfg["project"]["seed"],
    )

    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_bio_file(train, out_dir / "train.bio")
    save_bio_file(dev, out_dir / "dev.bio")
    save_bio_file(test, out_dir / "test.bio")

    def count_entities(data):
        counter = {"PROD": 0, "ORG": 0, "MINE": 0, "LOC": 0}
        for sample in data:
            for _, lab in sample:
                if lab.startswith("B-"):
                    et = lab[2:]
                    if et in counter:
                        counter[et] += 1
        return counter

    stats = {
        "num_sentences_raw": len(sentences),
        "num_sentences_used": len(cleaned),
        "num_samples_labeled": len(samples),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "entity_counts_all": count_entities(samples),
        "entity_counts_train": count_entities(train),
        "entity_counts_test": count_entities(test),
    }
    save_json(stats, out_dir / "dataset_stats.json")
    print("[dataset] 构建完成:", stats)
    return stats
