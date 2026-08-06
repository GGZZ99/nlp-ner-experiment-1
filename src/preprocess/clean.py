"""语料基础整理：空白规范化、长度过滤、去重（实验一不含敏感词过滤）。"""

from __future__ import annotations

import re
from pathlib import Path

from src.utils import resolve_path, write_lines


def basic_clean_sentence(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_sentences(
    sentences: list[str],
    *,
    min_len: int = 8,
    max_len: int = 200,
    **_ignored,
) -> tuple[list[str], dict]:
    """整理句子列表：去空白、长度过滤、去重。"""
    kept: list[str] = []
    seen: set[str] = set()
    stats = {
        "input": len(sentences),
        "empty_or_len": 0,
        "duplicate_removed": 0,
        "kept": 0,
    }

    for raw in sentences:
        s = basic_clean_sentence(raw)
        if not s or not (min_len <= len(s) <= max_len):
            stats["empty_or_len"] += 1
            continue
        if not re.search(r"[\u4e00-\u9fff]", s):
            stats["empty_or_len"] += 1
            continue
        if s in seen:
            stats["duplicate_removed"] += 1
            continue
        seen.add(s)
        kept.append(s)

    stats["kept"] = len(kept)
    return kept, stats


def clean_corpus_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    min_len: int = 8,
    max_len: int = 200,
) -> tuple[list[str], dict]:
    input_path = resolve_path(input_path)
    output_path = resolve_path(output_path) if output_path else input_path
    with open(input_path, "r", encoding="utf-8") as f:
        sentences = [ln.rstrip("\n") for ln in f if ln.strip()]
    cleaned, stats = clean_sentences(sentences, min_len=min_len, max_len=max_len)
    write_lines(cleaned, output_path)
    stats["input_file"] = str(input_path)
    stats["output_file"] = str(output_path)
    return cleaned, stats
