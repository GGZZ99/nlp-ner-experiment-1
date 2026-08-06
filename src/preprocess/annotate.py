"""基于领域词典的远监督BIO标注。

最长优先匹配，避免短实体覆盖长实体（如“铜”覆盖“电解铜”）。
"""

from __future__ import annotations

from pathlib import Path

from src.utils import read_lines, resolve_path


ENTITY_FILE_MAP = {
    "PROD": "prod.txt",
    "ORG": "org.txt",
    "MINE": "mine.txt",
    "LOC": "loc.txt",
}


def load_entity_dict(dict_dir: str | Path) -> list[tuple[str, str]]:
    """返回 (实体文本, 类型) 列表，按长度降序。"""
    dict_dir = resolve_path(dict_dir)
    items: list[tuple[str, str]] = []
    for etype, fname in ENTITY_FILE_MAP.items():
        for w in read_lines(dict_dir / fname):
            items.append((w, etype))
    items.sort(key=lambda x: len(x[0]), reverse=True)
    return items


def bio_tag_sentence(sentence: str, entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """对单句进行字级BIO标注，返回 [(char, label), ...]。"""
    n = len(sentence)
    tags = ["O"] * n
    covered = [False] * n

    for ent, etype in entities:
        if not ent:
            continue
        start = 0
        while True:
            idx = sentence.find(ent, start)
            if idx < 0:
                break
            end = idx + len(ent)
            if not any(covered[idx:end]):
                tags[idx] = f"B-{etype}"
                for j in range(idx + 1, end):
                    tags[j] = f"I-{etype}"
                for j in range(idx, end):
                    covered[j] = True
            start = idx + 1

    return list(zip(list(sentence), tags))


def annotate_corpus(
    sentences: list[str],
    dict_dir: str | Path,
    min_len: int = 8,
    max_len: int = 200,
) -> list[list[tuple[str, str]]]:
    """批量标注，仅保留含实体且长度合适的句子。"""
    entities = load_entity_dict(dict_dir)
    samples: list[list[tuple[str, str]]] = []
    for sent in sentences:
        sent = sent.strip().replace(" ", "")
        if not (min_len <= len(sent) <= max_len):
            continue
        tagged = bio_tag_sentence(sent, entities)
        if any(lab != "O" for _, lab in tagged):
            samples.append(tagged)
    return samples


def save_bio_file(samples: list[list[tuple[str, str]]], path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            for ch, lab in sample:
                f.write(f"{ch}\t{lab}\n")
            f.write("\n")


def load_bio_file(path: str | Path) -> list[list[tuple[str, str]]]:
    path = resolve_path(path)
    samples: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    samples.append(cur)
                    cur = []
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                cur.append((parts[0], parts[1]))
    if cur:
        samples.append(cur)
    return samples