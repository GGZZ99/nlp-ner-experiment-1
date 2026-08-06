"""NER评估：token级准确率 + 实体级精确率/召回率/F1."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def _extract_entities(tags: list[str]) -> set[tuple[str, int, int]]:
    """从BIO序列抽取实体: (type, start, end_exclusive)."""
    ents: set[tuple[str, int, int]] = set()
    i, n = 0, len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{etype}":
                j += 1
            ents.add((etype, i, j))
            i = j
        else:
            i += 1
    return ents


def entity_level_scores(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    entity_types: Iterable[str] | None = None,
) -> dict:
    """实体级 micro P/R/F1，并按类别统计."""
    gold_all: set[tuple[str, int, int, int]] = set()
    pred_all: set[tuple[str, int, int, int]] = set()
    gold_by: dict[str, set] = defaultdict(set)
    pred_by: dict[str, set] = defaultdict(set)

    for sid, (gt, pr) in enumerate(zip(y_true, y_pred)):
        for et, s, e in _extract_entities(gt):
            item = (sid, et, s, e)
            gold_all.add(item)
            gold_by[et].add(item)
        for et, s, e in _extract_entities(pr):
            item = (sid, et, s, e)
            pred_all.add(item)
            pred_by[et].add(item)

    def prf(g: set, p: set) -> dict[str, float]:
        tp = len(g & p)
        precision = tp / len(p) if p else 0.0
        recall = tp / len(g) if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": len(g),
        }

    overall = prf(gold_all, pred_all)
    by_type = {}
    types = list(entity_types) if entity_types else sorted(set(gold_by) | set(pred_by))
    for et in types:
        by_type[et] = prf(gold_by.get(et, set()), pred_by.get(et, set()))
    return {"overall": overall, "by_type": by_type}


def token_accuracy(y_true_flat: list[str], y_pred_flat: list[str]) -> float:
    if not y_true_flat:
        return 0.0
    correct = sum(a == b for a, b in zip(y_true_flat, y_pred_flat))
    return round(correct / len(y_true_flat), 4)


def evaluate_predictions(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    entity_types: Iterable[str] | None = None,
) -> dict:
    flat_t = [t for seq in y_true for t in seq]
    flat_p = [t for seq in y_pred for t in seq]
    ent = entity_level_scores(y_true, y_pred, entity_types)
    return {
        "accuracy": token_accuracy(flat_t, flat_p),
        "precision": ent["overall"]["precision"],
        "recall": ent["overall"]["recall"],
        "f1": ent["overall"]["f1"],
        "by_type": ent["by_type"],
    }