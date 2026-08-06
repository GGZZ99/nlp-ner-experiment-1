"""单句实体识别推理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gensim.models import Word2Vec

from src.model.dataset import NERWindowDataset, build_label_maps
from src.model.fcnn_ner import FCNN_NER
from src.utils import get_device, load_config, resolve_path


ENTITY_CN = {
    "PROD": "有色金属产品",
    "ORG": "有色金属组织机构",
    "MINE": "有色金属矿产",
    "LOC": "有色金属矿产地名",
}


def _bio_to_entities(chars: list[str], tags: list[str]) -> list[dict[str, Any]]:
    ents: list[dict[str, Any]] = []
    i, n = 0, len(chars)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{etype}":
                j += 1
            text = "".join(chars[i:j])
            ents.append(
                {
                    "text": text,
                    "type": etype,
                    "type_name": ENTITY_CN.get(etype, etype),
                    "start": i,
                    "end": j,
                }
            )
            i = j
        else:
            i += 1
    return ents


def predict_sentence(
    text: str,
    model_path: str | Path,
    *,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """加载权重，对单句做实体识别。"""
    cfg = cfg or load_config()
    device = get_device()
    ckpt = torch.load(resolve_path(model_path), map_location=device)
    mcfg = ckpt["config"]

    granularity = mcfg["granularity"]
    emb_name = "word2vec_char.model" if granularity == "char" else "word2vec_word.model"
    w2v = Word2Vec.load(str(resolve_path(Path(cfg["paths"]["embedding_dir"]) / emb_name)))

    labels = cfg["labels"]
    label2id, id2label = build_label_maps(labels)

    text = text.strip().replace(" ", "")
    # 伪标注占位，推理时不用标签
    sample = [(ch, "O") for ch in text]
    ds = NERWindowDataset(
        [sample],
        w2v,
        label2id,
        window_size=mcfg["window_size"],
        granularity=granularity,
    )

    model = FCNN_NER(
        input_dim=mcfg["input_dim"],
        num_labels=mcfg["num_labels"],
        hidden_units=mcfg["hidden_units"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pred_tags: list[str] = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, _ = ds[i]
            logits = model(x.unsqueeze(0).to(device))
            pid = int(logits.argmax(dim=-1).item())
            pred_tags.append(id2label[pid])

    chars = list(text)
    entities = _bio_to_entities(chars, pred_tags)
    return {
        "text": text,
        "char_tags": list(zip(chars, pred_tags)),
        "entities": entities,
        "model": str(model_path),
        "granularity": granularity,
    }


def format_prediction(result: dict[str, Any]) -> str:
    lines = [f"句子: {result['text']}", "字级标签:"]
    for ch, tag in result["char_tags"]:
        lines.append(f"  {ch}\t{tag}")
    lines.append("识别实体:")
    if not result["entities"]:
        lines.append("  （无）")
    else:
        for e in result["entities"]:
            lines.append(f"  [{e['type_name']}/{e['type']}] {e['text']}  span=[{e['start']},{e['end']})")
    return "\n".join(lines)
