"""字粒度 / 词粒度 Word2Vec 静态向量训练."""

from __future__ import annotations

from pathlib import Path

import jieba
from gensim.models import Word2Vec

from src.preprocess.annotate import load_bio_file
from src.utils import resolve_path


def _load_all_sentences(processed_dir: Path) -> list[str]:
    texts: list[str] = []
    for name in ("train.bio", "dev.bio", "test.bio"):
        path = processed_dir / name
        if not path.exists():
            continue
        for sample in load_bio_file(path):
            texts.append("".join(ch for ch, _ in sample))
    return texts


def train_word2vec_models(cfg: dict) -> dict[str, Path]:
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    emb_dir = resolve_path(cfg["paths"]["embedding_dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)

    sentences = _load_all_sentences(processed_dir)
    if not sentences:
        raise FileNotFoundError("未找到BIO数据，请先构建数据集。")

    w2v_cfg = cfg["word2vec"]
    common = dict(
        vector_size=w2v_cfg["vector_size"],
        window=w2v_cfg["window"],
        min_count=w2v_cfg["min_count"],
        sg=w2v_cfg["sg"],
        workers=w2v_cfg["workers"],
        epochs=w2v_cfg["epochs"],
        seed=cfg["project"]["seed"],
    )

    # 字粒度：每个字为一个token
    char_corpus = [list(s) for s in sentences]
    char_model = Word2Vec(sentences=char_corpus, **common)
    char_path = emb_dir / "word2vec_char.model"
    char_model.save(str(char_path))

    # 词粒度：jieba分词
    word_corpus = [list(jieba.cut(s)) for s in sentences]
    word_model = Word2Vec(sentences=word_corpus, **common)
    word_path = emb_dir / "word2vec_word.model"
    word_model.save(str(word_path))

    print(f"[word2vec] 字向量词表大小: {len(char_model.wv)}")
    print(f"[word2vec] 词向量词表大小: {len(word_model.wv)}")
    print(f"[word2vec] 已保存: {char_path}, {word_path}")
    return {"char": char_path, "word": word_path}