"""滑动窗口NER数据集：将上下文embedding拼接后作为全连接网络输入."""

from __future__ import annotations

from typing import Literal

import jieba
import numpy as np
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset


def build_label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


class NERWindowDataset(Dataset):
    """
    对每个中心位置构造窗口特征。
    - char模式：token=字
    - word模式：先jieba分词，将词向量按字对齐（词内每个字共享该词向量）
    """

    def __init__(
        self,
        samples: list[list[tuple[str, str]]],
        w2v: Word2Vec,
        label2id: dict[str, int],
        window_size: int = 2,
        granularity: Literal["char", "word"] = "char",
    ):
        self.window_size = window_size
        self.granularity = granularity
        self.emb_dim = w2v.vector_size
        self.unk = np.zeros(self.emb_dim, dtype=np.float32)

        self.features: list[np.ndarray] = []
        self.targets: list[int] = []
        self.meta: list[tuple[int, int, str]] = []  # (sample_idx, char_idx, label)

        for sid, sample in enumerate(samples):
            chars = [ch for ch, _ in sample]
            labels = [lab for _, lab in sample]
            emb_seq = self._build_char_embeddings(chars, w2v)
            n = len(chars)
            for i in range(n):
                vecs = []
                for j in range(i - window_size, i + window_size + 1):
                    if 0 <= j < n:
                        vecs.append(emb_seq[j])
                    else:
                        vecs.append(self.unk)
                feat = np.concatenate(vecs, axis=0).astype(np.float32)
                self.features.append(feat)
                self.targets.append(label2id.get(labels[i], label2id["O"]))
                self.meta.append((sid, i, labels[i]))

    def _build_char_embeddings(self, chars: list[str], w2v: Word2Vec) -> list[np.ndarray]:
        if self.granularity == "char":
            return [self._get_vec(ch, w2v) for ch in chars]

        # word粒度：分词后映射回字序列
        text = "".join(chars)
        words = list(jieba.cut(text))
        emb_seq: list[np.ndarray] = []
        for w in words:
            vec = self._get_vec(w, w2v)
            for _ in w:
                emb_seq.append(vec)
        # 防御：分词后长度不一致时回退字向量
        if len(emb_seq) != len(chars):
            return [self._get_vec(ch, w2v) for ch in chars]
        return emb_seq

    def _get_vec(self, token: str, w2v: Word2Vec) -> np.ndarray:
        if token in w2v.wv:
            return w2v.wv[token].astype(np.float32)
        return self.unk.copy()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y