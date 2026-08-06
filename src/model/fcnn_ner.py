"""全连接神经网络实体识别模型."""

from __future__ import annotations

import torch
import torch.nn as nn


class FCNN_NER(nn.Module):
    """
    输入: 拼接后的窗口向量 [batch, (2w+1)*emb_dim]
    输出: 各BIO标签logits [batch, num_labels]
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_units: int = 600,
        num_hidden_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers 至少为 1")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, num_labels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)