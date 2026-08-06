"""全连接NER训练与评估流程."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate.metrics import evaluate_predictions
from src.model.dataset import NERWindowDataset, build_label_maps
from src.model.fcnn_ner import FCNN_NER
from src.preprocess.annotate import load_bio_file
from src.utils import get_device, resolve_path, set_seed


def _predict_sequences(
    model: nn.Module,
    samples: list[list[tuple[str, str]]],
    w2v: Word2Vec,
    label2id: dict[str, int],
    id2label: dict[int, str],
    window_size: int,
    granularity: str,
    device: torch.device,
    batch_size: int = 256,
) -> list[list[str]]:
    ds = NERWindowDataset(samples, w2v, label2id, window_size, granularity)  # type: ignore[arg-type]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    pred_ids: list[int] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            pred_ids.extend(logits.argmax(dim=-1).cpu().tolist())

    # 还原为句子级标签序列
    seq_preds: list[list[str]] = [[] for _ in samples]
    for (sid, _cid, _), pid in zip(ds.meta, pred_ids):
        seq_preds[sid].append(id2label[pid])
    return seq_preds


def train_and_evaluate(
    cfg: dict[str, Any],
    granularity: Literal["char", "word"] = "char",
    num_hidden_layers: int | None = None,
    hidden_units: int | None = None,
    run_name: str = "default",
    verbose: bool = True,
) -> dict[str, Any]:
    set_seed(cfg["project"]["seed"])
    device = get_device()
    model_cfg = cfg["model"]
    labels = cfg["labels"]
    label2id, id2label = build_label_maps(labels)

    processed = resolve_path(cfg["paths"]["processed_dir"])
    train_samples = load_bio_file(processed / "train.bio")
    dev_samples = load_bio_file(processed / "dev.bio")
    test_samples = load_bio_file(processed / "test.bio")

    emb_name = "word2vec_char.model" if granularity == "char" else "word2vec_word.model"
    w2v = Word2Vec.load(str(resolve_path(Path(cfg["paths"]["embedding_dir"]) / emb_name)))

    window_size = model_cfg["window_size"]
    n_layers = num_hidden_layers if num_hidden_layers is not None else model_cfg["num_hidden_layers"]
    h_units = hidden_units if hidden_units is not None else model_cfg["hidden_units"]

    train_ds = NERWindowDataset(train_samples, w2v, label2id, window_size, granularity)
    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True)

    input_dim = (2 * window_size + 1) * w2v.vector_size
    model = FCNN_NER(
        input_dim=input_dim,
        num_labels=len(labels),
        hidden_units=h_units,
        num_hidden_layers=n_layers,
        dropout=model_cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["learning_rate"])

    best_f1 = -1.0
    best_state = None
    patience = model_cfg["early_stop_patience"]
    wait = 0
    history = []

    for epoch in range(1, model_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        iterator = tqdm(train_loader, desc=f"{run_name} epoch {epoch}", disable=not verbose)
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / max(len(train_ds), 1)

        # 验证集
        y_true = [[lab for _, lab in s] for s in dev_samples]
        y_pred = _predict_sequences(
            model, dev_samples, w2v, label2id, id2label, window_size, granularity, device
        )
        dev_metrics = evaluate_predictions(y_true, y_pred, cfg["entity_types"])
        history.append({"epoch": epoch, "loss": round(avg_loss, 4), **{k: dev_metrics[k] for k in ("accuracy", "precision", "recall", "f1")}})
        if verbose:
            print(
                f"[{run_name}] epoch={epoch} loss={avg_loss:.4f} "
                f"dev_f1={dev_metrics['f1']:.4f} acc={dev_metrics['accuracy']:.4f}"
            )

        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[{run_name}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试集评估
    y_true = [[lab for _, lab in s] for s in test_samples]
    y_pred = _predict_sequences(
        model, test_samples, w2v, label2id, id2label, window_size, granularity, device
    )
    test_metrics = evaluate_predictions(y_true, y_pred, cfg["entity_types"])

    model_dir = resolve_path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "granularity": granularity,
                "num_hidden_layers": n_layers,
                "hidden_units": h_units,
                "window_size": window_size,
                "input_dim": input_dim,
                "num_labels": len(labels),
            },
        },
        model_dir / f"{run_name}.pt",
    )

    result = {
        "run_name": run_name,
        "granularity": granularity,
        "num_hidden_layers": n_layers,
        "hidden_units": h_units,
        "best_dev_f1": best_f1,
        "test": test_metrics,
        "history": history,
    }
    if verbose:
        print(f"[{run_name}] test metrics:", test_metrics)
    return result