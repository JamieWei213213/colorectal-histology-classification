"""Training and reporting, independent of MedMNIST/torchvision for offline tests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed=42):
    # Set before the first CUDA operation. Cross-version bitwise identity is not promised.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, training=False, seed=42, workers=0, device="cpu"):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training,
        num_workers=workers, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=str(device).startswith("cuda"), drop_last=False,
    )


def environment():
    versions = {}
    for name in ["torch", "torchvision", "medmnist", "numpy", "pandas", "scikit-learn", "matplotlib", "Pillow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_cuda_version": torch.version.cuda,
        "determinism": "seeded; deterministic algorithms warn on unsupported operations; hardware/version differences remain",
    }


class CompactCNN(nn.Module):
    """Three 3x3 conv blocks: 3->32->64->128; global average pooling; nine logits."""
    def __init__(self, classes=9):
        super().__init__()
        blocks = []
        in_channels = 3
        for channels in (32, 64, 128):
            blocks.extend([nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)])
            in_channels = channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(128, classes))

    def forward(self, x):
        return self.head(self.features(x))


def set_training_mode(model):
    model.train()
    # A frozen ResNet stem/layers 1-3 must also retain their pretrained BN statistics.
    if getattr(model, "freeze_early_layers", False):
        for name, child in model.named_children():
            if name not in ("layer4", "fc"):
                child.eval()


def pass_batches(model, loader, device, optimizer=None, scaler=None):
    training = optimizer is not None
    set_training_mode(model) if training else model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    truths, probabilities = [], []
    amp = str(device).startswith("cuda")
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=amp)
            labels = labels.reshape(-1).to(device=device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, enabled=amp):
                logits = model(images)
                loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; do not report this run as successful.")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach()) * labels.numel()
            total += labels.numel()
            truths.append(labels.detach().cpu().numpy())
            probabilities.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
    if total == 0:
        raise ValueError("Empty data loader")
    return total_loss / total, np.concatenate(truths), np.concatenate(probabilities)


def score_predictions(truth, probabilities, class_names):
    truth = np.asarray(truth).reshape(-1)
    probabilities = np.asarray(probabilities)
    k = len(class_names)
    if probabilities.shape != (len(truth), k):
        raise ValueError("Expected N x number-of-classes probability matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    if not np.isin(truth, np.arange(k)).all():
        raise ValueError("Labels outside class mapping")
    predicted = probabilities.argmax(axis=1)
    labels = list(range(k))
    return {
        "n_samples": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(truth, predicted, labels=labels, target_names=class_names,
                                            output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def fit_model(model, train_loader, val_loader, class_names, output, device="cpu",
              epochs=6, lr=1e-3, patience=2, metadata=None):
    """Select the best completed epoch ONLY by validation macro-F1; ties keep earlier epoch."""
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "best.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Use a fresh run directory; refusing to overwrite {checkpoint}")
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    best, stale, history = -1.0, 0, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        tick = time.perf_counter()
        train_loss, train_y, train_p = pass_batches(model, train_loader, device, optimizer, scaler)
        val_loss, val_y, val_p = pass_batches(model, val_loader, device)
        val_metrics = score_predictions(val_y, val_p, class_names)
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": float(accuracy_score(train_y, train_p.argmax(1))),
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - tick,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if val_metrics["macro_f1"] > best:
            best, stale = val_metrics["macro_f1"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": best, "class_names": class_names,
                        "metadata": metadata or {}}, checkpoint)
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    model.eval()
    summary = {"best_epoch": saved["epoch"], "best_val_macro_f1": best,
               "epochs_completed": len(history), "epochs_budget": epochs,
               "training_seconds": time.perf_counter() - start,
               "checkpoint_sha256": sha256(checkpoint),
               "parameters": sum(p.numel() for p in model.parameters()),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    save_json(output / "training_summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    table = pd.DataFrame(history)
    axes[0].plot(table.epoch, table.train_loss, label="train")
    axes[0].plot(table.epoch, table.val_loss, label="validation")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(table.epoch, table.val_macro_f1, label="validation macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle((metadata or {}).get("scope", "Training"))
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    return summary


def write_evaluation(output, truth, probabilities, class_names, images=None, scope="Final test"):
    """Write genuine measured metrics and traceable examples; never synthesize results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = score_predictions(truth, probabilities, class_names)
    metrics["scope"] = scope
    save_json(output / "metrics.json", metrics)
    pd.DataFrame(metrics["per_class"]).T.to_csv(output / "classification_report.csv")
    predicted = probabilities.argmax(1)
    frame = pd.DataFrame({"split_index": np.arange(len(truth)), "true_id": truth,
                          "predicted_id": predicted, "max_probability": probabilities.max(1)})
    for c in range(len(class_names)):
        frame[f"probability_{c}"] = probabilities[:, c]
    frame.to_csv(output / "predictions.csv", index=False)
    cm = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output / "confusion_matrix.csv")
    normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, values, title in zip(axes, [cm, normalized], ["Counts", "Fraction of each true class"]):
        chart = ax.imshow(values, cmap="Blues")
        for r in range(len(class_names)):
            for c in range(len(class_names)):
                text = str(int(values[r, c])) if title == "Counts" else f"{values[r, c]:.2f}"
                ax.text(c, r, text, ha="center", va="center", fontsize=7,
                        color="white" if values[r, c] > values.max() * 0.55 else "black")
        ax.set_xticks(range(len(class_names)), [str(i) for i in range(len(class_names))])
        ax.set_yticks(range(len(class_names)), [f"{i}: {n}" for i, n in enumerate(class_names)], fontsize=7)
        ax.set_xlabel("Predicted class ID")
        ax.set_ylabel("True class")
        ax.set_title(title)
        fig.colorbar(chart, ax=ax, fraction=0.04)
    fig.suptitle(scope)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    errors = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j]]
    errors.sort(key=lambda x: (-x[0], x[1], x[2]))
    confusion_rows = [{"true_class": class_names[i], "predicted_class": class_names[j],
                       "count": n, "fraction_of_true_class": n / max(int(cm[i].sum()), 1)}
                      for n, i, j in errors]
    pd.DataFrame(confusion_rows, columns=["true_class", "predicted_class", "count", "fraction_of_true_class"]).to_csv(output / "confusion_pairs.csv", index=False)
    lines = [f"# Error analysis — {scope}", "", "Measured directed confusions (ranked by count):", ""]
    lines += [f"- {class_names[i]} → {class_names[j]}: {n} images; {n / max(cm[i].sum(), 1):.1%} of the true class." for n, i, j in errors[:5]]
    if not errors:
        lines.append("- No errors in this evaluated sample; this does not establish clinical validity.")
    lines += ["", "These counts show where predictions failed, not why. Inspect the saved example images and original labels.",
              "Possible explanations such as downsampling, similar color/texture, or mixed tissue require visual review and higher-resolution evidence.",
              "No claim about a biological mechanism, immune response, prognosis, or patient diagnosis follows from this table.",
              "See docs/error_analysis.md for the observation-versus-hypothesis worksheet."]
    (output / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if images is not None:
        for correct, filename in [(True, "correct_examples.png"), (False, "incorrect_examples.png")]:
            selected = np.flatnonzero((truth == predicted) == correct)[:9]
            fig, axes = plt.subplots(3, 3, figsize=(11, 9))
            for ax in axes.flat:
                ax.axis("off")
            for ax, idx in zip(axes.flat, selected):
                ax.imshow(images[idx], interpolation="nearest")
                ax.set_title(f"index {idx}\nTrue: {class_names[int(truth[idx])]}\nPred: {class_names[int(predicted[idx])]}\np={probabilities[idx].max():.3f}", fontsize=7, wrap=True)
            fig.suptitle(f"{scope}: {'correct' if correct else 'incorrect'} examples (first in split order)" if len(selected) else f"{scope}: no {'correct' if correct else 'incorrect'} examples")
            fig.tight_layout()
            fig.savefig(output / filename, dpi=150)
            plt.close(fig)
    return metrics
