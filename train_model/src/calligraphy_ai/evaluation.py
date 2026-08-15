import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from calligraphy_ai.core.model import MODEL_ARCHITECTURE, MultiTaskCNN
from calligraphy_ai.core.plotting import configure_chinese_font
from calligraphy_ai.dataset import CalligraphyDataset
from calligraphy_ai.paths import DATA_DIR, LOGS_DIR


def expected_calibration_error(confidences, predictions, targets, bins=15):
    confidences = np.asarray(confidences)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            accuracy = (predictions[mask] == targets[mask]).mean()
            error += mask.mean() * abs(accuracy - confidences[mask].mean())
    return float(error)


def summary_metrics(targets, predictions):
    macro = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }


def optional_mean(values):
    return float(np.mean(values)) if values else None


def save_confusion_matrix(targets, predictions, labels, path, title):
    configure_chinese_font()
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(labels)))
    size = max(8, min(20, len(labels) * 0.7))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        title=title,
        xlabel="預測類別",
        ylabel="真實類別",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_checkpoint(
    run_dir,
    batch_size=128,
    device=None,
    model_class=MultiTaskCNN,
    model_architecture=MODEL_ARCHITECTURE,
):
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CalligraphyDataset(DATA_DIR, LOGS_DIR / "Summary.csv", phase="test")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    model = model_class(dataset.num_authors, dataset.num_styles).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    author_targets, author_predictions, author_confidences = [], [], []
    style_targets, style_predictions, style_confidences = [], [], []
    author_loss_sum = style_loss_sum = 0.0
    author_top3_correct = style_top3_correct = 0
    joint_correct = 0
    processed = 0
    inference_seconds = 0.0

    with torch.inference_mode():
        for images, authors, styles in tqdm(loader, desc="Testing"):
            images = images.to(device)
            authors_device = authors.to(device)
            styles_device = styles.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            author_logits, style_logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started

            author_loss_sum += F.cross_entropy(
                author_logits, authors_device, reduction="sum"
            ).item()
            style_loss_sum += F.cross_entropy(
                style_logits, styles_device, reduction="sum"
            ).item()

            author_probabilities = author_logits.softmax(dim=1)
            style_probabilities = style_logits.softmax(dim=1)
            author_top_values, author_top_indices = author_probabilities.topk(3, dim=1)
            style_top_values, style_top_indices = style_probabilities.topk(3, dim=1)

            author_top3_correct += author_top_indices.eq(authors_device[:, None]).any(dim=1).sum().item()
            style_top3_correct += style_top_indices.eq(styles_device[:, None]).any(dim=1).sum().item()
            joint_correct += (
                author_top_indices[:, 0].eq(authors_device)
                & style_top_indices[:, 0].eq(styles_device)
            ).sum().item()
            processed += images.size(0)

            author_targets.extend(authors.numpy().tolist())
            style_targets.extend(styles.numpy().tolist())
            author_predictions.extend(author_top_indices[:, 0].cpu().numpy().tolist())
            style_predictions.extend(style_top_indices[:, 0].cpu().numpy().tolist())
            author_confidences.extend(author_top_values[:, 0].cpu().numpy().tolist())
            style_confidences.extend(style_top_values[:, 0].cpu().numpy().tolist())

    author_classes = dataset.author_encoder.classes_.tolist()
    style_classes = dataset.style_encoder.classes_.tolist()
    metadata = pd.read_csv(LOGS_DIR / "Summary.csv")
    author_names = dict(zip(metadata["Label"], metadata["Calligrapher Name"]))

    author_metrics = summary_metrics(author_targets, author_predictions)
    style_metrics = summary_metrics(style_targets, style_predictions)
    author_metrics.update({
        "top3_accuracy": author_top3_correct / processed,
        "loss": author_loss_sum / processed,
        "ece_15_bins": expected_calibration_error(
            author_confidences, author_predictions, author_targets
        ),
        "mean_confidence": float(np.mean(author_confidences)),
        "correct_mean_confidence": optional_mean([
            c for c, p, t in zip(author_confidences, author_predictions, author_targets) if p == t
        ]),
        "incorrect_mean_confidence": optional_mean([
            c for c, p, t in zip(author_confidences, author_predictions, author_targets) if p != t
        ]),
    })
    style_metrics.update({
        "top3_accuracy": style_top3_correct / processed,
        "loss": style_loss_sum / processed,
        "ece_15_bins": expected_calibration_error(
            style_confidences, style_predictions, style_targets
        ),
        "mean_confidence": float(np.mean(style_confidences)),
        "correct_mean_confidence": optional_mean([
            c for c, p, t in zip(style_confidences, style_predictions, style_targets) if p == t
        ]),
        "incorrect_mean_confidence": optional_mean([
            c for c, p, t in zip(style_confidences, style_predictions, style_targets) if p != t
        ]),
    })

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "run_name": run_dir.name,
        "model_architecture": model_architecture,
        "test_samples": processed,
        "combined_loss": (author_loss_sum + style_loss_sum) / processed,
        "author": author_metrics,
        "style": style_metrics,
        "joint": {
            "accuracy": joint_correct / processed,
            "correct_samples": joint_correct,
        },
        "parameter_count": parameter_count,
        "model_size_bytes": checkpoint_path.stat().st_size,
        "inference_total_seconds": inference_seconds,
        "inference_ms_per_image": inference_seconds * 1000 / processed,
        "inference_images_per_second": processed / inference_seconds,
        "device": str(device),
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    author_report = classification_report(
        author_targets,
        author_predictions,
        labels=np.arange(len(author_classes)),
        target_names=[f"{label} ({author_names.get(label, label)})" for label in author_classes],
        output_dict=True,
        zero_division=0,
    )
    style_report = classification_report(
        style_targets,
        style_predictions,
        labels=np.arange(len(style_classes)),
        target_names=style_classes,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(author_report).transpose().to_csv(
        run_dir / "author_classification_report.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(style_report).transpose().to_csv(
        run_dir / "style_classification_report.csv", encoding="utf-8-sig"
    )

    save_confusion_matrix(
        author_targets,
        author_predictions,
        [author_names.get(label, label) for label in author_classes],
        run_dir / "author_confusion_matrix.png",
        "作者分類混淆矩陣",
    )
    save_confusion_matrix(
        style_targets,
        style_predictions,
        style_classes,
        run_dir / "style_confusion_matrix.png",
        "書體分類混淆矩陣",
    )

    return metrics
