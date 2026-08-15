import json
import os
import random
import shutil
from datetime import datetime

import numpy as np
import scipy.stats as stats
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from calligraphy_ai.core.model import MODEL_ARCHITECTURE, MultiTaskCNN, MultiTaskLoss
from calligraphy_ai.core.trainer import train_one_epoch, validate
from calligraphy_ai.core.visualize import plot_history
from calligraphy_ai.dataset import get_full_dataset, seed_worker
from calligraphy_ai.evaluation import evaluate_checkpoint
from calligraphy_ai.paths import DATA_DIR, LOGS_DIR, RUNS_DIR
from calligraphy_ai.utils.utils import EarlyStopping, set_seed

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 0.001
EPOCHS = 50
K_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, [mean, mean]
    margin = float(stats.sem(values) * stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    return mean, [mean - margin, mean + margin]


def flatten_numeric_metrics(data, prefix=""):
    """Flatten nested metric dictionaries into dotted numeric keys."""
    flattened = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_numeric_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[name] = float(value)
    return flattened


def aggregate_test_metrics(fold_metrics):
    """Return mean, sample standard deviation and 95% t CI for every metric."""
    flattened = [flatten_numeric_metrics(metrics) for metrics in fold_metrics]
    common_keys = sorted(set.intersection(*(set(item) for item in flattened)))
    aggregate = {}
    for key in common_keys:
        values = np.asarray([item[key] for item in flattened], dtype=float)
        mean, interval = confidence_interval(values)
        aggregate[key] = {
            "values": values.tolist(),
            "mean": mean,
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "95_ci_lower": interval[0],
            "95_ci_upper": interval[1],
        }
    return aggregate


def save_last(path, epoch, model, optimizer, scheduler, scaler, history, stopper):
    state = {
        "epoch": epoch,
        "model_architecture": MODEL_ARCHITECTURE,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "early_stopping_state": stopper.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def resume(path, model, optimizer, scheduler, scaler, stopper):
    empty = {"train_loss": [], "val_loss": [], "val_acc_author": [], "val_acc_style": []}
    if not path.exists():
        return 0, empty
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    if state.get("model_architecture") != MODEL_ARCHITECTURE:
        raise ValueError(f"Checkpoint architecture does not match {MODEL_ARCHITECTURE}.")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state.get("scaler_state_dict", {}))
    stopper.load_state_dict(state["early_stopping_state"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_random_state"]])
    return state["epoch"] + 1, state["history"]


def main():
    set_seed(SEED)
    run_date = os.environ.get("CALLIGRAPHY_RUN_DATE", datetime.now().strftime("%Y%m%d"))
    version = os.environ.get("CALLIGRAPHY_RUN_VERSION", "v1")
    run_dir = RUNS_DIR / f"{run_date}_{MODEL_ARCHITECTURE}_5fold_{version}"
    run_dir.mkdir(parents=True, exist_ok=True)

    augmented, clean, num_authors, num_styles, author_labels, style_labels = get_full_dataset(
        DATA_DIR, LOGS_DIR / "Summary.csv"
    )
    splitter = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_number, (train_indices, val_indices) in enumerate(
        splitter.split(np.zeros(len(author_labels)), author_labels), start=1
    ):
        fold_dir = run_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_path = fold_dir / "best.pt"
        last_path = fold_dir / "last.pt"
        generator = torch.Generator().manual_seed(SEED + fold_number)
        options = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": NUM_WORKERS > 0,
            "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
        }
        train_loader = DataLoader(
            Subset(augmented, train_indices), shuffle=True, generator=generator, **options
        )
        val_loader = DataLoader(Subset(clean, val_indices), shuffle=False, **options)

        train_authors = author_labels[train_indices]
        train_styles = style_labels[train_indices]
        author_weights = torch.tensor(
            compute_class_weight("balanced", classes=np.unique(train_authors), y=train_authors),
            dtype=torch.float,
        )
        style_weights = torch.tensor(
            compute_class_weight("balanced", classes=np.unique(train_styles), y=train_styles),
            dtype=torch.float,
        )
        model = MultiTaskCNN(num_authors, num_styles).to(DEVICE)
        criterion = MultiTaskLoss(author_weights, style_weights, DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        stopper = EarlyStopping(patience=8, verbose=True, path=best_path)
        start_epoch, history = resume(last_path, model, optimizer, scheduler, scaler, stopper)

        for epoch in range(start_epoch, EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
            val_loss, author_acc, style_acc = validate(model, val_loader, criterion, DEVICE)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc_author"].append(author_acc)
            history["val_acc_style"].append(style_acc)
            scheduler.step(val_loss)
            stopper(val_loss, model)
            save_last(last_path, epoch, model, optimizer, scheduler, scaler, history, stopper)
            print(
                f"Fold {fold_number} epoch {epoch + 1}/{EPOCHS}: "
                f"val_loss={val_loss:.4f}, author={author_acc:.2f}%, style={style_acc:.2f}%"
            )
            if stopper.early_stop:
                break

        plot_history(history, fold_dir / "training_result.png")
        best_epoch = int(np.argmin(history["val_loss"]))
        fold_results.append(
            {
                "fold": fold_number,
                "best_epoch": best_epoch + 1,
                "val_loss": history["val_loss"][best_epoch],
                "author_accuracy": history["val_acc_author"][best_epoch],
                "style_accuracy": history["val_acc_style"][best_epoch],
                "best_model": str(best_path),
            }
        )

    author_mean, author_ci = confidence_interval([x["author_accuracy"] for x in fold_results])
    style_mean, style_ci = confidence_interval([x["style_accuracy"] for x in fold_results])
    best_fold = min(fold_results, key=lambda item: item["val_loss"])
    shutil.copy2(best_fold["best_model"], run_dir / "best.pt")
    fold_test_metrics = []
    for result in fold_results:
        fold_dir = run_dir / f"fold_{result['fold']}"
        print(f"Testing fold {result['fold']} on the fixed Kaggle test set...")
        fold_test_metrics.append(evaluate_checkpoint(fold_dir, device=DEVICE))
    summary = {
        "model_architecture": MODEL_ARCHITECTURE,
        "folds": fold_results,
        "author_accuracy_mean": author_mean,
        "author_accuracy_95_ci": author_ci,
        "style_accuracy_mean": style_mean,
        "style_accuracy_95_ci": style_ci,
        "selected_fold": best_fold["fold"],
        "fold_test_metrics": fold_test_metrics,
        "test_aggregate": aggregate_test_metrics(fold_test_metrics),
    }
    (run_dir / "cross_validation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
