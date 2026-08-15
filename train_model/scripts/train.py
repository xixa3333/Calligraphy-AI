import os
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from calligraphy_ai.core.model import MODEL_ARCHITECTURE, MultiTaskCNN, MultiTaskLoss
from calligraphy_ai.core.trainer import train_one_epoch, validate
from calligraphy_ai.core.visualize import plot_history
from calligraphy_ai.dataset import get_dataloaders
from calligraphy_ai.experiment import (
    BEST_MODEL_PATH,
    LAST_CHECKPOINT_PATH,
    RUN_DIR,
    RUN_NAME,
    TRAINING_PLOT_PATH,
)
from calligraphy_ai.paths import DATA_DIR, LOGS_DIR
from calligraphy_ai.utils.utils import EarlyStopping, set_seed

BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 0.001
EPOCHS = 50
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_last_checkpoint(epoch, model, optimizer, scheduler, scaler, history, stopper):
    checkpoint = {
        "epoch": epoch,
        "run_name": RUN_NAME,
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
    temporary = LAST_CHECKPOINT_PATH.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, LAST_CHECKPOINT_PATH)


def resume_from_last(model, optimizer, scheduler, scaler, stopper):
    empty_history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc_author": [],
        "val_acc_style": [],
    }
    if not LAST_CHECKPOINT_PATH.exists():
        return 0, empty_history

    checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if checkpoint.get("run_name") != RUN_NAME:
        raise ValueError(f"Checkpoint does not belong to run {RUN_NAME!r}.")
    if checkpoint.get("model_architecture") != MODEL_ARCHITECTURE:
        raise ValueError(f"Checkpoint architecture does not match {MODEL_ARCHITECTURE!r}.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    stopper.load_state_dict(checkpoint["early_stopping_state"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_random_state"]])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming at epoch {start_epoch + 1}/{EPOCHS}")
    return start_epoch, checkpoint["history"]


def balanced_weights(labels):
    return torch.tensor(
        compute_class_weight("balanced", classes=np.unique(labels), y=labels),
        dtype=torch.float,
    )


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    print(f"Training run: {RUN_NAME}")
    print(f"Run directory: {RUN_DIR}")
    print(f"Device: {DEVICE}")

    train_loader, val_loader, num_authors, num_styles, author_labels, style_labels = (
        get_dataloaders(
            DATA_DIR,
            LOGS_DIR / "Summary.csv",
            batch_size=BATCH_SIZE,
            random_state=SEED,
            num_workers=NUM_WORKERS,
        )
    )
    model = MultiTaskCNN(num_authors, num_styles).to(DEVICE)
    criterion = MultiTaskLoss(
        balanced_weights(author_labels), balanced_weights(style_labels), DEVICE
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    stopper = EarlyStopping(patience=8, verbose=True, path=BEST_MODEL_PATH)
    start_epoch, history = resume_from_last(model, optimizer, scheduler, scaler, stopper)

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler=scaler
        )
        val_loss, author_acc, style_acc = validate(model, val_loader, criterion, DEVICE)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc_author"].append(author_acc)
        history["val_acc_style"].append(style_acc)
        scheduler.step(val_loss)
        stopper(val_loss, model)
        save_last_checkpoint(epoch, model, optimizer, scheduler, scaler, history, stopper)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, author={author_acc:.2f}%, style={style_acc:.2f}%"
        )
        if stopper.early_stop:
            break

    plot_history(history, save_path=TRAINING_PLOT_PATH)
    from calligraphy_ai.evaluation import evaluate_checkpoint

    metrics = evaluate_checkpoint(RUN_DIR, device=DEVICE)
    print(f"Test metrics saved to {RUN_DIR / 'test_metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()
