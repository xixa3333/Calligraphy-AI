import matplotlib.pyplot as plt

from calligraphy_ai.core.plotting import configure_chinese_font


def plot_history(history, save_path=None):
    if save_path is None:
        from calligraphy_ai.paths import ARTIFACTS_DIR

        save_path = ARTIFACTS_DIR / "reports" / "training_result.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="訓練損失")
    axes[0].plot(epochs, history["val_loss"], label="驗證損失")
    axes[0].set(title="損失曲線", xlabel="訓練週期", ylabel="損失")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["val_acc_author"], label="作者準確率", color="green")
    axes[1].plot(epochs, history["val_acc_style"], label="書體準確率", color="orange")
    axes[1].set(title="驗證準確率", xlabel="訓練週期", ylabel="準確率（%）")
    axes[1].legend()
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"訓練曲線已儲存：{save_path}")
