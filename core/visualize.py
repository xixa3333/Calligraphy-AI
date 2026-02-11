import matplotlib.pyplot as plt

def plot_history(history, save_path='logs/training_result.png'):
    """
    輸入 history 字典，輸出折線圖
    history keys: 'train_loss', 'val_loss', 'val_acc_author', 'val_acc_style'
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc_author'], label='Author Acc', color='green')
    plt.plot(epochs, history['val_acc_style'], label='Style Acc', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"圖表已儲存至: {save_path}")
    # plt.show() # 如果是在 Server 上跑可以註解掉這行