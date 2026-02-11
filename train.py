import torch
import torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from dataset import get_dataloaders
from core.model import MultiTaskCNN, MultiTaskLoss
from core.trainer import train_one_epoch, validate
from core.visualize import plot_history
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import EarlyStopping 
import random
import os

# --- 設定參數 ---
DATA_ROOT = 'data'
CSV_PATH = 'logs/Summary.csv'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    """
    鎖定所有隨機源，確保實驗可重現
    """
    # 1. Python 內建隨機
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. NumPy 隨機
    np.random.seed(seed)
    
    # 3. PyTorch 隨機
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果有多張 GPU
    
    # 4. 確保卷積運算的算法固定 (會稍微犧牲一點速度，但保證結果一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"隨機種子已鎖定為: {seed}")

def main():
    set_seed(42)
    print(f"使用裝置: {DEVICE}")
    
    # 1. 準備資料
    # 注意：這裡接收回傳的 all_author_labels, all_style_labels
    train_loader, val_loader, num_authors, num_styles, all_auth_lbls, all_style_lbls = get_dataloaders(
        DATA_ROOT, CSV_PATH, BATCH_SIZE
    )
    print(f"類別數量 -> 作者: {num_authors}, 書體: {num_styles}")
    
    # --- [關鍵修改] 計算權重 (Weighted Cross Entropy) ---
    print("正在針對資料不平衡計算權重...")
    
    # (1) 計算作者權重：資料越少，權重越高 (Inverse Frequency)
    # 使用 sklearn 的 'balanced' 模式: n_samples / (n_classes * np.bincount(y))
    auth_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(all_auth_lbls), 
        y=all_auth_lbls
    )
    auth_weights_tensor = torch.tensor(auth_weights, dtype=torch.float)
    
    # (2) 計算書體權重：同樣邏輯 (解決「行草」過少問題)
    style_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(all_style_lbls), 
        y=all_style_lbls
    )
    style_weights_tensor = torch.tensor(style_weights, dtype=torch.float)
    
    print(f"作者權重 (前5): {auth_weights[:5]}")
    print(f"書體權重: {style_weights}") # 你會發現行草的權重應該會特別高

    # 2. 建立模型
    model = MultiTaskCNN(num_authors, num_styles).to(DEVICE)
    
    # 3. 定義 Loss (傳入剛剛算好的權重)
    criterion = MultiTaskLoss(
        author_weights=auth_weights_tensor, 
        style_weights=style_weights_tensor,
        device=DEVICE
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    early_stopping = EarlyStopping(patience=8, verbose=True, path='weights/best_model.pth')

    # 4. 開始訓練
    history = {'train_loss': [], 'val_loss': [], 'val_acc_author': [], 'val_acc_style': []}
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # 取得當前學習率 (方便觀察)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, acc_author, acc_style = validate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc_author'].append(acc_author)
        history['val_acc_style'].append(acc_style)
        
        print(f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"Val Acc -> Author: {acc_author:.2f}% | Style: {acc_style:.2f}%")

        scheduler.step(v_loss)

        early_stopping(v_loss, model)
        
        if early_stopping.early_stop:
            print("早停機制啟動！停止訓練 (Early Stopping)")
            break

    print("載入表現最好的模型權重...")
    model.load_state_dict(torch.load('weights/best_model.pth'))

    plot_history(history)
    print("訓練完成！")

if __name__ == '__main__':
    main()