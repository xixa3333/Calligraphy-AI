import torch
import torch.optim as optim
import numpy as np
import scipy.stats as stats # 用於計算信賴區間
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler

# 保持你原本的引用
from dataset import get_dataloaders, CalligraphyDataset,get_full_dataset # 假設你 dataset.py 裡有這個類別
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
K_FOLDS = 5 # 新增：設定 5 折
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"隨機種子已鎖定為: {seed}")

def calculate_confidence_interval(data, confidence=0.95):
    """計算平均值、變異數與 95% 信賴區間"""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data, ddof=1) # 樣本變異數
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, var, (mean - h, mean + h)

def main():
    set_seed(42)
    
    # 1. 準備完整資料集 (移除原本重複的 full_dataset 定義)
    # 直接使用優化過的 get_full_dataset 取得兩個 Dataset 對象
    train_ds, val_ds, num_authors, num_styles, all_auth_lbls, all_style_lbls = get_full_dataset(DATA_ROOT, CSV_PATH)
    
    # 2. 初始化 5-Fold 分割器
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # 存放結果的容器
    fold_auth_accs = []
    fold_style_accs = []
    best_overall_avg_acc = 0.0
    best_fold_idx = -1

    print(f"\n開始執行 {K_FOLDS}-Fold 交叉驗證 (資料總數: {len(train_ds)})")

    # 3. 進入 5-Fold 迴圈
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_auth_lbls)), all_auth_lbls)):
        print(f"\n" + "="*30)
        print(f"FOLD {fold + 1} / {K_FOLDS}")
        print("="*30)

        # 3.1 建立當前 Fold 的 DataLoader (確保訓練有增強，驗證無增強)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # 這裡的 num_workers 建議根據你的 CPU 核心設定 (例如 2 或 4) 以加速讀取
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0)

        # 3.2 重新計算權重 (使用已經提取好的 all_auth_lbls)
        curr_train_auth = all_auth_lbls[train_idx]
        curr_train_style = all_style_lbls[train_idx]
        
        auth_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(curr_train_auth), y=curr_train_auth), dtype=torch.float)
        style_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(curr_train_style), y=curr_train_style), dtype=torch.float)

        # 3.3 初始化模型與損失函數
        model = MultiTaskCNN(num_authors, num_styles).to(DEVICE)
        criterion = MultiTaskLoss(author_weights=auth_weights, style_weights=style_weights, device=DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        fold_model_path = f'weights/best_model_fold_{fold+1}.pth'
        # 建議 EarlyStopping 的 path 也要隨 fold 改變
        early_stopping = EarlyStopping(patience=8, verbose=True, path=fold_model_path)

        # 3.4 訓練 Fold
        best_fold_auth_acc = 0.0
        best_fold_style_acc = 0.0

        for epoch in range(EPOCHS):
            # 使用我們定義好的 train_loader 和 val_loader
            t_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            v_loss, acc_author, acc_style = validate(model, val_loader, criterion, DEVICE)
            
            scheduler.step(v_loss)
            early_stopping(v_loss, model)

            if acc_author > best_fold_auth_acc: best_fold_auth_acc = acc_author
            if acc_style > best_fold_style_acc: best_fold_style_acc = acc_style

            if early_stopping.early_stop: break

        fold_auth_accs.append(best_fold_auth_acc)
        fold_style_accs.append(best_fold_style_acc)
        
        current_avg = (best_fold_auth_acc + best_fold_style_acc) / 2
        if current_avg > best_overall_avg_acc:
            best_overall_avg_acc = current_avg
            best_fold_idx = fold + 1
            # 儲存全域最佳模型
            torch.save(model.state_dict(), 'weights/best_model.pth')

    # 4. 統計運算
    a_mean, a_var, a_ci = calculate_confidence_interval(fold_auth_accs)
    s_mean, s_var, s_ci = calculate_confidence_interval(fold_style_accs)

    # 5. 輸出報告
    print("\n" + "★"*40)
    print(f"🎉 5-Fold 交叉驗證最終統計報告")
    print("★"*40)
    print(f"最佳 Fold 模型: 第 {best_fold_idx} 組 (已存為 weights/best_model.pth)")
    print("-" * 20)
    print(f"[作者辨識 Author Accuracy]")
    print(f"  平均值: {a_mean:.2f}%")
    print(f"  變異數: {a_var:.4f}")
    print(f"  95% 信賴區間: [{a_ci[0]:.2f}%, {a_ci[1]:.2f}%]")
    print("-" * 20)
    print(f"[書體辨識 Style Accuracy]")
    print(f"  平均值: {s_mean:.2f}%")
    print(f"  變異數: {s_var:.4f}")
    print(f"  95% 信賴區間: [{s_ci[0]:.2f}%, {s_ci[1]:.2f}%]")
    print("★"*40)

if __name__ == '__main__':
    # 確保資料夾存在
    os.makedirs('weights', exist_ok=True)
    main()