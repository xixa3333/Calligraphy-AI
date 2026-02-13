import torch
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from core.preprocess import calligraphy_preprocess

# 引用你的模型架構
from core.model import MultiTaskCNN

# --- 設定參數 ---
csv_path = 'logs/Summary.csv'
model_path = 'weights/best_model.pth'  # 訓練好的權重檔
test_dir = 'data/test'         # 測試集資料夾 (如果有的話)
img_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. 準備工具：標籤還原器與前處理
# ==========================================
def load_encoders(csv_path):
    """讀取 CSV 以重建 LabelEncoder，確保與訓練時一致"""
    df = pd.read_csv(csv_path)
    
    # 建立作者編碼器
    author_encoder = LabelEncoder()
    author_encoder.fit(df['Label'])
    
    # 建立風格編碼器
    style_encoder = LabelEncoder()
    style_encoder.fit(df['Style'])
    
    # 建立 Label 到 Style 的對照字典 (用於驗證正確性)
    label_to_style = dict(zip(df['Label'], df['Style']))
    label_to_name = dict(zip(df['Label'], df['Calligrapher Name'])) # 英文Label轉中文名
    
    return author_encoder, style_encoder, label_to_name, label_to_style

def preprocess_image(img_path, target_size=128):
    """與訓練時完全相同的前處理：灰階 -> Resize -> Otsu -> Normalize"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = calligraphy_preprocess(img, target_size)
    
    # 轉 Tensor: (H, W) -> (1, 1, H, W)  [Batch, Channel, H, W]
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

# ==========================================
# 2. 核心功能：預測與評估
# ==========================================
def predict_single_image(model, img_path, author_encoder, style_encoder, label_to_name):
    """預測單張圖片"""
    img_tensor = preprocess_image(img_path)
    if img_tensor is None:
        print(f"錯誤：無法讀取圖片 {img_path}")
        return

    img_tensor = img_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        author_logits, style_logits = model(img_tensor)
        
        # 取得機率最高的 index
        author_idx = torch.argmax(author_logits, dim=1).item()
        style_idx = torch.argmax(style_logits, dim=1).item()
        
        # 取得信心分數 (Softmax)
        author_conf = torch.nn.functional.softmax(author_logits, dim=1)[0][author_idx].item()
        style_conf = torch.nn.functional.softmax(style_logits, dim=1)[0][style_idx].item()

    # 轉回文字標籤
    pred_author_label = author_encoder.inverse_transform([author_idx])[0]
    pred_style = style_encoder.inverse_transform([style_idx])[0]
    pred_author_name = label_to_name.get(pred_author_label, pred_author_label)

    print(f"--- 預測結果 [{os.path.basename(img_path)}] ---")
    print(f"作者: {pred_author_name} ({pred_author_label}) | 信心度: {author_conf:.2%}")
    print(f"書體: {pred_style} | 信心度: {style_conf:.2%}")
    
    # 顯示圖片 (Optional)
    plt.imshow(cv2.imread(img_path), cmap='gray')
    plt.title(f"Pred: {pred_author_label} / {pred_style}") # 用英文顯示避免亂碼
    plt.axis('off')
    plt.show()

def evaluate_test_set(model, test_dir, author_encoder, style_encoder, label_to_style):
    """評估整個 Test 資料夾的準確率"""
    print(f"\n正在評估測試集: {test_dir} ...")
    
    true_authors = []
    pred_authors = []
    true_styles = []
    pred_styles = []
    
    # 遍歷 data/test 資料夾
    # 結構假設: data/test/{author_label}/{image.jpg}
    
    valid_folders = [d for d in os.listdir(test_dir) if d in author_encoder.classes_]
    
    for label_name in tqdm(valid_folders):
        folder_path = os.path.join(test_dir, label_name)
        
        # 取得真實標籤 (Ground Truth)
        gt_author_idx = author_encoder.transform([label_name])[0]
        gt_style_name = label_to_style.get(label_name)
        gt_style_idx = style_encoder.transform([gt_style_name])[0]
        
        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(folder_path, img_name)
            img_tensor = preprocess_image(img_path)
            
            if img_tensor is None: continue
            
            img_tensor = img_tensor.to(device)
            
            # 預測
            model.eval()
            with torch.no_grad():
                a_out, s_out = model(img_tensor)
                p_a = torch.argmax(a_out, 1).item()
                p_s = torch.argmax(s_out, 1).item()
                
            true_authors.append(gt_author_idx)
            pred_authors.append(p_a)
            true_styles.append(gt_style_idx)
            pred_styles.append(p_s)

    # 計算準確率
    acc_author = accuracy_score(true_authors, pred_authors)
    acc_style = accuracy_score(true_styles, pred_styles)
    
    print(f"\n========== 最終測試報告 ==========")
    print(f"測試集總樣本數: {len(true_authors)}")
    print(f"作者辨識準確率 (Author Accuracy): {acc_author:.2%}")
    print(f"書體辨識準確率 (Style Accuracy):  {acc_style:.2%}")
    
    # 詳細報表 (可選)
    target_names = [str(c) for c in author_encoder.classes_]
    print("\n--- 作者分類詳細報表 ---")
    print(classification_report(true_authors, pred_authors, target_names=target_names))

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == '__main__':
    # A. 載入標籤
    print("載入標籤編碼器...")
    author_enc, style_enc, lbl_to_name, lbl_to_style = load_encoders(csv_path)
    
    num_authors = len(author_enc.classes_)
    num_styles = len(style_enc.classes_)
    print(f"作者數: {num_authors}, 書體數: {num_styles}")

    # B. 載入模型架構與權重
    print(f"載入模型: {model_path}")
    model = MultiTaskCNN(num_authors, num_styles).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("權重載入成功！")
    else:
        print("錯誤：找不到權重檔，請先執行訓練！")
        exit()

    # --- 功能選擇 ---
    
    # 模式 1: 測試單張圖片 (請修改這裡的路徑來測試你的圖片)
    # test_img = 'data/test/wxz/some_image.jpg' 
    # if os.path.exists(test_img):
    #     predict_single_image(model, test_img, author_enc, style_enc, lbl_to_name)
    
    # 模式 2: 評估整個測試集 (如果有 data/test 資料夾)
    if os.path.exists(test_dir):
        evaluate_test_set(model, test_dir, author_enc, style_enc, lbl_to_style)
    else:
        print(f"找不到測試資料夾 {test_dir}，跳過批次評估。")