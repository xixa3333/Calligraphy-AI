import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from core.preprocess import calligraphy_preprocess

# --- 定義數據增強 (僅用於訓練集) ---
# 針對書法圖片，輕微的旋轉和位移是合理的，但不建議翻轉 (Flip)
train_transforms = transforms.Compose([
    transforms.ToPILImage(), # 轉為 PIL 以便使用 torchvision transforms
    transforms.RandomRotation(degrees=15),      # 隨機旋轉 -10~10 度
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.10)), # 隨機平移 5%
    transforms.ToTensor()    # 轉回 Tensor
])

def preprocess_image_to_array(img_path, target_size=128):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return calligraphy_preprocess(img, target_size)

class CalligraphyDataset(Dataset):
    def __init__(self, root_dir, csv_path, phase='train', img_size=128, transform=None):
        self.phase_path = os.path.join(root_dir, phase)
        self.img_size = img_size
        self.transform = transform
        self.image_paths = []
        self.author_labels = []
        self.style_labels = []

        # 讀取標籤對照
        df = pd.read_csv(csv_path)
        self.author_encoder = LabelEncoder()
        self.author_encoder.fit(df['Label'])
        self.style_encoder = LabelEncoder()
        self.style_encoder.fit(df['Style'])
        
        label_to_style = dict(zip(df['Label'], df['Style']))

        # 掃描資料夾
        for label_name in os.listdir(self.phase_path):
            label_dir = os.path.join(self.phase_path, label_name)
            if not os.path.isdir(label_dir):
                continue
            
            if label_name in self.author_encoder.classes_:
                author_idx = self.author_encoder.transform([label_name])[0]
                style_name = label_to_style.get(label_name)
                style_idx = self.style_encoder.transform([style_name])[0]
                
                # 簡單的過採樣策略：如果資料量特別少(例如行草)，可以考慮在這裡重複路徑
                # 但因為我們採用了 Weighted Loss，這裡可以保持原始分佈，讓 Loss 去處理平衡問題
                
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.image_paths.append(os.path.join(label_dir, img_name))
                        self.author_labels.append(author_idx)
                        self.style_labels.append(style_idx)

        self.num_authors = len(self.author_encoder.classes_)
        self.num_styles = len(self.style_encoder.classes_)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_arr = preprocess_image_to_array(path, self.img_size)
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0) # (1, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, self.author_labels[idx], self.style_labels[idx]

def get_dataloaders(data_root, csv_path, batch_size=32, split_ratio=0.8):
    full_dataset = CalligraphyDataset(data_root, csv_path, phase='train', transform=train_transforms)
    
    # [關鍵] 回傳所有標籤，以便 main.py 計算 Loss 權重
    all_author_labels = full_dataset.author_labels
    all_style_labels = full_dataset.style_labels
    
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 驗證集建議不要做隨機增強，因此這裡簡單切分 (若求嚴謹可建立第二個無 transform 的 dataset)
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset.num_authors, full_dataset.num_styles, all_author_labels, all_style_labels