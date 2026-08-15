import os
import random
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torchvision import transforms
from calligraphy_ai.core.preprocess import calligraphy_preprocess
from calligraphy_ai.paths import PREPROCESSED_DIR

# --- 定義數據增強 (僅用於訓練集) ---
# 針對書法圖片，輕微的旋轉和位移是合理的，但不建議翻轉 (Flip)
train_transforms = transforms.Compose([
    transforms.ToPILImage(), # 轉為 PIL 以便使用 torchvision transforms
    transforms.RandomRotation(degrees=15),      # 隨機旋轉 -10~10 度
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.10)), # 隨機平移 5%
    transforms.ToTensor()    # 轉回 Tensor
])

def preprocess_image_to_array(img_path, target_size=128):
    encoded = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to decode image: {img_path}")
    return calligraphy_preprocess(img, target_size)

class CalligraphyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        csv_path,
        phase='train',
        img_size=128,
        transform=None,
        use_preprocessed=True,
    ):
        self.phase_path = os.path.join(root_dir, phase)
        self.img_size = img_size
        self.transform = transform
        self.phase = phase
        self.use_preprocessed = (
            use_preprocessed and (PREPROCESSED_DIR / ".complete").exists()
        )
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
        for label_name in sorted(os.listdir(self.phase_path)):
            label_dir = os.path.join(self.phase_path, label_name)
            if not os.path.isdir(label_dir):
                continue
            
            if label_name in self.author_encoder.classes_:
                author_idx = self.author_encoder.transform([label_name])[0]
                style_name = label_to_style.get(label_name)
                style_idx = self.style_encoder.transform([style_name])[0]
                
                # 簡單的過採樣策略：如果資料量特別少(例如行草)，可以考慮在這裡重複路徑
                # 但因為我們採用了 Weighted Loss，這裡可以保持原始分佈，讓 Loss 去處理平衡問題
                
                for img_name in sorted(os.listdir(label_dir)):
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
        if self.use_preprocessed:
            source_path = os.path.relpath(path, self.phase_path)
            cached_path = PREPROCESSED_DIR / self.phase / f"{source_path}.npy"
            try:
                img_arr = np.load(cached_path, allow_pickle=False).astype(np.float32) / 255.0
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Unable to load preprocessed image: {cached_path}") from error
        else:
            img_arr = preprocess_image_to_array(path, self.img_size)
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0) # (1, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, self.author_labels[idx], self.style_labels[idx]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    data_root,
    csv_path,
    batch_size=32,
    split_ratio=0.8,
    random_state=42,
    num_workers=0,
):
    """Create a stratified split and augment training samples only."""
    train_dataset = CalligraphyDataset(
        data_root, csv_path, phase='train', transform=train_transforms
    )
    val_dataset = CalligraphyDataset(
        data_root, csv_path, phase='train', transform=None
    )

    if train_dataset.image_paths != val_dataset.image_paths:
        raise RuntimeError("Training and validation datasets are not index-aligned.")

    # Split once, then apply the same indices to augmented and clean datasets.
    all_author_labels = np.asarray(train_dataset.author_labels)
    all_style_labels = np.asarray(train_dataset.style_labels)
    
    indices = np.arange(len(train_dataset))
    
    train_indices, val_indices = train_test_split(
        indices,
        train_size=split_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=all_author_labels,
    )
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    
    generator = torch.Generator().manual_seed(random_state)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    
    return (
        train_loader,
        val_loader,
        train_dataset.num_authors,
        train_dataset.num_styles,
        all_author_labels[train_indices],
        all_style_labels[train_indices],
    )

# 在 dataset.py 中新增
def get_full_dataset(data_root, csv_path):
    # 建立兩個 Dataset，一個用於訓練（有增強），一個用於驗證（無增強）
    train_ds = CalligraphyDataset(data_root, csv_path, transform=train_transforms)
    val_ds = CalligraphyDataset(data_root, csv_path, transform=None)
    
    # 標籤直接從 list 拿，秒開！
    all_auth_lbls = np.array(train_ds.author_labels) 
    all_style_lbls = np.array(train_ds.style_labels)
    
    num_authors = train_ds.num_authors
    num_styles = train_ds.num_styles
    
    # 回傳兩個 Dataset
    return train_ds, val_ds, num_authors, num_styles, all_auth_lbls, all_style_lbls
