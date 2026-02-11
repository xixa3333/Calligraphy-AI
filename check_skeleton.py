import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from core.preprocess import calligraphy_preprocess

def check_random_samples(data_root, num_samples=10):
    all_images = []
    # 遍歷所有作者資料夾收集圖片路徑
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        print("錯誤：找不到任何圖片！請確認 data/train 路徑是否正確。")
        return

    # 隨機挑選 10 張
    selected_paths = random.sample(all_images, min(num_samples, len(all_images)))
    
    plt.figure(figsize=(20, 8))
    
    for i, img_path in enumerate(selected_paths):
        # 1. 讀取原始灰階圖
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if original is None: continue
        
        # 2. 透過你統一的 preprocess 進行處理
        # 注意：為了對比，我們只取處理後的內容
        skeleton = calligraphy_preprocess(original, target_size=128)
        
        # 3. 準備顯示 (原圖 Resize 到 128x128 方便對比)
        original_res = cv2.resize(original, (128, 128))
        
        # 左邊顯示原圖，右邊顯示骨架
        combined = np.hstack((original_res, (skeleton * 255).astype(np.uint8)))
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(combined, cmap='gray')
        plt.title(f"Sample {i+1}\n{os.path.basename(os.path.dirname(img_path))}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 請確認路徑指向你的訓練資料夾
    check_random_samples('data/train')