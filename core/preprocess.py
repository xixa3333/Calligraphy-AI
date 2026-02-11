import cv2
import numpy as np

def calligraphy_preprocess(img_gray, target_size=128):
    if img_gray is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # 1. 二值化 (產生白底黑字)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 自動裁切邊界
    # 尋找非背景(黑色筆劃)的像素座標
    coords = cv2.findNonZero(cv2.bitwise_not(binary))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_gray[y:y+h, x:x+w]
        
        # 3. 等比例縮放計算
        aspect_ratio = w / h
        if aspect_ratio > 1: # 寬字
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else: # 長字
            new_h = target_size
            new_w = int(target_size * aspect_ratio)
        
        # 縮放影像
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 4. 補白邊 (建立全白底圖，將字貼在中央)
        canvas = np.full((target_size, target_size), 255, dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        img_final = canvas
    else:
        img_final = cv2.resize(img_gray, (target_size, target_size))

    # 5. 骨架細化 (加入膨脹修復斷裂)
    img_inv = cv2.bitwise_not(img_final)
    kernel = np.ones((2, 2), np.uint8) # 填補 Sample 1 等圖片的細縫
    img_inv = cv2.dilate(img_inv, kernel, iterations=1)
    
    thinned = cv2.ximgproc.thinning(img_inv, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    img_skeleton = cv2.bitwise_not(thinned)

    return img_skeleton.astype('float32') / 255.0