import os
import sys
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import base64
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from core.model import MultiTaskCNN
from core.preprocess import calligraphy_preprocess

# --- [新增] 資源路徑處理函式 ---
def resource_path(relative_path):
    """ 取得資源的絕對路徑，用於 PyInstaller 打包後讀取檔案 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 會將檔案解壓縮到 sys._MEIPASS
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# --- 修改 Flask 設定，確保能找到 templates ---
app = Flask(__name__, template_folder=resource_path('templates'))

# --- 配置與載入 (全部使用 resource_path 包裹) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 修改路徑
MODEL_PATH = resource_path(os.path.join('weights', 'best_model.pth'))
CSV_PATH = resource_path(os.path.join('logs', 'Summary.csv'))
IMG_SIZE = 128

# 載入 Label 編碼器與對照表
df = pd.read_csv(CSV_PATH)
author_enc = LabelEncoder().fit(df['Label'])
style_enc = LabelEncoder().fit(df['Style'])
label_to_name = dict(zip(df['Label'], df['Calligrapher Name']))

# 載入模型
num_authors = len(author_enc.classes_)
num_styles = len(style_enc.classes_)
model = MultiTaskCNN(num_authors, num_styles).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def preprocess_web_image(img_array, target_size=128):
    return calligraphy_preprocess(img_array, target_size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        encoded_data = data.split(',')[1]
        image_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        img_processed = preprocess_web_image(img_raw, IMG_SIZE)
        img_tensor = torch.from_numpy(img_processed).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            auth_out, style_out = model(img_tensor)
            auth_probs = torch.nn.functional.softmax(auth_out, dim=1)
            style_probs = torch.nn.functional.softmax(style_out, dim=1)
            
            top3_auth_p, top3_auth_i = torch.topk(auth_probs, 3)
            auth_results = []
            for i in range(3):
                idx = top3_auth_i[0][i].item()
                auth_results.append({
                    'name': label_to_name.get(author_enc.inverse_transform([idx])[0]),
                    'confidence': f"{top3_auth_p[0][i].item()*100:.2f}%"
                })

            top3_style_p, top3_style_i = torch.topk(style_probs, 3)
            style_results = []
            for i in range(3):
                idx = top3_style_i[0][i].item()
                style_results.append({
                    'name': style_enc.inverse_transform([idx])[0],
                    'confidence': f"{top3_style_p[0][i].item()*100:.2f}%"
                })

        return jsonify({
            'success': True,
            'top3_authors': auth_results,
            'top3_styles': style_results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Hugging Face 預設埠號為 7860，且必須設為 0.0.0.0 才能對外連線
    app.run(host='0.0.0.0', port=7860)
