---
title: Calligraphy AI
emoji: 🖋️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🖌️ Calligraphy AI - 書法風格與名家辨識系統

> **你的字跡像哪位古代名家？**
> **一款基於深度學習的書法辨識系統，無需安裝，手機/電腦打開網頁即用！**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/xixa3333/Calligraphy-AI)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)](https://www.docker.com/)

---

## 📖 專案簡介

你是否好奇過，自己隨手寫下的字，究竟有幾分**王羲之**的飄逸，或是**顏真卿**的厚重？

**Calligraphy AI** 是一個結合電腦視覺與深度學習的專案。我訓練了一個多任務卷積神經網路 (Multi-Task CNN)，能夠同時分析書法圖片的「**書體風格**」（如楷書、行書、草書）以及「**相似名家**」。

本專案已**全面升級為雲端 Web App**，並部署於 Hugging Face Spaces。現在，你不需要下載任何執行檔，只要拿出手機掃描或開啟網頁，就能立即體驗！

---

## ✨ 全新功能特色

### 1. ☁️ 雲端部署 & 跨平台支援
* **免安裝**：不用下載 `.exe`，也不用配置 Python 環境。
* **手機友善**：支援 Android / iOS 手機瀏覽器，介面自動響應 (RWD)。
* **相機整合**：手機端支援**直接呼叫相機**拍照上傳。

### 2. 🏆 Top 3 預測排名 (新功能)
不再只有單一結果！系統會列出**前三名**最相似的書法家與字體風格，並附上信心水準 (Confidence Level) 進度條，讓你更全面了解 AI 的判斷依據。

### 3. 🎨 雙模式輸入
* **✍️ 手寫畫板**：在螢幕上直接揮毫（支援壓感與觸控）。
* **📷 拍照裁剪**：拍下紙上的字跡，內建**裁剪工具**，幫助你去除雜亂背景，提升辨識準度。

---

## 🚀 快速開始

### 線上體驗 (推薦)
直接點擊下方連結進入系統：
👉 **[Calligraphy AI - 線上版](https://huggingface.co/spaces/xixa3333/Calligraphy-AI)**

### 使用步驟
1.  **選擇模式**：切換「手寫畫板」或「拍照上傳」。
2.  **輸入單字**：
    * 若是拍照，請使用內建工具將圖片**裁剪至單一個字**。
    * **⚠️ 注意：本系統針對「單字」訓練，請勿上傳整首詩或多個字。**
3.  **開始辨識**：點擊按鈕，AI 將在 1~2 秒內回傳分析結果。

---

## 🧠 核心技術與前處理

### 1. 骨架細化 (Skeletonization)
考慮到現代硬筆（原子筆、鋼筆）與傳統毛筆的粗細差異，本系統導入了 **Zhang-Suen 骨架細化演算法**：
* **歸一化**：將所有字體（無論厚重或纖細）還原為「1 像素寬」的中心骨架。
* **抗干擾**：有效消除筆畫粗細帶來的特徵雜訊，讓模型專注於「結構」與「筆順」。

### 2. 模型架構
* **Backbone**: 自定義 Multi-Task CNN (4層卷積 + Batch Normalization + ReLU)。
* **Loss Function**: 針對資料不平衡問題，採用加權損失函數 (Weighted Loss)。
* **Backend**: Flask (Python) + PyTorch。
* **Frontend**: HTML5 + Canvas API + Cropper.js。

---

## 🏆 模型效能

本模型在測試集（共 21,007 筆資料）上的表現如下：

| 辨識任務 | 準確度 (Accuracy) | 說明 |
| :--- | :--- | :--- |
| **書體風格 (Style)** | **95.28%** | 能精確區分楷書、行書、草書等 |
| **書法名家 (Author)** | **93.65%** | 在 20 位不同朝代的書法家中進行分類 |

---

## 🛠️ 本地開發指南 (For Developers)

如果你想在本地端運行或改進此專案：

### 1. 安裝依賴
```bash
git clone [https://huggingface.co/spaces/xixa3333/Calligraphy-AI](https://huggingface.co/spaces/xixa3333/Calligraphy-AI)
cd Calligraphy-AI
pip install -r requirements.txt

```

*注意：需安裝 `opencv-contrib-python-headless` 以支援細化演算法。*

### 2. 啟動伺服器

```bash
python app.py

```

預設將在 `http://127.0.0.1:7860` 啟動服務。

### 3. Docker 部署

本專案包含完整 `Dockerfile`，可直接建置映像檔：

```bash
docker build -t calligraphy-ai .
docker run -p 7860:7860 calligraphy-ai

```

---

## 📊 資料來源

本模型的訓練資料來自 Kaggle 開源資料集：

* [Chinese Calligraphy Styles by Calligraphers](https://www.kaggle.com/datasets/yuanhaowang486/chinese-calligraphy-styles-by-calligraphers)
