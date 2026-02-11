
---
# 🖌️ Calligraphy AI - 書法風格與名家辨識系統

> **你的字跡像哪位古代名家？**
> **一款基於深度學習的書法辨識系統，無需安裝環境，下載即用！**

---

![Downloads](https://img.shields.io/github/downloads/xixa3333/Calligraphy-AI/total)

## 📖 專案簡介

你是否好奇過，自己隨手寫下的字，究竟有幾分**王羲之**的飄逸，或是**顏真卿**的厚重？

**Calligraphy AI** 是一個結合電腦視覺與深度學習的趣味專案。我訓練了一個多任務卷積神經網路 (Multi-Task CNN)，能夠同時分析書法圖片的「**書體風格**」（如楷書、行書、草書）以及「**相似名家**」。

本專案已打包為 Windows 執行檔 (`.exe`)，內建輕量級網頁伺服器，讓你在電腦上輕鬆體驗 AI 辨識的樂趣，完全不需要懂程式代碼！

## ✨ 系統特色

* **🎯 雙重辨識模型**：
    * **作者辨識**：支援王羲之、顏真卿、柳公權、米芾等 20 位歷代名家。
    * **書體辨識**：精準分辨楷書、行書、草書、行草等風格。
* **⚠️ 單字專用 (Single Character Only)**：
    * 本模型針對**單一漢字**進行訓練與優化。
    * **請務必裁切至單個字體再上傳**，上傳整句或多字圖片將無法正確辨識。
* **⚡ 懶人包設計**：
    * 提供打包好的 `.exe` 檔，**不需安裝 Python 或 PyTorch**。
    * 透過瀏覽器操作，介面直觀好上手。
* **🧠 核心技術**：
    * 使用 **PyTorch** 建構 CNN 模型。
    * 採用 **加權損失函數 (Weighted Loss)** 解決資料不平衡問題。
    * 後端採用 **Flask** 框架開發。

---

## 🏆 模型效能與測試結果

本模型經過嚴謹的訓練與測試，在測試集（共 21,007 筆資料）上的表現如下：

| 辨識任務 | 準確度 (Accuracy) | 說明 |
| :--- | :--- | :--- |
| **書體風格 (Style)** | **95.28%** | 能精確區分楷書、行書、草書等 |
| **書法名家 (Author)** | **93.65%** | 在 20 位不同朝代的書法家中進行分類 |

> 測試數據顯示，即便面對風格相近的作者（如行書名家），模型仍具有極高的辨識可靠度。

## 🚀 快速開始 (一般使用者)

如果你只是想體驗功能，請依照以下步驟：

1.  **下載檔案**：前往本專案的 **Releases** 頁面下載 `CalligraphyAI.exe`。
2.  **執行程式**：雙擊 `CalligraphyAI.exe`。
    * *注意：由於包含完整 AI 模型，首次啟動可能需要等待 10~30 秒進行解壓縮，請耐心等候。*
    * *系統會跳出一個黑色視窗 (Console)，這是伺服器後台，請勿關閉它。*
3.  **開始體驗**：程式啟動後會自動開啟預設瀏覽器（網址為 `http://127.0.0.1:5000`）。
4.  **上傳圖片**：
    * 拿出一張白紙，寫毛筆字或硬筆字。
    * 拍照或掃描後上傳，AI 將立即分析你的「書法靈魂」！

---

## 🛠️ 開發者指南 (程式碼研究)

如果你想研究原始碼或自行訓練模型，請參考以下說明。

### 1. 環境設定
```bash
git clone [https://github.com/your-username/Calligraphy-AI.git](https://github.com/your-username/Calligraphy-AI.git)
cd Calligraphy-AI
pip install -r requirements.txt

```

### 2. 專案結構

* `app.py`: Flask 網頁應用程式入口 (Web Server)。
* `train.py`: 模型訓練腳本。
* `core/`: 核心模組
* `model.py`: 定義 MultiTaskCNN 網路架構。
* `dataset.py`: 資料讀取與影像預處理。


* `templates/`: 網頁前端 HTML 模板。

### 3. 模型架構

本專案使用自定義的 CNN 架構，包含 4 層卷積層 (Convolutional Layers)，每層皆配置 **Batch Normalization** 與 **ReLU** 激活函數。輸出層分為兩個分支 (Author & Style)，並使用 **Dropout** 防止過度擬合。

### 4. 訓練模型

若你有自己的資料集，可以修改 `logs/Summary.csv` 對照表並執行：

```bash
python train.py

```

---

## 📊 資料來源

本模型的訓練資料來自 Kaggle 開源資料集，感謝貢獻者整理：

* **資料集名稱**: [Chinese Calligraphy Styles by Calligraphers](https://www.kaggle.com/datasets/yuanhaowang486/chinese-calligraphy-styles-by-calligraphers)
* **內容**: 包含 20 位書法家與多種字體風格，經清理與標準化後用於訓練。