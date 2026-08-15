---
title: Calligraphy AI
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.10"
app_port: 7860
pinned: false
---

# Calligraphy AI

中文書法家與書體的多任務影像分類專案。專案分為模型訓練與網站部署兩個獨立區塊。

目前網站正式部署模型為 **Shared Linear 5-fold 的 Fold 5 `best.pt`**。Fold 5 依最低 validation loss 選出，未使用 test set 反向挑選模型。

## 正式模型表現

固定 Kaggle test set 共 21,007 張圖片：

| 指標 | Fold 5 |
| --- | ---: |
| Author Accuracy | 93.12% |
| Author Top-3 Accuracy | 98.56% |
| Author Macro F1 | 0.9078 |
| Style Accuracy | 95.09% |
| Style Top-3 Accuracy | 99.70% |
| Style Macro F1 | 0.9237 |
| Joint Accuracy | 91.53% |
| Combined Loss | 0.3729 |

五折平均為 Author Accuracy 92.95%、Style Accuracy 94.87%、Joint Accuracy 91.26%。完整分析請參考 [5-fold 穩定性評估](train_model/docs/5fold評判穩定性.md)與[三模型比較](train_model/docs/20260814_model_comparison.md)。

## 目錄

```text
train_model/
├── artifacts/       資料集、前處理快取、訓練結果與歷史模型
├── scripts/         前處理、訓練、評估及比較工具
├── src/             訓練用 Python 套件
└── requirements.txt 訓練環境相依套件

web/
├── artifacts/       網站正式模型與標籤資料
├── templates/       網頁模板
├── app.py           Flask 入口
├── core/            Shared Linear 模型與推論影像前處理
└── requirements.txt 網站環境相依套件
```

根目錄只保留說明文件，以及 Git、Codex 和 Hugging Face Docker Space 必須使用的隱藏檔或設定檔。`Dockerfile` 只會打包 `web/`，不會把訓練資料與實驗模型部署到網站。

## 訓練

```powershell
cd train_model
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python scripts\train.py
```

執行可中斷續訓的 Shared Linear 5-fold：

```powershell
python scripts\train_folds.py
```

其他工具：

```powershell
python scripts\preprocess_dataset.py
python scripts\evaluate.py
python scripts\compare_runs.py
```

## 啟動網站

```powershell
cd web
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

網站使用 `web/artifacts/production/weight/best.pt`（目前為 Fold 5），並從 `web/artifacts/production/metadata/Summary.csv` 讀取分類標籤。

## Docker

請在專案根目錄執行：

```bash
docker build -t calligraphy-ai .
docker run --rm -p 7860:7860 calligraphy-ai
```

資料集來源：[Chinese Calligraphy Styles by Calligraphers](https://www.kaggle.com/datasets/yuanhaowang486/chinese-calligraphy-styles-by-calligraphers)

線上展示：[Hugging Face Space](https://huggingface.co/spaces/xixa3333/Calligraphy-AI)
