# Calligraphy AI

以多任務卷積神經網路辨識中文書法作品的書法家與字體風格。專案包含 Flask 網頁介面、影像骨架化前處理、模型訓練、5-fold 交叉驗證與測試工具。

[Hugging Face Space](https://huggingface.co/spaces/xixa3333/Calligraphy-AI)

## 專案結構

```text
artifacts/                資料集、模型權重與訓練報表
config/requirements.txt  Python 相依套件
deploy/Dockerfile         容器設定
scripts/                  訓練、預測與檢查工具
src/calligraphy_ai/       應用程式套件
README.md                 專案說明
```

根目錄只保留 README 作為一般檔案；Git 所需的 `.gitignore`、`.gitattributes` 等隱藏檔仍保留。

## 安裝與啟動

需要 Python 3.10。PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r config\requirements.txt
$env:PYTHONPATH = "src"
python -m calligraphy_ai.web
```

開啟 <http://127.0.0.1:7860>。

## 常用指令

執行前請先設定 `PYTHONPATH=src`。

```powershell
python scripts\train.py
python scripts\train_folds.py
python scripts\predict.py
python scripts\inspect_preprocessing.py
```

## Docker

```bash
docker build -f deploy/Dockerfile -t calligraphy-ai .
docker run --rm -p 7860:7860 calligraphy-ai
```

## 模型結果

以 84,022 張影像進行 5-fold 交叉驗證：

| 任務 | 平均準確率 | 95% 信賴區間 |
| --- | ---: | ---: |
| 書法家辨識 | 93.14% | 92.75%–93.54% |
| 字體風格辨識 | 94.80% | 94.34%–95.25% |

測試集共 21,007 張影像，書法家辨識準確率為 93.51%，字體風格辨識準確率為 94.89%。

## 資料來源

[Chinese Calligraphy Styles by Calligraphers](https://www.kaggle.com/datasets/yuanhaowang486/chinese-calligraphy-styles-by-calligraphers)
