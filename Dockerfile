# 使用官方 Python 映像
FROM python:3.10-slim

# 安裝系統依賴 (OpenCV 運作所需)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有專案檔案 (包含 weights/ 和 core/)
COPY . .

# 設定 Hugging Face 預設埠號
ENV FLASK_APP=app.py
EXPOSE 7860

# 啟動 Flask (必須監聽 0.0.0.0)
CMD ["python", "app.py"]