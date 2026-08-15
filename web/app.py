import base64
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import LabelEncoder

from core.model import MultiTaskCNN
from core.preprocess import calligraphy_preprocess

WEB_ROOT = Path(__file__).resolve().parent
MODEL_PATH = WEB_ROOT / "artifacts" / "production" / "weight" / "best.pt"
CSV_PATH = WEB_ROOT / "artifacts" / "production" / "metadata" / "Summary.csv"
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder=str(WEB_ROOT / "templates"))

metadata = pd.read_csv(CSV_PATH)
author_encoder = LabelEncoder().fit(metadata["Label"])
style_encoder = LabelEncoder().fit(metadata["Style"])
label_to_name = dict(zip(metadata["Label"], metadata["Calligrapher Name"]))

model = MultiTaskCNN(len(author_encoder.classes_), len(style_encoder.classes_)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()


def decode_image(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("Invalid image data URL.")
    encoded = data_url.split(",", 1)[1]
    raw = base64.b64decode(encoded, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to decode the uploaded image.")
    return image


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        image = decode_image(payload.get("image"))
        processed = calligraphy_preprocess(image, IMG_SIZE)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            author_logits, style_logits = model(tensor)
            author_probs = torch.softmax(author_logits, dim=1)
            style_probs = torch.softmax(style_logits, dim=1)
            author_values, author_indices = torch.topk(author_probs, 3)
            style_values, style_indices = torch.topk(style_probs, 3)

        authors = []
        for confidence, index in zip(author_values[0], author_indices[0]):
            label = author_encoder.inverse_transform([index.item()])[0]
            authors.append(
                {
                    "name": label_to_name.get(label, label),
                    "confidence": f"{confidence.item() * 100:.2f}%",
                }
            )
        styles = [
            {
                "name": style_encoder.inverse_transform([index.item()])[0],
                "confidence": f"{confidence.item() * 100:.2f}%",
            }
            for confidence, index in zip(style_values[0], style_indices[0])
        ]
        return jsonify(success=True, top3_authors=authors, top3_styles=styles)
    except (ValueError, TypeError, KeyError) as error:
        return jsonify(success=False, error=str(error)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(success=False, error="Prediction failed."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
