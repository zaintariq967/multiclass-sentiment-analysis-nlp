"""Flask web app for the multi-class sentiment analyzer.

Run from the project root:   python -m app.app
"""
import json
import re
import sys
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets joblib find src.preprocess

MODEL_PATH = ROOT / "assets" / "models" / "sentiment_model.pkl"
METRICS_PATH = ROOT / "results" / "metrics.json"
MAX_CHARS = 1000
MAX_WORDS = 80

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
classes = list(model.classes_)


def explain(text: str, label: str, base_prob: float):
    """Leave-one-word-out influence: how much does each word support the verdict?"""
    parts = re.findall(r"\s+|\S+", text)
    word_idx = [i for i, p in enumerate(parts) if not p.isspace()][:MAX_WORDS]
    variants = []
    for i in word_idx:
        variants.append("".join(p for j, p in enumerate(parts) if j != i))
    probs = model.predict_proba(variants) if variants else []
    col = classes.index(label)
    influence = {i: base_prob - float(row[col]) for i, row in zip(word_idx, probs)}
    return [{"t": p, "w": round(influence.get(i, 0.0), 4), "space": p.isspace()}
            for i, p in enumerate(parts)]


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Enter some text to analyze."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text is too long. Keep it under {MAX_CHARS} characters."), 400
    proba = model.predict_proba([text])[0]
    k = int(proba.argmax())
    label = classes[k]
    return jsonify(
        label=label,
        confidence=float(proba[k]),
        probabilities={c: float(p) for c, p in zip(classes, proba)},
        tokens=explain(text, label, float(proba[k])),
    )


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify(error="Run `python -m src.train` first."), 404
    return jsonify(json.loads(METRICS_PATH.read_text()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
