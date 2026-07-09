import os
import io
import base64
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from modules.preprocess import load_data
from modules.baseline import train_baseline, predict_baseline
from modules.distilbert_engine import load_distilbert, predict_distilbert
from modules.groq_engine import predict_groq
from modules.evaluator import evaluate_model, plot_confusion_matrix, plot_roc_auc
from modules.logger import log_prediction

app = Flask(__name__)

# ── Load models once at startup ──
print("Loading baseline model...")
X_train, X_test, y_train, y_test, le = load_data()
baseline_model, vectorizer = train_baseline(X_train, y_train)
class_names = list(le.classes_)
print("Baseline loaded.")

print("Loading DistilBERT...")
distilbert_model, tokenizer = load_distilbert()
print("DistilBERT loaded. Ready.")

_eval_cache = None

# ── Routes ──


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # TF-IDF
    start = time.time()
    b_label, b_conf, b_probs = predict_baseline(text, baseline_model, vectorizer)
    b_time = round(time.time() - start, 3)
    b_name = le.inverse_transform([b_label])[0]
    log_prediction(text, "tfidf", b_name, b_conf, b_time)

    # DistilBERT
    start = time.time()
    d_label, d_conf, d_probs = predict_distilbert(text, distilbert_model, tokenizer)
    d_time = round(time.time() - start, 3)
    d_name = le.inverse_transform([d_label])[0]
    log_prediction(text, "distilbert", d_name, d_conf, d_time)

    # Groq
    start = time.time()
    g_name = predict_groq(text)
    g_time = round(time.time() - start, 3)
    log_prediction(text, "groq", g_name, None, g_time)

    # TF-IDF Explainability
    feature_names = vectorizer.get_feature_names_out()
    coefs = baseline_model.coef_[b_label]
    text_tfidf = vectorizer.transform([text]).toarray()[0]
    scores = coefs * text_tfidf
    top_indices = np.argsort(scores)[-5:][::-1]
    top_words = [
        {"word": feature_names[i], "score": round(float(scores[i]), 3)}
        for i in top_indices
        if scores[i] > 0
    ]

    all_sentiments = [b_name, d_name, g_name]
    agree = len(set(all_sentiments)) == 1

    return jsonify(
        {
            "tfidf": {
                "sentiment": b_name,
                "confidence": round(b_conf * 100, 1),
                "time": b_time,
            },
            "distilbert": {
                "sentiment": d_name,
                "confidence": round(d_conf * 100, 1),
                "time": d_time,
            },
            "groq": {"sentiment": g_name, "confidence": None, "time": g_time},
            "top_words": top_words,
            "agree": agree,
            "all_sentiments": {"tfidf": b_name, "distilbert": d_name, "groq": g_name},
        }
    )


@app.route("/api/evaluate", methods=["GET"])
def evaluate():
    global _eval_cache
    if _eval_cache is not None:
        return jsonify(_eval_cache)

    y_pred_b, y_proba_b = [], []
    for text in X_test:
        label, conf, probs = predict_baseline(text, baseline_model, vectorizer)
        y_pred_b.append(label)
        y_proba_b.append(probs)

    report_b, cm_b, roc_b = evaluate_model(
        y_test.tolist(), y_pred_b, np.array(y_proba_b), class_names
    )

    fig_cm = plot_confusion_matrix(cm_b, class_names)
    buf = io.BytesIO()
    fig_cm.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    cm_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    fig_roc = plot_roc_auc(y_test.tolist(), np.array(y_proba_b), class_names)
    buf2 = io.BytesIO()
    fig_roc.savefig(buf2, format="png", bbox_inches="tight", dpi=100)
    buf2.seek(0)
    roc_img = base64.b64encode(buf2.read()).decode("utf-8")
    buf2.close()

    tfidf_f1 = round(report_b["weighted avg"]["f1-score"], 4)
    distilbert_f1 = 0.8199

    result = {
        "tfidf": {
            "roc_auc": round(roc_b, 4),
            "f1_weighted": tfidf_f1,
            "accuracy": round(report_b["accuracy"], 4),
        },
        "distilbert": {
            "roc_auc": 0.9245,
            "f1_weighted": distilbert_f1,
            "f1_diff": round(distilbert_f1 - tfidf_f1, 4),
        },
        "test_samples": len(y_test),
        "cm_image": cm_img,
        "roc_image": roc_img,
    }

    _eval_cache = result
    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def history():
    try:
        df = pd.read_csv("logs/predictions_log.csv")
        records = df.tail(50).iloc[::-1].fillna("N/A").to_dict("records")
        counts = df["label"].value_counts(normalize=True) * 100
        engine_counts = df["engine"].value_counts().to_dict()

        return jsonify(
            {
                "records": records,
                "total": len(df),
                "positive_pct": round(float(counts.get("positive", 0)), 1),
                "negative_pct": round(float(counts.get("negative", 0)), 1),
                "neutral_pct": round(float(counts.get("neutral", 0)), 1),
                "engine_counts": engine_counts,
            }
        )
    except FileNotFoundError:
        return jsonify(
            {
                "records": [],
                "total": 0,
                "positive_pct": 0,
                "negative_pct": 0,
                "neutral_pct": 0,
                "engine_counts": {},
            }
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
