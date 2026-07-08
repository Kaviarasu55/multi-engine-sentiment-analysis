import requests
import numpy as np
import os

HF_API_URL = (
    "https://api-inference.huggingface.co/models/KaviarasuE/sentiment-distilbert"
)


def predict_distilbert(text, model=None, tokenizer=None):
    response = requests.post(HF_API_URL, json={"inputs": text})
    result = response.json()

    # HF returns list of label/score dicts
    scores = {item["label"]: item["score"] for item in result[0]}

    label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    predicted_label = max(scores, key=scores.get)
    confidence = round(scores[predicted_label], 4)

    probs = np.array(
        [scores.get("LABEL_0", 0), scores.get("LABEL_1", 0), scores.get("LABEL_2", 0)]
    )

    return label_map[predicted_label], confidence, probs
