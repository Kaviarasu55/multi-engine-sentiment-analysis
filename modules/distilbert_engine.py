import requests
import numpy as np
import time
import os

HF_API_URL = (
    "https://api-inference.huggingface.co/models/KaviarasuE/sentiment-distilbert"
)


def predict_distilbert(text, model=None, tokenizer=None):
    headers={"Authorization": f"Bearer{os.getenv('HF_TOKEN')}"}
    for attempt in range(3):
        response = requests.post(HF_API_URL, json={"inputs": text})
        result = response.json()

        # Handle HF cold start
        if "error" in result:
            if attempt < 2:
                time.sleep(10)
                continue
            return 0, 0.0, np.array([0.33, 0.33, 0.34])

        scores = {item["label"]: item["score"] for item in result[0]}
        label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
        predicted_label = max(scores, key=scores.get)
        confidence = round(scores[predicted_label], 4)
        probs = np.array(
            [
                scores.get("LABEL_0", 0),
                scores.get("LABEL_1", 0),
                scores.get("LABEL_2", 0),
            ]
        )

        return label_map[predicted_label], confidence, probs