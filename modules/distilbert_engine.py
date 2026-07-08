from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

_model = None
_tokenizer = None


def predict_distilbert(text, model=None, tokenizer=None):
    global _model, _tokenizer
    if _model is None:
        _tokenizer = DistilBertTokenizer.from_pretrained(
            "KaviarasuE/sentiment-distilbert"
        )
        _model = DistilBertForSequenceClassification.from_pretrained(
            "KaviarasuE/sentiment-distilbert"
        )
        _model.eval()

    inputs = _tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = _model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = round(float(np.max(probabilities)), 4)
    return predicted_label, confidence, probabilities
