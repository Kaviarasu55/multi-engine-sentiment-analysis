from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

def load_distilbert():
    tokenizer = DistilBertTokenizer.from_pretrained('KaviarasuE/sentiment-distilbert')
    model = DistilBertForSequenceClassification.from_pretrained('KaviarasuE/sentiment-distilbert')
    model.eval()
    return model, tokenizer

def predict_distilbert(text, model=None, tokenizer=None):
    if model is None:
        model, tokenizer = load_distilbert()

    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    predicted_label = int(np.argmax(probabilities))
    confidence = round(float(np.max(probabilities)), 4)

    return predicted_label, confidence, probabilities