from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import os

def train_baseline(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    X_train_tfidf = vectorizer.fit_transform(X_train)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    model = LogisticRegression(class_weight=class_weight_dict, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/baseline_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    return model, vectorizer

def predict_baseline(text, model=None, vectorizer=None):
    if model is None:
        model = joblib.load('models/baseline_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

    text_tfidf = vectorizer.transform([text])
    predicted_label = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = round(float(np.max(probabilities)), 4)

    return predicted_label, confidence, probabilities