import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def load_data():
    df = pd.read_csv('data/dataset.csv')
    df = df[['text', 'airline_sentiment']]
    df = df.dropna()
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.strip() != '']

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['airline_sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    return X_train, X_test, y_train, y_test, le