from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
from modules.preprocess import load_data
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

def train():
    X_train, X_test, y_train, y_test, le = load_data()

    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=3000, stratify=y_train, random_state=42
    )

    train_df = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    test_df = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

    train_df = train_df.map(tokenize, batched=True)
    test_df = test_df.map(tokenize, batched=True)

    train_df.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_df.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )

    args = TrainingArguments(
        output_dir='models/distilbert_finetuned',
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_dir='logs'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_df,
        eval_dataset=test_df
    )

    trainer.train()
    trainer.save_model('models/distilbert_finetuned')
    tokenizer.save_pretrained('models/distilbert_finetuned')
    print("Fine-tuning complete.")

if __name__ == '__main__':
    train()