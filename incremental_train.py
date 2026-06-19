from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from modules.preprocess import load_data
from sklearn.model_selection import train_test_split
import torch

MODEL_PATH = 'models/distilbert_finetuned'

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

def incremental_train(X_chunk, y_chunk, X_test, y_test):
    train_df = Dataset.from_dict({
        'text': X_chunk.tolist(),
        'label': y_chunk.tolist()
    })
    test_df = Dataset.from_dict({
        'text': X_test.tolist(),
        'label': y_test.tolist()
    })

    train_df = train_df.map(tokenize, batched=True)
    test_df = test_df.map(tokenize, batched=True)

    train_df.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_df.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
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
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("Incremental round complete. Model updated.")

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, le = load_data()

    # Remove first 3000 already trained — train on remaining
    X_remaining = X_train.iloc[3000:]
    y_remaining = y_train.iloc[3000:]

    # Split remaining into 3 chunks of ~2900 each
    chunk_size = len(X_remaining) // 3

    for i in range(3):
        start = i * chunk_size
        end = start + chunk_size if i < 2 else len(X_remaining)
        X_chunk = X_remaining.iloc[start:end]
        y_chunk = y_remaining.iloc[start:end]
        print(f"\n--- Round {i+2}: samples {3000+start} to {3000+end} ---")
        incremental_train(X_chunk, y_chunk, X_test, y_test)

    print("\nAll incremental rounds complete!")