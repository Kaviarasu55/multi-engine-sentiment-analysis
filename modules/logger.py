import csv
import os
from datetime import datetime

LOG_FILE = 'logs/predictions_log.csv'

HEADERS = ['timestamp', 'text', 'engine', 'label', 'confidence', 'response_time']

def init_log():
    if not os.path.exists(LOG_FILE):
        os.makedirs('logs', exist_ok=True)
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)

def log_prediction(text, engine, label, confidence, response_time):
    init_log()
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            text,
            engine,
            label,
            confidence if confidence is not None else 'N/A',
            round(response_time, 4)
        ])