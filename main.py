from flask import Flask
from threading import Thread
import logging
import os
import json
import random
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram.ext import Updater, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression

# === FLASK KEEP-ALIVE ===
app = Flask(__name__)
@app.route("/")
def home():
    return "Bot is alive!"
def run():
    app.run(host='0.0.0.0', port=8080)
def keep_alive():
    t = Thread(target=run)
    t.start()

# === GOOGLE SHEET SETUP ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_json = os.environ.get("GOOGLE_CREDS")
creds_dict = json.loads(creds_json)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# === TELEGRAM SETUP ===
TELEGRAM_TOKEN = "8174193582:AAGrcq5TOTlOV9l_JVPlEV_E0o6RuI6nmuE"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# === FUNCTION: Calculate Accuracy ===
def calculate_accuracy():
    data = sheet.get_all_values()
    total, correct = 0, 0
    for row in data:
        try:
            pred = list(map(int, row[1:4]))
            real = list(map(int, row[4:7]))
            if set(pred) == set(real):
                correct += 1
            total += 1
        except:
            continue
    return round(correct / total * 100, 2) if total else 0

# === FUNCTION: Train Ensemble Model ===
def train_model(data):
    X, y1, y2, y3 = [], [], [], []
    for i in range(len(data) - 1):
        try:
            row = list(map(int, data[i][4:7]))
            next_row = list(map(int, data[i + 1][4:7]))
            X.append(row)
            y1.append(next_row[0])
            y2.append(next_row[1])
            y3.append(next_row[2])
        except:
            continue

    if not X: return None, None, None

    def build_ensemble(y):
        models = [
            ('gb', GradientBoostingClassifier()),
            ('rf', RandomForestClassifier()),
            ('ada', AdaBoostClassifier()),
            ('et', ExtraTreesClassifier()),
            ('bag', BaggingClassifier()),
            ('lr', LogisticRegression(max_iter=1000))
        ]
        ensemble = VotingClassifier(estimators=models, voting='soft')
        return ensemble.fit(X, y)

    return build_ensemble(y1), build_ensemble(y2), build_ensemble(y3)

# === FUNCTION: Predict Next ===
def predict_next(models, last_result):
    if not all(models): return random.sample(range(1,7), 3)
    x = np.array(last_result).reshape(1, -1)
    try:
        return [int(m.predict(x)[0]) for m in models]
    except:
        return random.sample(range(1,7), 3)

# === TELEGRAM HANDLER ===
def handle_message(update, context):
    text = update.message.text.strip()
    try:
        parts = list(map(int, text.split()))
        if len(parts) != 3:
            raise ValueError
    except:
        update.message.reply_text("Vui lòng nhập đúng 3 số từ 1 đến 6, cách nhau bằng dấu cách.")
        return

    data = sheet.get_all_values()
    last_row = len(data)
    prediction = None

    if last_row > 0:
        last_line = data[-1]
        if len(last_line) >= 3 and all(last_line[i] for i in range(1,4)) and (len(last_line) < 7 or not all(last_line[i] for i in range(4,7))):
            sheet.update(values=[parts], range_name=f"E{last_row}:G{last_row}")
            try:
                prediction = list(map(int, last_line[1:4]))
            except:
                prediction = None

    models = train_model(data)
    new_prediction = predict_next(models, parts)
    sheet.update(values=[new_prediction], range_name=f"B{last_row+1}:D{last_row+1}")

    if prediction:
        result = "ĐÚNG" if set(prediction) == set(parts) else "SAI"
        acc = calculate_accuracy()
        update.message.reply_text(
            f"Dự đoán trước: {prediction} → {result}\n"
            f"Tỉ lệ đúng: {acc}%\n"
            f"Dự đoán tiếp theo: {new_prediction}"
        )
    else:
        update.message.reply_text(f"Dự đoán tiếp theo: {new_prediction}")

# === MAIN ===
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    scheduler = BackgroundScheduler()
    scheduler.start()
    updater.start_polling()
    print("Bot đang chạy...")
    updater.idle()

# === RUN ===
if __name__ == "__main__":
    keep_alive()
    main()
