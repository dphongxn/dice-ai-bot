from flask import Flask
from threading import Thread

# === Flask server để giữ bot luôn chạy ===
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is alive and ready!"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

# === Import AI và bot ===
import logging
import random
import gspread
import numpy as np
import pandas as pd

from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from telegram.ext import Updater, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler

# === Kết nối Google Sheets ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# === Cấu hình bot ===
TELEGRAM_TOKEN = "8174193582:AAGrcq5TOTlOV9l_JVPlEV_E0o6RuI6nmuE"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

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

def train_model(data):
    X, y1, y2, y3 = [], [], [], []
    for i in range(len(data) - 1):
        try:
            row = data[i]
            next_row = data[i + 1]
            x = list(map(int, row[4:7]))
            y = list(map(int, next_row[4:7]))
            X.append(x)
            y1.append(y[0])
            y2.append(y[1])
            y3.append(y[2])
        except:
            continue
    if not X:
        return None

    def get_ensemble(y):
        models = [
            ('rf', RandomForestClassifier()),
            ('gb', GradientBoostingClassifier()),
            ('et', ExtraTreesClassifier()),
            ('lr', LogisticRegression(max_iter=1000)),
            ('svm', SVC(probability=True)),
            ('knn', KNeighborsClassifier())
        ]
        ensemble = VotingClassifier(estimators=models, voting='soft')
        return ensemble.fit(X, y)

    return get_ensemble(y1), get_ensemble(y2), get_ensemble(y3)

def predict_next(models, last_result):
    if not models or not last_result:
        return random.sample(range(1, 7), 3)
    x = np.array(last_result).reshape(1, -1)
    try:
        return [int(m.predict(x)[0]) for m in models]
    except:
        return random.sample(range(1, 7), 3)

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
        if len(last_line) >= 3 and all(last_line[i] for i in range(1, 4)) and (len(last_line) < 7 or not all(last_line[i] for i in range(4, 7))):
            sheet.update(values=[parts], range_name=f"E{last_row}:G{last_row}")
            try:
                prediction = list(map(int, last_line[1:4]))
            except:
                prediction = None

    models = train_model(data)
    new_prediction = predict_next(models, parts)
    sheet.update(values=[new_prediction], range_name=f"B{last_row + 1}:D{last_row + 1}")

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

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    scheduler = BackgroundScheduler()
    scheduler.start()
    updater.start_polling()
    print("Bot đang chạy...")
    updater.idle()

# === CHẠY CHÍNH ===
if __name__ == "__main__":
    keep_alive()
    main()
