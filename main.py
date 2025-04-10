import os
import json
import logging
import random
import numpy as np
import gspread
import pandas as pd
from flask import Flask
from threading import Thread
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from oauth2client.service_account import ServiceAccountCredentials
from telegram.ext import Updater, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler

# ================== FLASK UPTIME ===================
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive!"

def run():
    app.run(host='0.0.0.0', port=10000)

def keep_alive():
    t = Thread(target=run)
    t.start()

# ================== GOOGLE SHEET SETUP ===================
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
keyfile_dict = json.loads(os.getenv("GOOGLE_CREDS"))
creds = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# ================== AI TRAINING ===================
def train_model(data):
    X, y1, y2, y3 = [], [], [], []
    for i in range(len(data)-1):
        try:
            x = list(map(int, data[i][4:7]))
            y = list(map(int, data[i+1][4:7]))
            X.append(x)
            y1.append(y[0])
            y2.append(y[1])
            y3.append(y[2])
        except:
            continue
    if not X: return None

    def get_ensemble(y):
        models = [
            ('gb', GradientBoostingClassifier()),
            ('rf', RandomForestClassifier()),
            ('lr', LogisticRegression(max_iter=1000)),
            ('dt', DecisionTreeClassifier()),
            ('nb', GaussianNB()),
            ('svc', SVC(probability=True))
        ]
        ensemble = VotingClassifier(estimators=models, voting='soft')
        return ensemble.fit(X, y)

    return get_ensemble(y1), get_ensemble(y2), get_ensemble(y3)

def predict_next(models, last_result):
    if not all(models) or not last_result:
        return random.sample(range(1, 7), 3)
    x = np.array(last_result).reshape(1, -1)
    try:
        return [int(m.predict(x)[0]) for m in models]
    except:
        return random.sample(range(1, 7), 3)

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

# ================== TELEGRAM BOT ===================
TOKEN = os.getenv("TELEGRAM_TOKEN")
logging.basicConfig(level=logging.INFO)

def handle_message(update, context):
    text = update.message.text.strip()
    try:
        parts = list(map(int, text.split()))
        if len(parts) != 3 or not all(1 <= n <= 6 for n in parts):
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
            sheet.update(f"E{last_row}:G{last_row}", [parts])
            try: prediction = list(map(int, last_line[1:4]))
            except: prediction = None

    models = train_model(data)
    new_prediction = predict_next(models, parts)
    sheet.update(f"B{last_row+1}:D{last_row+1}", [new_prediction])

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
    updater = Updater(TOKEN, use_context=True, workers=4)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    BackgroundScheduler().start()
    updater.start_polling()
    updater.idle()

# ================== START ===================
if __name__ == "__main__":
    keep_alive()
    main()
