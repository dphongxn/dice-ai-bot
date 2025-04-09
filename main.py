from flask import Flask
from threading import Thread
import logging
import random
import gspread
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from oauth2client.service_account import ServiceAccountCredentials
from telegram.ext import Updater, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler
import os

# === Flask setup để giữ bot sống ===
app = Flask(__name__)
@app.route("/")
def home():
    return "✅ Bot is alive!"
def run(): app.run(host="0.0.0.0", port=10000)
def keep_alive():
    t = Thread(target=run)
    t.start()

# === Cài đặt logging và biến môi trường ===
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CREDENTIALS_JSON = os.getenv("CREDENTIALS_JSON_PATH", "credentials.json")

# === Google Sheet ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_JSON, scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# === AI: Huấn luyện 6 mô hình cho từng chỉ số (3 chỉ số × 6 mô hình) ===
def train_models(data):
    X, Y = [], [[], [], []]
    for i in range(len(data) - 1):
        try:
            x = list(map(int, data[i][4:7]))
            y = list(map(int, data[i+1][4:7]))
            X.append(x)
            for j in range(3): Y[j].append(y[j])
        except: continue
    if not X: return [None]*6

    models = [
        GradientBoostingClassifier(),
        RandomForestClassifier(),
        LogisticRegression(max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        SVC(probability=True)
    ]
    trained = []
    for i in range(3):
        for model in models:
            trained.append(model.fit(X, Y[i]))
    return trained

# === Dự đoán dựa trên bỏ phiếu của 6 mô hình ===
def majority_vote(trained_models, last_result):
    if not trained_models or not last_result:
        return random.sample(range(1,7), 3)
    x = np.array(last_result).reshape(1, -1)
    results = []
    for i in range(3):
        preds = [trained_models[i + j*3].predict(x)[0] for j in range(6)]
        results.append(int(pd.Series(preds).mode()[0]))
    return results

# === Tính tỉ lệ đúng ===
def calculate_accuracy():
    data = sheet.get_all_values()
    total, correct = 0, 0
    for row in data:
        try:
            pred = list(map(int, row[1:4]))
            real = list(map(int, row[4:7]))
            if set(pred) == set(real): correct += 1
            total += 1
        except: continue
    return round(correct / total * 100, 2) if total else 0

# === Xử lý tin nhắn Telegram ===
def handle_message(update, context):
    text = update.message.text.strip()
    try:
        parts = list(map(int, text.split()))
        if len(parts) != 3: raise ValueError
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
            try: prediction = list(map(int, last_line[1:4]))
            except: prediction = None

    models = train_models(data)
    new_prediction = majority_vote(models, parts)
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

# === Chạy bot Telegram ===
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True, workers=4)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    scheduler = BackgroundScheduler()
    scheduler.start()
    updater.start_polling()
    print("Bot đang chạy...")
    updater.idle()

# === Bắt đầu ===
if __name__ == "__main__":
    keep_alive()
    main()
