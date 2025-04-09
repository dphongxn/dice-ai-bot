import os
import json
from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive!"

def run():
    app.run(host='0.0.0.0', port=10000)

def keep_alive():
    t = Thread(target=run)
    t.start()

# --- BẮT ĐẦU BOT ---
import logging
import random
import gspread
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from oauth2client.service_account import ServiceAccountCredentials
from telegram.ext import Updater, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler

# Ghi file credentials.json từ biến môi trường
if os.getenv("GOOGLE_CREDENTIALS_JSON"):
    with open("credentials.json", "w") as f:
        json.dump(json.loads(os.environ["GOOGLE_CREDENTIALS_JSON"]), f)

# Thiết lập log
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Bạn cần thêm TELEGRAM_TOKEN trong Render Environment

# Kết nối Google Sheet
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# Tính tỉ lệ đúng
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

# Huấn luyện nhiều mô hình và chọn mô hình tốt nhất
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

    models = [RandomForestClassifier(), GradientBoostingClassifier(), 
              KNeighborsClassifier(), DecisionTreeClassifier(), 
              LogisticRegression(max_iter=1000), GaussianNB()]

    def fit_models(targets): return [m.fit(X, targets) for m in models]
    m1 = fit_models(y1)
    m2 = fit_models(y2)
    m3 = fit_models(y3)

    return m1, m2, m3

# Dự đoán theo mô hình trung bình
def predict_next(model_group, last_result):
    if not model_group: return random.sample(range(1, 7), 3)
    x = np.array(last_result).reshape(1, -1)
    try:
        return [int(np.mean([m.predict(x)[0] for m in group])) for group in model_group]
    except:
        return random.sample(range(1, 7), 3)

# Xử lý tin nhắn Telegram
def handle_message(update, context):
    text = update.message.text.strip()
    try:
        parts = list(map(int, text.split()))
        if len(parts) != 3:
            raise ValueError
    except:
        update.message.reply_text("Vui lòng nhập đúng 3 số từ 1 đến 6.")
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

# Khởi động bot
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True, workers=4)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    BackgroundScheduler().start()
    updater.start_polling()
    print("Bot is running...")
    updater.idle()

# Chạy
if __name__ == "__main__":
    keep_alive()
    main()
