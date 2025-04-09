from flask import Flask
from threading import Thread
import logging
import random
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram.ext import Updater, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

=== Flask app giữ cho bot sống ===

app = Flask(name)

@app.route('/') def home(): return "Bot is alive!"

def run(): app.run(host='0.0.0.0', port=8080)

def keep_alive(): t = Thread(target=run) t.start()

=== Thiết lập Telegram bot và Google Sheet ===

TELEGRAM_TOKEN = "8174193582:AAGrcq5TOTlOV9l_JVPlEV_E0o6RuI6nmuE" scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"] creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope) client = gspread.authorize(creds) sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

=== Huấn luyện 6 mô hình học máy ===

def train_models(data): X, y1, y2, y3 = [], [], [], [] for i in range(len(data) - 1): try: features = list(map(int, data[i][4:7])) targets = list(map(int, data[i+1][4:7])) if len(features) == 3 and len(targets) == 3: X.append(features) y1.append(targets[0]) y2.append(targets[1]) y3.append(targets[2]) except: continue if not X: return None

models = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    LogisticRegression(max_iter=1000),
    SVC(probability=True),
    GaussianNB()
]

trained = []
for ys in [y1, y2, y3]:
    group = []
    for model in models:
        try:
            clone = model.__class__(**model.get_params())
            clone.fit(X, ys)
            group.append(clone)
        except:
            group.append(None)
    trained.append(group)
return trained

=== Dự đoán với trung bình của nhiều mô hình ===

def predict(models, last_result): if not models or not last_result: return random.sample(range(1,7), 3) try: x = np.array(last_result).reshape(1, -1) result = [] for group in models: preds = [] for model in group: if model: preds.append(model.predict(x)[0]) if preds: result.append(int(round(np.mean(preds)))) else: result.append(random.randint(1,6)) return result except: return random.sample(range(1,7), 3)

=== Tính tỉ lệ đúng ===

def calculate_accuracy(): data = sheet.get_all_values() total, correct = 0, 0 for row in data: try: pred = list(map(int, row[1:4])) real = list(map(int, row[4:7])) if set(pred) == set(real): correct += 1 total += 1 except: continue return round(correct / total * 100, 2) if total else 0

=== Xử lý tin nhắn Telegram ===

def handle_message(update, context): text = update.message.text.strip() try: parts = list(map(int, text.split())) if len(parts) != 3: raise ValueError except: update.message.reply_text("Vui lòng nhập đúng 3 số từ 1 đến 6, cách nhau bằng dấu cách.") return

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

models = train_models(data)
new_prediction = predict(models, parts)
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

=== Khởi chạy bot ===

def main(): updater = Updater(TELEGRAM_TOKEN, use_context=True) dp = updater.dispatcher dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message)) scheduler = BackgroundScheduler() scheduler.start() updater.start_polling() print("Bot đang chạy...") updater.idle()

if name == "main": keep_alive() main()

