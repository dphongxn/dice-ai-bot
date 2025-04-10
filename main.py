import os
import json
import random
import logging
import numpy as np
import pandas as pd
from flask import Flask, request
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram token từ biến môi trường
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Google credentials từ biến môi trường
google_creds = os.environ.get("GOOGLE_CREDS")
creds_dict = json.loads(google_creds)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# Flask App (cho UptimeRobot)
app = Flask(__name__)

@app.route('/')
def home():
    return 'Ensemble AI đang hoạt động.'

@app.route('/ping', methods=['GET', 'HEAD'])
def ping():
    return 'pong', 200

# Lấy dữ liệu train
def load_data():
    values = sheet.get_all_values()[1:]  # Bỏ header
    data = []
    labels = []
    for row in values:
        try:
            x = list(map(int, row[4:7]))
            y = 1 if sorted(map(int, row[1:4])) == sorted(x) else 0
            data.append(x)
            labels.append(y)
        except:
            continue
    return np.array(data), np.array(labels)

# Tạo mô hình Ensemble AI
def train_model():
    X, y = load_data()
    if len(X) < 10:
        return None  # Dữ liệu quá ít
    models = [
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('nb', GaussianNB()),
        ('svc', SVC(probability=True))
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X, y)
    return ensemble

# Dự đoán
def predict():
    model = train_model()
    if model is None:
        return random.sample(range(1, 7), 3)
    best = None
    best_score = -1
    for _ in range(100):
        x = [random.randint(1, 6) for _ in range(3)]
        score = model.predict_proba([x])[0][1]
        if score > best_score:
            best = x
            best_score = score
    return best

# Cập nhật kết quả vào Sheet
def update_sheet():
    values = sheet.get_all_values()
    last_row = len(values)
    row = values[-1]
    if all(cell.strip() for cell in row[4:7]) and not all(cell.strip() for cell in row[1:4]):
        new_prediction = predict()
        sheet.update(range_name=f"B{last_row + 1}:D{last_row + 1}", values=[new_prediction])
        logger.info(f"Đã cập nhật dự đoán: {new_prediction}")

# Command /start từ người dùng
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Xin chào! Đây là Ensemble AI dự đoán xúc xắc. Hãy nhập kết quả 3 viên để nhận dự đoán tiếp theo!")

# Command /predict để test nhanh
def manual_predict(update: Update, context: CallbackContext):
    result = predict()
    update.message.reply_text(f"Dự đoán từ Ensemble AI: {result}")

def main():
    updater = Updater(token=TOKEN, use_context=True, workers=4)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", manual_predict))

    # Lịch trình cập nhật Google Sheet
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_sheet, 'interval', minutes=1)
    scheduler.start()

    # Start bot
    updater.start_polling()
    app.run(host='0.0.0.0', port=10000)

if __name__ == '__main__':
    main()
