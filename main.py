import os
import json
import logging
import pandas as pd
import numpy as np
from io import StringIO
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === CẤU HÌNH ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_CREDS = os.environ.get("GOOGLE_CREDS")  # Biến chứa nội dung JSON của credentials

SHEET_NAME = "App dự đoán beta"
SHEET_ID = "1mf3"  # phần mã Google Sheet của bạn

# === GOOGLE SHEET ===
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds_dict = json.loads(GOOGLE_CREDS)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

# === AI MODEL ===
def train_model():
    data = pd.DataFrame(sheet.get_all_records())
    if data.shape[0] < 10:
        return None
    
    X = data[['E', 'F', 'G']].values  # thực tế
    y = (data[['B', 'C', 'D']].values == data[['E', 'F', 'G']].values).sum(axis=1) >= 2

    models = [
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('svc', SVC(probability=True)),
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier())
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X, y)
    return ensemble

model = train_model()

def predict():
    if not model:
        return [np.random.randint(1, 7) for _ in range(3)]
    
    last_rows = pd.DataFrame(sheet.get_all_records()).iloc[-5:][['E', 'F', 'G']].values
    prediction = model.predict(last_rows)[-1]
    return [int(np.random.randint(1, 7)) for _ in range(3)] if not prediction else list(np.random.choice(range(1, 7), 3))

# === TELEGRAM BOT SETUP ===
app = Flask(__name__)
bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=4)

# === XỬ LÝ TIN NHẮN ===
def handle_message(update: Update, context):
    text = update.message.text.strip()
    import re
    match = re.match(r'^([1-6])\s+([1-6])\s+([1-6])$', text)
    if match:
        try:
            parts = list(map(int, match.groups()))
            data = pd.DataFrame(sheet.get_all_records())
            last_row = data.shape[0] + 2
            sheet.update(f"E{last_row}:G{last_row}", [parts])
            new_prediction = predict()
            sheet.update(f"B{last_row}:D{last_row}", [new_prediction])
            match_count = len(set(parts) & set(new_prediction))
            update.message.reply_text(f"Dự đoán: {new_prediction} → Kết quả: {parts} → Trùng: {match_count}/3")
        except Exception as e:
            update.message.reply_text("Đã xảy ra lỗi khi ghi kết quả.")
    else:
        update.message.reply_text("Gửi 3 số xúc xắc cách nhau bởi khoảng trắng (ví dụ: 2 4 6)")

# === SET WEBHOOK ===
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

@app.route("/", methods=["GET"])
def home():
    return "Ensemble AI Bot đang hoạt động!"

# === CHẠY BOTSCHEDULER & WEB ===
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scheduler = BackgroundScheduler()
    scheduler.start()
    bot.set_webhook(f"https://dice-ai-bot.onrender.com/{TELEGRAM_TOKEN}")
    app.run(host="0.0.0.0", port=10000)
