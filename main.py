import os
import json
import logging
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến môi trường
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
GOOGLE_CREDS = os.environ.get("GOOGLE_CREDS")

# Khởi tạo bot
bot = Bot(token=TELEGRAM_TOKEN)

# Flask App
app = Flask(__name__)

# Đọc Google credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(GOOGLE_CREDS)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open("App dự đoán 1mf3").worksheet("App dự đoán beta")

# Hàm lấy dữ liệu huấn luyện
def get_training_data():
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df[['E', 'F', 'G']].dropna()
    X = df.shift(1).dropna().astype(int)
    y = df.iloc[1:].astype(int)
    y = y[['E', 'F', 'G']].apply(lambda row: sorted([int(row['E']), int(row['F']), int(row['G'])]), axis=1)
    return X.values, y.values

# Hàm huấn luyện và dự đoán
def predict_dice():
    X, y = get_training_data()
    y_flat = [",".join(map(str, row)) for row in y]

    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True)),
        ('knn', KNeighborsClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('nb', GaussianNB())
    ]

    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X, y_flat)

    last_row = X[-1].reshape(1, -1)
    prediction = ensemble.predict(last_row)[0]
    return list(map(int, prediction.split(',')))

# Hàm xử lý tin nhắn
def handle_message(update: Update, context):
    text = update.message.text.strip()
    try:
        if text.lower().startswith("kq"):
            parts = list(map(int, text.replace("kq", "").strip().split()))
            if len(parts) != 3: raise ValueError("Sai định dạng. Nhập: kq 1 2 3")

            last_row = len(sheet.get_all_values())
            sheet.update(range_name=f"E{last_row}:G{last_row}", values=[parts])

            prediction = predict_dice()
            sheet.update(range_name=f"B{last_row+1}:D{last_row+1}", values=[prediction])

            correct = sorted(parts) == sorted(prediction)
            update.message.reply_text(
                f"Dự đoán: {prediction} | Kết quả: {parts} => {'ĐÚNG' if correct else 'SAI'}"
            )
        else:
            update.message.reply_text("Gõ 'kq 1 2 3' để ghi kết quả xúc xắc!")
    except Exception as e:
        logger.error(str(e))
        update.message.reply_text("Đã có lỗi xảy ra: " + str(e))

# Thiết lập dispatcher
dispatcher = Dispatcher(bot, None, workers=4, use_context=True)
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# Route cho Telegram Webhook
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "OK", 200

# Route kiểm tra server
@app.route("/", methods=["GET"])
def index():
    return "EnsembleAI đang hoạt động!", 200

# Hàm khởi động webhook
def set_webhook():
    webhook_url = f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}"
    bot.set_webhook(url=webhook_url)

if __name__ == "__main__":
    set_webhook()
    app.run(host="0.0.0.0", port=10000)
