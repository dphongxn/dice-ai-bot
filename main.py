import os
import json
import logging
import numpy as np
import pandas as pd
from flask import Flask, request
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from apscheduler.schedulers.background import BackgroundScheduler

# Logging
logging.basicConfig(level=logging.INFO)

# ENV
TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "App dự đoán 1mf3")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "App dự đoán beta")
GOOGLE_CREDS = json.loads(os.getenv("GOOGLE_CREDS"))

# Bot & Flask setup
bot = Bot(token=TOKEN)
app = Flask(__name__)
dispatcher = Dispatcher(bot, None, workers=4)

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(GOOGLE_CREDS, scope)
client = gspread.authorize(creds)
sheet = client.open(SPREADSHEET_NAME).worksheet(WORKSHEET_NAME)

# ML Models
def train_models():
    data = sheet.get_all_values()[1:]
    df = pd.DataFrame(data, columns=["A", "B", "C", "E", "F", "G"])
    df = df.dropna()
    X = df[["E", "F", "G"]].astype(int)
    y = df[["B", "C", "D"]].astype(int)

    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('svm', SVC(probability=True)),
        ('nb', GaussianNB()),
        ('knn', KNeighborsClassifier())
    ]

    classifiers = []
    for name, model in models:
        clf = model.fit(X, y)
        classifiers.append((name, clf))

    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X, y)

    return ensemble

model = train_models()

# Handlers
def start(update, context):
    update.message.reply_text("Gửi kết quả xúc xắc dạng: 1 2 3")

def handle_message(update, context):
    try:
        text = update.message.text.strip()
        parts = list(map(int, text.split()))
        if len(parts) != 3 or not all(1 <= p <= 6 for p in parts):
            update.message.reply_text("Vui lòng gửi đúng định dạng: 1 2 3")
            return

        last_row = len(sheet.get_all_values()) + 1
        sheet.update(range_name=f"E{last_row}:G{last_row}", values=[parts])

        pred = model.predict([parts])[0]
        pred_list = list(pred)
        sheet.update(range_name=f"B{last_row}:D{last_row}", values=[pred_list])

        update.message.reply_text(f"Dự đoán: {pred_list}")
    except Exception as e:
        logging.error(f"Lỗi xử lý tin nhắn: {e}")
        update.message.reply_text("Đã xảy ra lỗi!")

# Scheduler để giữ máy sống
scheduler = BackgroundScheduler()
scheduler.start()

# Register handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# Webhook route
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

# Home check route
@app.route("/", methods=["GET"])
def index():
    return "Bot is running!"

# Khởi tạo webhook nếu chưa có
def set_webhook():
    domain = os.getenv("RENDER_EXTERNAL_URL")
    if domain:
        webhook_url = f"{domain}/{TOKEN}"
        bot.set_webhook(url=webhook_url)
        logging.info(f"Webhook set to: {webhook_url}")
    else:
        logging.warning("RENDER_EXTERNAL_URL not set!")

if __name__ == "__main__":
    set_webhook()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
