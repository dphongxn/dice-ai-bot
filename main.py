import os
import json
import numpy as np
import pandas as pd
import gspread
from flask import Flask, request
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Bot, Update
from telegram.ext import MessageHandler, Filters
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle))
import logging
from io import StringIO

# ===== Thiết lập Flask & logging =====
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ===== Lấy biến môi trường =====
TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = Bot(token=TOKEN)
PORT = int(os.environ.get('PORT', '10000'))

# ===== Lấy Google Sheet =====
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
google_creds = json.loads(os.getenv("GOOGLE_CREDS"))
creds = ServiceAccountCredentials.from_json_keyfile_dict(google_creds, scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# ===== Đọc dữ liệu để huấn luyện =====
def get_data():
    data = sheet.get_all_values()[1:]  # Bỏ dòng tiêu đề
    df = pd.DataFrame(data, columns=["D1", "D2", "D3", "R1", "R2", "R3"])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X = df[["R1", "R2", "R3"]]
    y = df[["D1", "D2", "D3"]]
    return X, y

# ===== Dự đoán bằng Ensemble =====
def train_and_predict(latest_result):
    X, y = get_data()
    models = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier()),
        ("knn", KNeighborsClassifier()),
        ("svm", SVC(probability=True)),
        ("gb", GradientBoostingClassifier())
    ]
    predictions = []
    for i in range(3):  # Dự đoán 3 viên xúc xắc
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X, y.iloc[:, i])
        pred = ensemble.predict([latest_result])
        predictions.append(int(pred[0]))
    return predictions

# ===== Xử lý tin nhắn Telegram =====
def handle(update: Update, context):
    text = update.message.text.strip()
    try:
        parts = list(map(int, text.split()))
        if len(parts) == 3 and all(1 <= x <= 6 for x in parts):
            sheet.append_row(parts, value_input_option="USER_ENTERED", table_range="E:G")
            prediction = train_and_predict(parts)
            sheet.append_row(prediction, value_input_option="USER_ENTERED", table_range="B:D")
            update.message.reply_text(f"EnsembleAI dự đoán: {prediction}")
        else:
            update.message.reply_text("Bạn cần nhập 3 số từ 1 đến 6. Ví dụ: 2 5 3")
    except Exception as e:
        logging.error(e)
        update.message.reply_text("Lỗi khi xử lý kết quả, thử lại nha.")

# ===== Webhook route =====
@app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return 'ok'

@app.route('/')
def home():
    return 'EnsembleAI đang hoạt động!'

# ===== Khởi tạo Telegram Dispatcher =====
dispatcher = Dispatcher(bot, None, use_context=True)
dispatcher.add_handler(CommandHandler('start', lambda update, ctx: update.message.reply_text("Gửi 3 số xúc xắc. Ví dụ: 1 3 6")))
dispatcher.add_handler(CommandHandler('help', lambda update, ctx: update.message.reply_text("Chỉ cần gửi kết quả xúc xắc như: 2 3 5")))
dispatcher.add_handler(CommandHandler('ping', lambda update, ctx: update.message.reply_text("Bot đang chạy nè!")))
dispatcher.add_handler(CommandHandler('kq', handle))
dispatcher.add_handler(CommandHandler('', handle))  # fallback
dispatcher.add_handler(type('AnyTextHandler', (), {'check_update': lambda *a, **kw: True, 'handle_update': handle})())

# ===== Khởi động Flask App =====
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
