import os
import json
import logging
from flask import Flask
from threading import Thread
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# === Keep alive server ===
app = Flask(__name__)
@app.route('/')
def index():
    return 'Bot is running...'

def run():
    app.run(host='0.0.0.0', port=10000)

Thread(target=run).start()

# === Google Sheets setup ===
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
google_creds = json.loads(os.getenv('GOOGLE_CREDS'))
creds = ServiceAccountCredentials.from_json_keyfile_dict(google_creds, scope)
client = gspread.authorize(creds)
sheet = client.open("App dự đoán 1mf3").worksheet("App dự đoán beta")

# === Model setup ===
def get_data():
    data = pd.DataFrame(sheet.get_all_records())
    valid_data = data.dropna(subset=["E", "F", "G"])  # Cột kết quả thực tế
    if len(valid_data) < 5:
        return None, None
    X = valid_data[["E", "F", "G"]].values[:-1]
    y = valid_data[["E", "F", "G"]].apply(lambda row: ''.join(sorted(map(str, row))), axis=1).values[1:]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded

def train_ensemble(X, y):
    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('svc', SVC(probability=True)),
        ('ada', AdaBoostClassifier()),
        ('dt', DecisionTreeClassifier())
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X, y)
    return ensemble

def predict_next(model, last_result):
    result = np.array(last_result).reshape(1, -1)
    prediction = model.predict(result)[0]
    return prediction

# === Telegram Bot setup ===
TOKEN = os.getenv('BOT_TOKEN')
updater = Updater(token=TOKEN, use_context=True, workers=4)
dispatcher = updater.dispatcher

def start(update, context):
    update.message.reply_text("Gửi kết quả xúc xắc theo cú pháp: 2 4 6")

def handle_message(update, context):
    text = update.message.text.strip()
    chat_id = update.effective_chat.id

    try:
        parts = list(map(int, text.split()))
        if len(parts) != 3 or any(p < 1 or p > 6 for p in parts):
            raise ValueError
    except ValueError:
        update.message.reply_text("Vui lòng gửi đúng 3 số từ 1 đến 6. Ví dụ: 2 3 6")
        return

    # Ghi kết quả vào sheet
    all_data = sheet.get_all_values()
    last_row = len(all_data) + 1
    sheet.update(f"E{last_row}:G{last_row}", [parts])

    # Lấy dữ liệu và huấn luyện
    X, y = get_data()
    if X is None:
        update.message.reply_text("Chưa đủ dữ liệu để dự đoán. Vui lòng gửi thêm!")
        return

    model = train_ensemble(X, y)

    # Dự đoán tiếp theo
    last_result = parts
    pred_label = predict_next(model, last_result)
    update.message.reply_text(f"Đã ghi nhận kết quả!\nDự đoán tiếp theo: {pred_label}")

def error_handler(update, context):
    print(f"Lỗi: {context.error}")
    if update:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Có lỗi xảy ra. Vui lòng thử lại sau!")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
dispatcher.add_error_handler(error_handler)

# === Start polling ===
updater.start_polling()
