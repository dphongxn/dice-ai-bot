import os
import json
import logging
import pandas as pd
import numpy as np
import gspread
import pytz
from flask import Flask
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_CREDS = os.getenv("GOOGLE_CREDS")

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(GOOGLE_CREDS)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# Hàm huấn luyện mô hình
def train_model():
    data = pd.DataFrame(sheet.get_all_records())
    if len(data) < 20:
        return None

    X = data[['E', 'F', 'G']].values
    y = data[['B', 'C', 'D']].values
    y = [sorted(row) for row in y]

    # Mã hóa đầu ra thành chuỗi
    y = [''.join(map(str, row)) for row in y]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier()),
        ('svc', SVC(probability=True)),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ]
    
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X_scaled, y)
    
    return ensemble, scaler

# Dự đoán
def predict(model, scaler):
    data = pd.DataFrame(sheet.get_all_records())
    if len(data) < 1:
        return None

    last_row = data.iloc[-1][['E', 'F', 'G']].values
    X = scaler.transform([last_row])
    prediction = model.predict(X)[0]
    return list(prediction)

# Cập nhật dự đoán tự động
def update_prediction():
    model_data = train_model()
    if not model_data:
        logging.warning("Không đủ dữ liệu để huấn luyện.")
        return

    model, scaler = model_data
    new_prediction = predict(model, scaler)
    if not new_prediction:
        return

    data = pd.DataFrame(sheet.get_all_records())
    last_row = len(data) + 1
    sheet.update(range_name=f"B{last_row}:D{last_row}", values=[new_prediction])
    logging.info(f"Cập nhật dự đoán: {new_prediction}")

# Telegram command
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Bot Ensemble AI dự đoán xúc xắc đã sẵn sàng!")

def status(update: Update, context: CallbackContext):
    update.message.reply_text("Bot đang chạy ổn định. Bạn có thể nhập kết quả mới trên Google Sheet.")

# Flask server để giữ bot hoạt động
app = Flask(__name__)

@app.route('/')
def home():
    return "Ensemble AI đang chạy..."

# Hàm chính
def main():
    updater = Updater(TOKEN, use_context=True, workers=4)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("status", status))

    # Scheduler
    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Ho_Chi_Minh"))
    scheduler.add_job(update_prediction, 'interval', minutes=1)
    scheduler.start()

    updater.start_polling()
    app.run(host='0.0.0.0', port=10000)

if __name__ == "__main__":
    main()
