from flask import Flask, request
import os
import logging
import gspread
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Bot, Update
from telegram.ext import Dispatcher, MessageHandler, Filters

# === CONFIG ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_CREDS = os.environ.get("GOOGLE_CREDS")  # Được đặt trong Environment Variable

# === FLASK SETUP ===
app = Flask(__name__)
bot = Bot(token=TELEGRAM_TOKEN)

@app.route('/')
def index():
    return "Bot is alive!"

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "OK"

# === SETUP GOOGLE SHEET ===
import json
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(GOOGLE_CREDS)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# === AI TRAINING FUNCTION ===
def train_models(data):
    X, y = [], []
    for i in range(len(data)-1):
        try:
            x_row = list(map(int, data[i][4:7]))
            y_row = list(map(int, data[i+1][4:7]))
            X.append(x_row)
            y.append(y_row)
        except: continue

    if not X:
        return None

    X = np.array(X)
    y = np.array(y)

    models = []
    for i in range(3):
        y_part = y[:, i]
        models.append([
            RandomForestClassifier().fit(X, y_part),
            GradientBoostingClassifier().fit(X, y_part),
            SVC(probability=True).fit(X, y_part),
            KNeighborsClassifier().fit(X, y_part),
            LogisticRegression(max_iter=1000).fit(X, y_part),
            GaussianNB().fit(X, y_part)
        ])
    return models

# === AI PREDICTION ===
def predict_next(models, last_result):
    if not models:
        return random.sample(range(1, 7), 3)
    result = []
    x = np.array(last_result).reshape(1, -1)
    for i in range(3):
        votes = [model.predict(x)[0] for model in models[i]]
        result.append(int(pd.Series(votes).mode()[0]))
    return result

# === ACCURACY ===
def calculate_accuracy(data):
    total, correct = 0, 0
    for row in data:
        try:
            pred = list(map(int, row[1:4]))
            real = list(map(int, row[4:7]))
            if set(pred) == set(real):
                correct += 1
            total += 1
        except: continue
    return round(correct / total * 100, 2) if total else 0

# === MESSAGE HANDLER ===
def handle_message(update, context):
    text = update.message.text.strip()
    try:
        parts = list(map(int, text.split()))
        if len(parts) != 3:
            raise ValueError
    except:
        update.message.reply_text("Vui lòng nhập đúng 3 số (1-6) cách nhau bởi dấu cách.")
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

    models = train_models(data)
    new_prediction = predict_next(models, parts)
    sheet.update(values=[new_prediction], range_name=f"B{last_row+1}:D{last_row+1}")

    if prediction:
        result = "ĐÚNG" if set(prediction) == set(parts) else "SAI"
        acc = calculate_accuracy(data)
        update.message.reply_text(
            f"Dự đoán trước: {prediction} → {result}\n"
            f"Tỉ lệ đúng: {acc}%\n"
            f"Dự đoán tiếp theo: {new_prediction}"
        )
    else:
        update.message.reply_text(f"Dự đoán tiếp theo: {new_prediction}")

# === DISPATCHER ===
dispatcher = Dispatcher(bot, None, workers=0)
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# === MAIN ===
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=PORT)
