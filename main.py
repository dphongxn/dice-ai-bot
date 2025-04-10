import os
import json
import logging
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters

# --- Logging ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- Environment Variables ---
TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_CREDS = os.environ.get("GOOGLE_CREDS")

# --- Flask App ---
app = Flask(__name__)
bot = Bot(token=TOKEN)

# --- Webhook Route ---
@app.route('/', methods=['GET'])
def index():
    return 'Ensemble AI bot is running!'

@app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return 'ok'

# --- Google Sheet Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(GOOGLE_CREDS)
credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(credentials)
sheet = client.open("App dự đoán 1mf3").worksheet("App dự đoán beta")

# --- ML Models Setup ---
def load_data():
    values = sheet.get_all_values()[1:]
    data = [row[4:7] for row in values if all(row[4:7])]
    df = pd.DataFrame(data, columns=["X1", "X2", "X3"]).astype(int)
    return df

def train_model(df):
    X = df.index.values.reshape(-1, 1)
    y = df[["X1", "X2", "X3"]].values

    models = [
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('ada', AdaBoostClassifier()),
        ('dt', DecisionTreeClassifier()),
        ('nb', GaussianNB())
    ]

    voting_model = VotingClassifier(estimators=models, voting='soft')
    voting_model.fit(X, y)
    return voting_model

def predict_next(model, next_index):
    pred = model.predict([[next_index]])
    return pred.flatten().tolist()

# --- Telegram Handlers ---
def start(update, context):
    update.message.reply_text("Chào bạn! Gửi kết quả xúc xắc (vd: 1 3 6) để tôi học và dự đoán.")

def handle_message(update, context):
    text = update.message.text.strip()
    parts = text.split()

    if len(parts) == 3 and all(p.isdigit() for p in parts):
        last_row = len(sheet.get_all_values()) + 1
        sheet.update(f"E{last_row}:G{last_row}", [parts])
        update.message.reply_text("Đã ghi nhận kết quả!")

        # Train and predict
        df = load_data()
        model = train_model(df)
        next_index = df.shape[0]
        pred = predict_next(model, next_index)
        sheet.update(f"B{last_row+1}:D{last_row+1}", [pred])
        update.message.reply_text(f"Dự đoán tiếp theo: {pred}")
    else:
        update.message.reply_text("Gửi đúng định dạng 3 số ví dụ: 2 3 5")

# --- Dispatcher Setup ---
from telegram.ext import Dispatcher
dispatcher = Dispatcher(bot, None, workers=4)
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# --- Start Flask ---
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))
    bot.set_webhook(f"https://{os.environ['RENDER_EXTERNAL_HOSTNAME']}/{TOKEN}")
    app.run(host="0.0.0.0", port=PORT)
