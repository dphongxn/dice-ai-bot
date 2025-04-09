from flask import Flask
from threading import Thread
import os, json, logging, random, numpy as np, gspread
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from oauth2client.service_account import ServiceAccountCredentials
from telegram.ext import Updater, MessageHandler, Filters
from apscheduler.schedulers.background import BackgroundScheduler

# --- Khởi tạo Flask giữ sống ---
app = Flask(__name__)
@app.route('/')
def home():
    return "Bot is alive!"

def run(): app.run(host='0.0.0.0', port=10000)
def keep_alive():
    t = Thread(target=run)
    t.start()

# --- Ghi file credentials.json từ biến môi trường GOOGLE_CREDS ---
creds_json = os.environ.get("GOOGLE_CREDS")
if creds_json:
    with open("credentials.json", "w") as f:
        json.dump(json.loads(creds_json), f)

# --- Google Sheets ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("app dự đoán 1mf3").worksheet("App dự đoán beta")

# --- Huấn luyện mô hình ---
def train_models(data):
    X, y1, y2, y3 = [], [], [], []
    for i in range(len(data) - 1):
        try:
            x = list(map(int, data[i][4:7]))
            y = list(map(int, data[i + 1][4:7]))
            if len(x) == 3 and len(y) == 3:
                X.append(x)
                y1.append(y[0])
                y2.append(y[1])
                y3.append(y[2])
        except: continue
    if not X: return None
    def build_ensemble(target):
        models = [
            ('gb', GradientBoostingClassifier()),
            ('rf', RandomForestClassifier()),
            ('et', ExtraTreesClassifier()),
            ('knn', KNeighborsClassifier()),
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier())
        ]
        return VotingClassifier(estimators=models, voting='hard').fit(X, target)
    return build_ensemble(y1), build_ensemble(y2), build_ensemble(y3)

# --- Dự đoán ---
def predict_next(models, last_result):
    if not all(models) or not last_result: return random.sample(range(1,7), 3)
    try:
        x = np.array(last_result).reshape(1, -1)
        pred = [int(model.predict(x)[0]) for model in models]
        return list(set(pred))[:3] + [random.choice([i for i in range(1,7) if i not in pred])]
    except: return random.sample(range(1,7), 3)

# --- Tính tỉ lệ đúng ---
def calculate_accuracy():
    data = sheet.get_all_values()
    total, correct = 0, 0
    for row in data:
        try:
            pred = list(map(int, row[1:4]))
            real = list(map(int, row[4:7]))
            if set(pred) == set(real): correct += 1
            total += 1
        except: continue
    return round(correct / total * 100, 2) if total else 0

# --- Xử lý tin nhắn Telegram ---
def handle_message(update, context):
    text = update.message.text.strip()
    try:
        result = list(map(int, text.split()))
        if len(result) != 3: raise ValueError
    except:
        update.message.reply_text("Vui lòng nhập 3 số từ 1-6, cách nhau bằng dấu cách.")
        return

    data = sheet.get_all_values()
    last_row = len(data)
    last_line = data[-1] if data else []
    prediction = None

    if len(last_line) >= 3 and all(last_line[1:4]) and (len(last_line) < 7 or not all(last_line[4:7])):
        sheet.update(f"E{last_row}:G{last_row}", [result])
        prediction = list(map(int, last_line[1:4]))

    models = train_models(data)
    new_prediction = predict_next(models, result)
    sheet.update(f"B{last_row + 1}:D{last_row + 1}", [new_prediction])

    if prediction:
        acc = calculate_accuracy()
        result_status = "ĐÚNG" if set(prediction) == set(result) else "SAI"
        update.message.reply_text(
            f"Dự đoán trước: {prediction} → {result_status}\n"
            f"Tỉ lệ đúng: {acc}%\n"
            f"Dự đoán tiếp theo: {new_prediction}"
        )
    else:
        update.message.reply_text(f"Dự đoán tiếp theo: {new_prediction}")

# --- Chạy bot ---
def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    TOKEN = os.getenv("BOT_TOKEN")
    updater = Updater(TOKEN, use_context=True, workers=4)
    dp = updater.dispatcher
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    scheduler = BackgroundScheduler()
    scheduler.start()
    updater.start_polling()
    print("Bot đang chạy...")
    updater.idle()

# --- CHẠY ---
if __name__ == '__main__':
    keep_alive()
    main()
