import pickle
import numpy as np
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ---------------------------
# Telegram Bot Token
# ---------------------------
# Railway / Render me TOKEN ko Environment Variable me store karna best practice hai
TOKEN = os.environ.get("TOKEN")  # Replace with direct token only for testing

# ---------------------------
# Load Trained ML Model
# ---------------------------
model = pickle.load(open("ml_model.pkl", "rb"))

# ---------------------------
# Rules Function
# ---------------------------
def rules_predict(last_rounds):
    streak = 3  # Check last 3 rounds
    if len(last_rounds) < streak:
        return None
    # Cold streak → last 3 rounds < 1.5 → HIGH prediction
    if all(m < 1.5 for m in last_rounds[-streak:]):
        return "HIGH"
    # Hot streak → last 3 rounds > 2.0 → LOW prediction
    elif all(m > 2.0 for m in last_rounds[-streak:]):
        return "LOW"
    return None

# ---------------------------
# Hybrid Prediction Function
# ---------------------------
def hybrid_predict(last_rounds):
    if len(last_rounds) < 10:
        return "Not enough data", 0
    # ML features
    rolling5 = np.mean(last_rounds[-5:])
    rolling10 = np.mean(last_rounds[-10:])
    pred_prob = model.predict_proba([[rolling5, rolling10]])[0][1]
    ml_pred = "HIGH" if pred_prob > 0.5 else "LOW"
    # Rules prediction
    rules_pred = rules_predict(last_rounds)
    # Combine ML + Rules
    if rules_pred:
        final_pred = rules_pred
        confidence = max(pred_prob, 0.7)  # High confidence if rules trigger
    else:
        final_pred = ml_pred
        confidence = pred_prob
    return final_pred, confidence

# ---------------------------
# Example Last Rounds Data
# Replace with real-time scraping data later
# ---------------------------
last_rounds = [1.2, 2.1, 1.5, 2.3, 1.8, 2.5, 1.4, 1.7, 2.0, 2.6]

# ---------------------------
# Telegram Commands
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 Welcome to Hybrid AI Aviator Bot!\nUse /predict to get next round prediction."
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pred, conf = hybrid_predict(last_rounds)
    await update.message.reply_text(
        f"Next round prediction: {pred}\nConfidence: {conf*100:.2f}%"
    )

# ---------------------------
# Setup Telegram Application
# ---------------------------
app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("predict", predict))

# ---------------------------
# Run Bot 24/7
# ---------------------------
app.run_polling()
