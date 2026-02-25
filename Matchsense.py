import os
import json
import time
import pickle
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from flask import Flask, request, render_template_string

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
CACHE_DIR = "cache"
MODEL_DIR = "models"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (non-commercial, educational project)"}

# ---------------- SCRAPING ----------------
def fetch_team_stats(team, league):
    fname = f"{CACHE_DIR}/{league}_{team}.json".replace(" ", "_")

    if os.path.exists(fname):
        with open(fname, "r") as f:
            return json.load(f)

    # Placeholder public scraping (safe fallback)
    try:
        time.sleep(1.5)  # rate limit
        data = {
            "goals_for": int(np.random.randint(10, 40)),
            "goals_against": int(np.random.randint(10, 40)),
            "wins": int(np.random.randint(1, 10)),
            "draws": int(np.random.randint(1, 10)),
            "losses": int(np.random.randint(1, 10)),
            "elo": int(np.random.randint(1300, 1900)),
            "home_strength": float(np.random.uniform(0.4, 0.7)),
            "away_strength": float(np.random.uniform(0.3, 0.6)),
            "rest_days": int(np.random.randint(2, 7))
        }
    except Exception:
        data = {
            "goals_for": 20,
            "goals_against": 20,
            "wins": 5,
            "draws": 5,
            "losses": 5,
            "elo": 1500,
            "home_strength": 0.5,
            "away_strength": 0.5,
            "rest_days": 4
        }

    with open(fname, "w") as f:
        json.dump(data, f, indent=2)

    return data

# ---------------- FEATURES ----------------
def build_features(home, away):
    features = {
        "goal_diff": home["goals_for"] - away["goals_for"],
        "defense_diff": away["goals_against"] - home["goals_against"],
        "elo_diff": home["elo"] - away["elo"],
        "form_diff": home["wins"] - away["wins"],
        "home_adv": home["home_strength"] - away["away_strength"],
        "rest_diff": home["rest_days"] - away["rest_days"],
    }
    return pd.DataFrame([features])

# ---------------- TRAINING ----------------
def generate_training_data(n=1000):
    X = pd.DataFrame({
        "goal_diff": np.random.randn(n),
        "defense_diff": np.random.randn(n),
        "elo_diff": np.random.randn(n),
        "form_diff": np.random.randn(n),
        "home_adv": np.random.randn(n),
        "rest_diff": np.random.randn(n),
    })
    y = np.random.choice([0, 1, 2], size=n)
    return X, y

def train_models():
    print("[INFO] Training models...")

    X, y = generate_training_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(max_iter=1000)  # compatible with old sklearn
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    acc_lr = accuracy_score(y_val, logreg.predict(X_val))
    acc_rf = accuracy_score(y_val, rf.predict(X_val))

    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/logreg.pkl", "wb") as f:
        pickle.dump(logreg, f)
    with open(f"{MODEL_DIR}/rf.pkl", "wb") as f:
        pickle.dump(rf, f)

    print(f"[OK] Logistic Regression Accuracy: {acc_lr:.2f}")
    print(f"[OK] Random Forest Accuracy: {acc_rf:.2f}")
    print("[OK] Models saved.")

# ---------------- LOAD MODELS ----------------
def load_models():
    if not os.path.exists(f"{MODEL_DIR}/scaler.pkl") or not os.path.exists(f"{MODEL_DIR}/rf.pkl"):
        print("[WARN] Models not found. Training automatically...")
        train_models()

    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/rf.pkl", "rb") as f:
        model = pickle.load(f)

    return scaler, model

# ---------------- PREDICT ----------------
def predict_match(home_team, away_team, date, league):
    home = fetch_team_stats(home_team, league)
    away = fetch_team_stats(away_team, league)

    X = build_features(home, away)
    scaler, model = load_models()
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[0]
    labels = ["Home Win", "Draw", "Away Win"]
    return dict(zip(labels, probs))

# ---------------- CLI ----------------
def cli():
  print("\n⚠️ This Project Is Trained on Dummy Dataset")
  print("\n⚠️ This Project Is Made for Fun")
    print("\n⚽ Football Match Predictor\n")
    home = input("Home Team: ").strip()
    away = input("Away Team: ").strip()
    date = input("Match Date (YYYY-MM-DD): ").strip()
    league = input("League: ").strip()

    probs = predict_match(home, away, date, league)

    print("\n🔮 Prediction Probabilities")
    for k, v in probs.items():
        print(f"{k}: {v*100:.1f}%")

    print("\n⚠️ Disclaimer: This model is probabilistic.")

# ---------------- WEB ----------------
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Football Predictor</title>
<style>
body { font-family: Arial; padding: 30px; max-width: 500px; margin: auto; }
input, button { width: 100%; padding: 10px; margin: 8px 0; }
.bar { height: 20px; background: #ddd; margin: 5px 0; }
.fill { height: 100%; background: #4caf50; }
</style>
</head>
<body>
<h2>Football Match Predictor</h2>
<form method="post">
<input name="home" placeholder="Home Team" required>
<input name="away" placeholder="Away Team" required>
<input name="date" placeholder="Match Date" required>
<input name="league" placeholder="League" required>
<button>Predict</button>
</form>

{% if result %}
<h3>Result</h3>
{% for k,v in result.items() %}
<p>{{k}}: {{(v*100)|round(1)}}%</p>
<div class="bar"><div class="fill" style="width: {{v*100}}%"></div></div>
{% endfor %}
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        result = predict_match(
            request.form["home"],
            request.form["away"],
            request.form["date"],
            request.form["league"]
        )
    return render_template_string(HTML, result=result)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--web", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_models()
    elif args.web:
        app.run(debug=True)
    else:
        cli()
