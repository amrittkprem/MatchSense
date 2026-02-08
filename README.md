# MatchSense: Football Match Outcome Prediction Pipeline

This repository provides a beginner-friendly, production-minded blueprint for an end-to-end football (soccer) match outcome prediction system. It is designed for learning and portfolio development, **not** gambling.

---

## 1. End-to-end system architecture

**High-level stages**

1. **Data ingestion**
   - Load historical match data from CSV (league, teams, date, scores, odds, ELO, etc.).
   - Validate columns, parse dates, and standardize team names.

2. **Preprocessing**
   - Filter incomplete rows (missing scores for training).
   - Sort matches chronologically.
   - Build a clean, consistent dataset with one row per match.

3. **Feature engineering**
   - **Team form**: rolling points/goal difference in last N matches.
   - **Home/away split**: separate home-form and away-form features.
   - **ELO features**: ELO ratings and differences.
   - **League context**: positions, points per game, seasonal averages (if available).
   - **Market/odds features** (optional): implied probabilities from bookmaker odds.

4. **Model training**
   - Baseline **multinomial logistic regression**.
   - Tree-based model (e.g., **XGBoost**, **LightGBM**, or **RandomForest**).
   - Use a **time-aware split** so future data is never used to predict the past.

5. **Evaluation** (probabilistic)
   - **Log loss** (cross-entropy): penalizes overconfident wrong predictions.
   - **Brier score**: measures probability calibration.
   - Optional: calibration curves or reliability diagrams.

6. **Calibration**
   - If probabilities are miscalibrated, apply `CalibratedClassifierCV` or isotonic regression.

7. **Inference**
   - Given a future fixture (e.g., Real Madrid vs Barcelona), build features from the latest data.
   - Output a probability vector for Home Win / Draw / Away Win.

---

## 2. Practical feature set (with rationale)

| Feature | Why it matters |
| --- | --- |
| Rolling points last N matches | Captures recent form (momentum) |
| Rolling goal difference (GD) | Proxy for dominance and attacking/defensive strength |
| Home/away split form | Teams perform differently at home vs away |
| ELO rating & ELO diff | Compact, proven strength rating system |
| League position/points | Relative strength in the league context |
| Rest days since last match | Fatigue and rotation impact |
| Odds-implied probabilities (optional) | Market consensus (strong signal, but careful of leakage) |

---

## 3. Code: Modular pipeline

### 3.1 Loading + cleaning data

```python
import pandas as pd

REQUIRED_COLS = [
    "date", "home_team", "away_team", "home_goals", "away_goals"
]


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Remove matches without final scores (training only uses completed games)
    df = df.dropna(subset=["home_goals", "away_goals"])

    # Standardize team names (basic cleanup)
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    return df
```

### 3.2 Feature engineering (form, GD, ELO diff)

```python
import numpy as np
import pandas as pd


def _result_points(home_goals: int, away_goals: int):
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def build_features(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    df = df.copy()

    # Target label
    df["result"] = np.select(
        [df.home_goals > df.away_goals, df.home_goals == df.away_goals],
        ["H", "D"],
        default="A",
    )

    # Track rolling form and GD per team
    teams = pd.concat([df.home_team, df.away_team]).unique()
    history = {team: [] for team in teams}

    home_form = []
    away_form = []
    home_gd = []
    away_gd = []

    for _, row in df.iterrows():
        home = row.home_team
        away = row.away_team

        # Rolling stats before current match
        home_hist = history[home][-form_window:]
        away_hist = history[away][-form_window:]

        home_points = sum(h["points"] for h in home_hist)
        away_points = sum(h["points"] for h in away_hist)
        home_gd_val = sum(h["gd"] for h in home_hist)
        away_gd_val = sum(h["gd"] for h in away_hist)

        home_form.append(home_points)
        away_form.append(away_points)
        home_gd.append(home_gd_val)
        away_gd.append(away_gd_val)

        # Update history after computing features
        hp, ap = _result_points(row.home_goals, row.away_goals)
        history[home].append({"points": hp, "gd": row.home_goals - row.away_goals})
        history[away].append({"points": ap, "gd": row.away_goals - row.home_goals})

    df["home_form"] = home_form
    df["away_form"] = away_form
    df["home_gd"] = home_gd
    df["away_gd"] = away_gd
    df["form_diff"] = df.home_form - df.away_form
    df["gd_diff"] = df.home_gd - df.away_gd

    # Optional ELO features if present
    if "home_elo" in df.columns and "away_elo" in df.columns:
        df["elo_diff"] = df.home_elo - df.away_elo
    else:
        df["elo_diff"] = 0.0

    return df
```

### 3.3 Train/validation split (time-aware)

```python
from sklearn.model_selection import TimeSeriesSplit


def time_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    valid = df.iloc[split_idx:]
    return train, valid
```

### 3.4 Train models + evaluate

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss


def train_models(train_df, valid_df, features, target="result"):
    X_train = train_df[features]
    y_train = train_df[target]
    X_valid = valid_df[features]
    y_valid = valid_df[target]

    # Baseline: multinomial logistic regression
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", max_iter=1000)),
    ])
    log_reg.fit(X_train, y_train)

    # Tree-based model (baseline version)
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)

    models = {"log_reg": log_reg, "rf": rf}

    for name, model in models.items():
        probs = model.predict_proba(X_valid)
        ll = log_loss(y_valid, probs)

        # Brier score for multiclass: average over classes
        brier = 0
        for i, cls in enumerate(model.classes_):
            y_true = (y_valid == cls).astype(int)
            brier += brier_score_loss(y_true, probs[:, i])
        brier /= len(model.classes_)

        print(f"{name} log loss: {ll:.4f} | brier: {brier:.4f}")

    return models
```

### 3.5 Calibration (optional)

```python
from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(base_model, X_train, y_train):
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated
```

### 3.6 Example prediction (Real Madrid vs Barcelona)

```python

def predict_fixture(model, fixture_features: dict):
    import pandas as pd
    X = pd.DataFrame([fixture_features])
    probs = model.predict_proba(X)[0]
    return dict(zip(model.classes_, probs))

# Example (dummy values, you would compute from latest data)
fixture = {
    "home_form": 10,
    "away_form": 8,
    "home_gd": 6,
    "away_gd": 4,
    "form_diff": 2,
    "gd_diff": 2,
    "elo_diff": 35,
}

# probs = predict_fixture(best_model, fixture)
# print(probs)
```

---

## 4. Common pitfalls (and how to avoid them)

- **Data leakage**: Never use future match outcomes or odds after kickoff. Keep features strictly pre-match.
- **Overfitting**: Use a time-aware split; avoid too many engineered features without validation.
- **Class imbalance**: Draws are often underrepresented; evaluate with log loss not accuracy alone.
- **Fundamental uncertainty**: Football is noisy (small scoreline, red cards, injuries). Even the best models have limited predictive power.

---

## 5. Portfolio extensions

- **Streamlit web UI** for predictions + match explorer.
- **MLflow or Weights & Biases** for experiment tracking.
- **Scheduled retraining** and rolling evaluation dashboard.
- **API deployment** with FastAPI + Docker.
- **Model monitoring** with calibration plots and performance drift alerts.

---

## 6. Running locally (example)

```bash
pip install pandas scikit-learn
python -c "import pandas, sklearn; print('ready')"
```

---

## Notes

This project is for learning and portfolio use. It should not be used for gambling or decision-making with financial risk.
