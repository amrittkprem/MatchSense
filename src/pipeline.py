"""Match outcome prediction pipeline components.

This module provides a clean, readable starting point for feature engineering,
training, and evaluating probabilistic football match outcome models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED_COLS = ["date", "home_team", "away_team", "home_goals", "away_goals"]


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame


def load_matches(path: str) -> pd.DataFrame:
    """Load match data from CSV and perform basic cleanup."""
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Training should only use completed matches with final scores.
    df = df.dropna(subset=["home_goals", "away_goals"])

    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()

    return df


def _result_points(home_goals: int, away_goals: int) -> Tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def build_features(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Create rolling form and goal-difference features."""
    df = df.copy()

    df["result"] = np.select(
        [df.home_goals > df.away_goals, df.home_goals == df.away_goals],
        ["H", "D"],
        default="A",
    )

    teams = pd.concat([df.home_team, df.away_team]).unique()
    history: Dict[str, List[Dict[str, int]]] = {team: [] for team in teams}

    home_form: List[int] = []
    away_form: List[int] = []
    home_gd: List[int] = []
    away_gd: List[int] = []

    for _, row in df.iterrows():
        home = row.home_team
        away = row.away_team

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

        hp, ap = _result_points(row.home_goals, row.away_goals)
        history[home].append({"points": hp, "gd": row.home_goals - row.away_goals})
        history[away].append({"points": ap, "gd": row.away_goals - row.home_goals})

    df["home_form"] = home_form
    df["away_form"] = away_form
    df["home_gd"] = home_gd
    df["away_gd"] = away_gd
    df["form_diff"] = df.home_form - df.away_form
    df["gd_diff"] = df.home_gd - df.away_gd

    if "home_elo" in df.columns and "away_elo" in df.columns:
        df["elo_diff"] = df.home_elo - df.away_elo
    else:
        df["elo_diff"] = 0.0

    return df


def time_split(df: pd.DataFrame, test_size: float = 0.2) -> SplitData:
    """Time-aware split: train is older data, valid is most recent slice."""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    valid = df.iloc[split_idx:]
    return SplitData(train=train, valid=valid)


def train_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: Iterable[str],
    target: str = "result",
) -> Dict[str, Pipeline | RandomForestClassifier]:
    X_train = train_df[list(features)]
    y_train = train_df[target]
    X_valid = valid_df[list(features)]
    y_valid = valid_df[target]

    log_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(multi_class="multinomial", max_iter=1000)),
        ]
    )
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)

    models: Dict[str, Pipeline | RandomForestClassifier] = {
        "log_reg": log_reg,
        "rf": rf,
    }

    for name, model in models.items():
        probs = model.predict_proba(X_valid)
        ll = log_loss(y_valid, probs)

        brier = 0.0
        for i, cls in enumerate(model.classes_):
            y_true = (y_valid == cls).astype(int)
            brier += brier_score_loss(y_true, probs[:, i])
        brier /= len(model.classes_)

        print(f"{name} log loss: {ll:.4f} | brier: {brier:.4f}")

    return models


def calibrate_model(
    base_model: Pipeline | RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def predict_fixture(
    model: Pipeline | RandomForestClassifier,
    fixture_features: Dict[str, float],
) -> Dict[str, float]:
    X = pd.DataFrame([fixture_features])
    probs = model.predict_proba(X)[0]
    return dict(zip(model.classes_, probs))
