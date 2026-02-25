# ⚽ Football Match Outcome Predictor

A Python-based machine learning system that predicts the probability of **Home Win, Draw, and Away Win** for a given football match using basic team-level features and simple ML models.

This project is designed as an **end-to-end ML engineering demo**, showcasing data ingestion, feature engineering, model training, and inference via both a CLI tool and an optional web interface. It focuses on system design and workflow rather than claiming high real-world prediction accuracy.

---

## 🚀 Features

- Command-line interface (CLI) for match prediction  
- Optional web UI built with Flask  
- End-to-end ML pipeline (training → saving → loading → inference)  
- Feature engineering from team statistics  
- Two ML models:
  - Logistic Regression (baseline)
  - Random Forest (improved model)  
- Probabilistic predictions:
  - Home Win  
  - Draw  
  - Away Win  
- Automatic model training if no trained model is found  
- Local caching layer for fetched data  
- Clear disclaimers about uncertainty and limitations  

---

## 🧠 How It Works

1. **User Input**
   - Home Team  
   - Away Team  
   - Match Date  
   - League  

2. **Data Layer**
   - Team statistics are fetched via a data-fetching module.
   - The current implementation uses placeholder/mock data to keep the pipeline stable and reproducible.
   - The system is designed so real data sources (APIs or scrapers) can be plugged in later.

3. **Feature Engineering**
   - Goal difference  
   - Defensive strength difference  
   - Form difference  
   - Team strength (ELO-like proxy)  
   - Home advantage  
   - Rest/fatigue difference  

4. **Models**
   - Logistic Regression (baseline classifier)  
   - Random Forest (non-linear classifier)  

5. **Output**
   - Probability of Home Win  
   - Probability of Draw  
   - Probability of Away Win  

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
