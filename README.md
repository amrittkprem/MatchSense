A Python-based machine learning system that predicts the probability of Home Win, Draw, and Away Win for a given football match using basic team statistics and simple ML models.

This project is built as an end-to-end ML engineering demo with:

Data fetching layer (currently mocked / placeholder)

Feature engineering

Model training and persistence

CLI tool

Optional web interface (Flask)

The goal of this project is to demonstrate system design and ML workflow, not to claim high real-world prediction accuracy.

🚀 Features

Command-line interface (CLI) for match prediction

Optional web UI built with Flask

End-to-end ML pipeline (training → saving → loading → inference)

Feature engineering from team-level statistics

Two ML models:

Logistic Regression (baseline)

Random Forest (improved model)

Probabilistic outputs for:

Home Win

Draw

Away Win

Automatic model training if no trained model is found

Local caching layer for fetched data

Clear disclaimers about uncertainty and limitations

🧠 How It Works

User Input

Home Team

Away Team

Match Date

League

Data Layer

Team statistics are fetched via a data-fetching module.

Current implementation uses placeholder/mock data to keep the pipeline stable and reproducible.

The system is designed so real data sources (APIs or scrapers) can be plugged in later.

Feature Engineering

Goal difference

Defensive strength difference

Form difference

Team strength (ELO-like proxy)

Home advantage

Rest/fatigue difference

Models

Logistic Regression (baseline classifier)

Random Forest (non-linear model)

Output

Probability of Home Win

Probability of Draw

Probability of Away Win

🛠️ Installation

Clone the repository and install dependencies:

pip install -r requirements.txt

requirements.txt

numpy
pandas
requests
beautifulsoup4
scikit-learn
flask

▶️ Usage
1. Train the Models (Run Once)
python football_match_predictor.py --train

This trains the ML models and saves them locally.

2. Run CLI Predictor
python football_match_predictor.py

You will be prompted to enter:

Home Team

Away Team

Match Date

League

3. Run Web Interface (Optional)
python football_match_predictor.py --web

Then open your browser at:

http://127.0.0.1:5000

⚠️ Disclaimer

Football match outcomes are highly unpredictable.

The model provides probabilistic predictions, not guarantees.

Current data-fetching uses placeholder/mock data to demonstrate the ML pipeline.

This project is intended for:

Learning ML engineering workflows

Portfolio demonstration

Experimentation

It is not intended for betting, financial decisions, or real-world prediction systems.

🔬 Model Performance

Since the current system is trained on synthetic data, accuracy is limited and not representative of real-world performance.

This project focuses on:

Pipeline design

ML system structure

Feature engineering patterns

Model training and inference flow

🧩 Future Improvements

Replace placeholder data with real football data (APIs or public datasets)

Add team ELO calculation

Incorporate expected goals (xG) features

Add player-level features (injuries, suspensions)

Improve model evaluation and validation

Deploy as a hosted web application

👤 Author

Built as a personal project to practice:

End-to-end ML pipelines

Feature engineering

Model training and deployment

CLI and web application integration
