# TrustLens - Explainable AI Audit Dashboard

> Real-time bias detection, SHAP explanations, confidence calibration, and drift monitoring for any ML model.

## What It Does
TrustLens wraps any ML model and gives you a live audit dashboard with:
- **SHAP feature explanations** - why did the model predict this?
- **Fairness metrics** - demographic parity + equalized odds across sensitive attributes
- **Data drift detection** - is your incoming data diverging from training data?
- **Immutable audit log** - every prediction logged to SQLite with full metadata

## Quick Start
```bash
git clone https://github.com/muhibwqr/trustlens-ai.git
cd trustlens-ai
pip install -r requirements.txt
python app.py
# Open http://localhost:8000
```

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/explain` | POST | Get prediction + SHAP values for any input |
| `/audit` | GET | View full prediction audit log |
| `/fairness` | GET | Get fairness metrics (demographic parity, equalized odds) |
| `/drift` | GET | Run data drift detection report |

## Tech Stack
Python, FastAPI, SHAP, Fairlearn, Evidently AI, SQLite, scikit-learn, Uvicorn

## Built for
Trustworthy AI Hackathon (George Hacks 2026) - Theme: Databases + ML/AI
