from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shap
import numpy as np
import sqlite3
import json
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import fairlearn.metrics as flm
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import uvicorn

app = FastAPI(title="TrustLens API", description="Explainable AI Audit Dashboard")

# --- Train model on startup ---
data = load_breast_cancer()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)
explainer = shap.TreeExplainer(model)

# --- SQLite audit log ---
conn = sqlite3.connect("audit_log.db", check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, input TEXT, prediction INTEGER,
    confidence REAL, shap_values TEXT
)""")
conn.commit()

@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open("static/index.html") as f:
        return f.read()

@app.post("/explain")
def explain(features: dict):
    try:
        input_df = pd.DataFrame([features])
        input_scaled = scaler.transform(input_df)
        pred = int(model.predict(input_scaled)[0])
        confidence = float(model.predict_proba(input_scaled)[0][pred])
        shap_vals = explainer.shap_values(input_scaled)[1][0]
        shap_dict = {k: round(float(v), 4) for k, v in zip(data.feature_names, shap_vals)}
        conn.execute("INSERT INTO predictions VALUES (NULL,?,?,?,?,?)", (
            datetime.datetime.utcnow().isoformat(), json.dumps(features),
            pred, confidence, json.dumps(shap_dict)
        ))
        conn.commit()
        return {"prediction": pred, "confidence": confidence, "shap_values": shap_dict,
                "label": data.target_names[pred]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/audit")
def get_audit_log(limit: int = 50):
    rows = conn.execute(
        "SELECT id, timestamp, prediction, confidence FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [{"id": r[0], "timestamp": r[1], "prediction": r[2], "confidence": r[3]} for r in rows]

@app.get("/fairness")
def fairness_report():
    preds = model.predict(X_test_s)
    sensitive = (X_test["mean radius"] > X_test["mean radius"].median()).astype(int)
    dp = flm.demographic_parity_difference(y_test, preds, sensitive_features=sensitive)
    eo = flm.equalized_odds_difference(y_test, preds, sensitive_features=sensitive)
    return {"demographic_parity_difference": round(float(dp), 4),
            "equalized_odds_difference": round(float(eo), 4),
            "note": "Lower is fairer. <0.1 is considered acceptable."}

@app.get("/drift")
def drift_report():
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=X_train[:100], current_data=X_test)
    result = report.as_dict()
    drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
    total = result["metrics"][0]["result"]["number_of_columns"]
    return {"drifted_columns": drifted, "total_columns": total,
            "drift_share": round(drifted / total, 3)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
