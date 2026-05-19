"""
src/predictor.py
----------------
Utility: load model, predict single employee, batch predict, generate recommendations.
Used by the Streamlit dashboard.
"""

import os, joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple

BASE   = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(BASE, "..", "models", "employee_perf_model.pkl")
ENCODER_PATH = os.path.join(BASE, "..", "models", "label_encoder.pkl")
DROP_COLS    = ["employee_id", "performance_score", "perf_band_next"]

# ── Intervention map: feature → recommended action ───────────────────────
INTERVENTIONS = {
    "on_time_delivery_rate": {
        "threshold": 0.70,
        "direction": "low",
        "action": "📅 Sprint Planning Workshop — improve task scheduling & prioritization",
        "icon": "⏰"
    },
    "training_hours": {
        "threshold": 20,
        "direction": "low",
        "action": "📚 Assign targeted L&D courses (e.g., AWS CCP, Azure DP-900, Agile cert)",
        "icon": "🎓"
    },
    "bug_count": {
        "threshold": 10,
        "direction": "high",
        "action": "🔧 Pair Programming & Code Quality Course — reduce defect density",
        "icon": "🐛"
    },
    "peer_feedback_score": {
        "threshold": 3.0,
        "direction": "low",
        "action": "🤝 Team Collaboration Workshop — improve communication & teamwork",
        "icon": "👥"
    },
    "manager_score": {
        "threshold": 3.0,
        "direction": "low",
        "action": "📋 1:1 Coaching Sessions with manager — align on goals & expectations",
        "icon": "🎯"
    },
    "avg_task_delay_days": {
        "threshold": 5.0,
        "direction": "high",
        "action": "🗂️ Time Management & Workflow Optimization training",
        "icon": "⚡"
    },
    "certifications_count": {
        "threshold": 1,
        "direction": "low",
        "action": "🏆 Encourage obtaining at least 1 industry certification this quarter",
        "icon": "📜"
    },
    "avg_login_hours": {
        "threshold": 6.0,
        "direction": "low",
        "action": "💡 Engagement check-in — investigate workload, burnout, or personal issues",
        "icon": "💬"
    },
    "sick_days": {
        "threshold": 10,
        "direction": "high",
        "action": "🏥 Employee Wellness Program — arrange health & wellbeing support",
        "icon": "❤️"
    },
    "code_review_score": {
        "threshold": 3.0,
        "direction": "low",
        "action": "👨‍💻 Mentorship Program — code review best practices & standards",
        "icon": "🔍"
    },
}

BAND_COLORS = {"High": "#22c55e", "Medium": "#f59e0b", "Low": "#ef4444"}
BAND_ICONS  = {"High": "🌟", "Medium": "📈", "Low": "⚠️"}

# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    """Load trained model + label encoder. Returns (model, encoder)."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

# ─────────────────────────────────────────────────────────────────────────────
def predict_employee(row: pd.DataFrame) -> Tuple[str, np.ndarray, Dict]:
    """
    Predict performance band + probability for a single employee row.
    Returns (predicted_label, probas_array, {label: prob} dict)
    """
    model, encoder = load_artifacts()
    if model is None:
        raise RuntimeError("Model not found. Please train first.")

    # Drop leakage cols if present
    X = row.drop(columns=[c for c in DROP_COLS if c in row.columns], errors="ignore")
    probas = model.predict_proba(X)[0]
    pred   = model.predict(X)[0]
    label  = encoder.inverse_transform([pred])[0]
    proba_dict = {encoder.inverse_transform([i])[0]: round(float(p), 4)
                  for i, p in enumerate(probas)}
    return label, probas, proba_dict

# ─────────────────────────────────────────────────────────────────────────────
def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Run predictions on entire dataset. Adds pred_band + probabilities."""
    model, encoder = load_artifacts()
    if model is None:
        raise RuntimeError("Model not found.")
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    preds  = model.predict(X)
    probas = model.predict_proba(X)
    labels = encoder.inverse_transform(preds)
    result = df.copy()
    result["pred_band"] = labels
    for i, cls in enumerate(encoder.classes_):
        result[f"prob_{encoder.inverse_transform([i])[0]}"] = probas[:, i].round(4)
    return result

# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(row: pd.Series) -> list:
    """Generate actionable HR recommendations for a single employee."""
    recs = []
    for feat, config in INTERVENTIONS.items():
        if feat not in row.index:
            continue
        val = row[feat]
        thr = config["threshold"]
        if config["direction"] == "low" and val < thr:
            severity = "High" if val < thr * 0.6 else "Medium"
            recs.append({
                "feature":  feat,
                "value":    val,
                "threshold": thr,
                "action":   config["action"],
                "icon":     config["icon"],
                "severity": severity,
            })
        elif config["direction"] == "high" and val > thr:
            severity = "High" if val > thr * 1.5 else "Medium"
            recs.append({
                "feature":  feat,
                "value":    val,
                "threshold": thr,
                "action":   config["action"],
                "icon":     config["icon"],
                "severity": severity,
            })
    # Sort: High severity first
    recs.sort(key=lambda x: 0 if x["severity"] == "High" else 1)
    return recs

# ─────────────────────────────────────────────────────────────────────────────
def get_feature_importance(top_n: int = 15) -> pd.Series:
    """Return feature importances from the trained RandomForest."""
    model, _ = load_artifacts()
    if model is None:
        return pd.Series(dtype=float)
    rf  = model.named_steps["clf"]
    pre = model.named_steps["pre"]
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(len(rf.feature_importances_))]
    return pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False).head(top_n)
