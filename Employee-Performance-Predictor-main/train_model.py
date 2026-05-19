"""
src/train_model.py
------------------
Full ML pipeline:
  data loading → preprocessing → model training → evaluation → saving
Uses RandomForest with class balancing and macro-F1 optimized GridSearch.
"""

import os, warnings, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "employee_perf_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl")
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "employee_features.csv")
DROP_COLS    = ["employee_id", "performance_score"]   # leakage guards
TARGET       = "perf_band_next"

# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert df[TARGET].notna().all(), "Target column has nulls"
    print(f"📂 Loaded {len(df)} rows | Columns: {df.shape[1]}")
    print("Class distribution:\n", df[TARGET].value_counts())
    return df

# ─────────────────────────────────────────────────────────────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ]), num_cols, cat_cols

# ─────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns])
    y = df[TARGET].astype(str)

    # Encode target labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=13
    )

    pre, num_cols, cat_cols = build_preprocessor(Xtr)

    # ── GridSearch on RandomForest ──────────────────────────────────────────
    rf_pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=13))
    ])
    param_grid = {
        "clf__n_estimators":    [300, 500],
        "clf__max_depth":       [8, 14, None],
        "clf__min_samples_leaf":[1, 3, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    gs = GridSearchCV(rf_pipe, param_grid, scoring="f1_macro",
                      cv=cv, n_jobs=-1, verbose=1)
    gs.fit(Xtr, ytr)

    best = gs.best_estimator_
    print(f"\n✅ Best CV F1_macro : {gs.best_score_:.4f}")
    print(f"   Best params      : {gs.best_params_}")

    # ── Hold-out evaluation ─────────────────────────────────────────────────
    ypred = best.predict(Xte)
    print("\n📊 Classification Report (hold-out test):")
    print(classification_report(yte, ypred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(yte, ypred))

    # ── Feature importances from RF ─────────────────────────────────────────
    try:
        feat_names = best.named_steps["pre"].get_feature_names_out()
        rf_imp     = best.named_steps["clf"].feature_importances_
        imp_series = pd.Series(rf_imp, index=feat_names).sort_values(ascending=False)
        print("\n🔑 Top-15 feature importances (MDI):")
        print(imp_series.head(15))
    except Exception as e:
        imp_series = pd.Series(dtype=float)
        print("Feature importance extraction skipped:", e)

    # ── Persist model & encoder ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    joblib.dump(le,   ENCODER_PATH)
    print(f"\n💾 Model saved   → {MODEL_PATH}")
    print(f"💾 Encoder saved → {ENCODER_PATH}")

    return best, le, imp_series

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_validate(DATA_PATH)
    train(df)
