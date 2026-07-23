import streamlit as st
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
except ImportError as e:
    st.error(f"❌ Missing ML dependency: {e}. Please run 'pip install -r requirements.txt'")

from dashboard.utils.preprocessing import (
    RANDOM_STATE,
    TEST_SIZE,
    CALORIES_BURNED_EXCLUDE,
    REQUIRED_CALORIES_EXCLUDE,
    encode_categorical_features,
    scale_numerical_features,
    get_metrics
)

@st.cache_resource(show_spinner="Training & Benchmark Evaluating Machine Learning Models...")
def train_all_models(df, sample_size=100000):
    """
    Train 3 regression models (Linear Regression, Random Forest, XGBoost)
    for both 'calories_burned' and 'required_calories'.
    Dynamically selects the winning model based on real validation metrics.
    """
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()
    else:
        df_sample = df.copy()

    df_cleaned = df_sample.dropna().copy()

    results = {
        "calories_burned": {},
        "required_calories": {}
    }

    # ==========================================
    # MODULE 1: CALORIES BURNED MODELS
    # ==========================================
    X_burned = df_cleaned.drop(columns=[c for c in CALORIES_BURNED_EXCLUDE if c in df_cleaned.columns])
    y_burned = df_cleaned["calories_burned"]

    cat_cols_b = X_burned.select_dtypes(include="object").columns.tolist()
    num_cols_b = X_burned.select_dtypes(include=["number"]).columns.tolist()

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_burned, y_burned, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_b_enc, X_test_b_enc, encoders_b = encode_categorical_features(X_train_b, X_test_b, cat_cols_b)
    X_train_b_sc, X_test_b_sc, scaler_b = scale_numerical_features(X_train_b_enc, X_test_b_enc, num_cols_b)

    models_b = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results["calories_burned"]["data"] = {
        "X_train": X_train_b_sc,
        "X_test": X_test_b_sc,
        "y_train": y_train_b,
        "y_test": y_test_b,
        "encoders": encoders_b,
        "scaler": scaler_b,
        "cat_cols": cat_cols_b,
        "num_cols": num_cols_b,
        "feature_names": X_train_b_sc.columns.tolist()
    }

    best_score_b = -1.0
    best_name_b = "Random Forest"

    for name, model in models_b.items():
        model.fit(X_train_b_sc, y_train_b)
        tr_pred = model.predict(X_train_b_sc)
        te_pred = model.predict(X_test_b_sc)

        metrics = get_metrics(y_train_b, tr_pred, y_test_b, te_pred)
        results["calories_burned"][name] = {
            "model": model,
            "metrics": metrics,
            "train_preds": tr_pred,
            "test_preds": te_pred
        }
        
        if metrics["test_r2"] > best_score_b:
            best_score_b = metrics["test_r2"]
            best_name_b = name

        if hasattr(model, "feature_importances_"):
            results["calories_burned"][name]["feature_importances"] = pd.Series(
                model.feature_importances_, index=X_train_b_sc.columns
            ).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            results["calories_burned"][name]["feature_importances"] = pd.Series(
                np.abs(model.coef_), index=X_train_b_sc.columns
            ).sort_values(ascending=False)

    results["calories_burned"]["best_model_name"] = best_name_b
    results["calories_burned"]["best_model"] = results["calories_burned"][best_name_b]["model"]
    results["calories_burned"]["best_metrics"] = results["calories_burned"][best_name_b]["metrics"]
    results["calories_burned"]["best_score"] = best_score_b

    # ==========================================
    # MODULE 2: REQUIRED CALORIES MODELS
    # ==========================================
    X_req = df_cleaned.drop(columns=[c for c in REQUIRED_CALORIES_EXCLUDE if c in df_cleaned.columns])
    y_req = df_cleaned["required_calories"]

    cat_cols_r = X_req.select_dtypes(include="object").columns.tolist()
    num_cols_r = X_req.select_dtypes(include=["number"]).columns.tolist()

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_req, y_req, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_r_enc, X_test_r_enc, encoders_r = encode_categorical_features(X_train_r, X_test_r, cat_cols_r)
    X_train_r_sc, X_test_r_sc, scaler_r = scale_numerical_features(X_train_r_enc, X_test_r_enc, num_cols_r)

    models_r = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1)
    }

    results["required_calories"]["data"] = {
        "X_train": X_train_r_sc,
        "X_test": X_test_r_sc,
        "y_train": y_train_r,
        "y_test": y_test_r,
        "encoders": encoders_r,
        "scaler": scaler_r,
        "cat_cols": cat_cols_r,
        "num_cols": num_cols_r,
        "feature_names": X_train_r_sc.columns.tolist()
    }

    best_score_r = -1.0
    best_name_r = "Random Forest"

    for name, model in models_r.items():
        model.fit(X_train_r_sc, y_train_r)
        tr_pred = model.predict(X_train_r_sc)
        te_pred = model.predict(X_test_r_sc)

        metrics = get_metrics(y_train_r, tr_pred, y_test_r, te_pred)
        results["required_calories"][name] = {
            "model": model,
            "metrics": metrics,
            "train_preds": tr_pred,
            "test_preds": te_pred
        }

        if metrics["test_r2"] > best_score_r:
            best_score_r = metrics["test_r2"]
            best_name_r = name

        if hasattr(model, "feature_importances_"):
            results["required_calories"][name]["feature_importances"] = pd.Series(
                model.feature_importances_, index=X_train_r_sc.columns
            ).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            results["required_calories"][name]["feature_importances"] = pd.Series(
                np.abs(model.coef_), index=X_train_r_sc.columns
            ).sort_values(ascending=False)

    results["required_calories"]["best_model_name"] = best_name_r
    results["required_calories"]["best_model"] = results["required_calories"][best_name_r]["model"]
    results["required_calories"]["best_metrics"] = results["required_calories"][best_name_r]["metrics"]
    results["required_calories"]["best_score"] = best_score_r

    return results
