import os
import sys
import textwrap
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "nti_logo.png")
css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "custom.css")

if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from dashboard.components.cards import render_hero_banner, render_glass_kpi_card, render_sidebar_header, render_section_title
from dashboard.components.charts import plot_gauge_meter
from dashboard.utils.data_loader import load_and_process_data
from dashboard.models.train import train_all_models
from dashboard.utils.preprocessing import safe_get_metric, safe_get_model_metrics

# Sidebar Branding
render_sidebar_header(logo_path)

render_hero_banner(
    title="Real-Time Fitness AI Prediction Console",
    subtitle="Interactive Caloric Expenditure & Daily Required Intake Target (TDEE/BMR) Inference Workbench",
    badge_text="AI Inference Engine",
    version="v2.4",
    logo_path=logo_path
)

df = load_and_process_data()
model_results = train_all_models(df)

# ==============================================================================
# RESPONSIVE 2-COLUMN ENTERPRISE SAAS CONSOLE (65% LEFT / 35% RIGHT)
# ==============================================================================
col_left, col_right = st.columns([65, 35])

with col_left:
    st.markdown("### 🎛️ Interactive Profile Input Wizard")

    # --------------------------------------------------------------------------
    # SECTION 1: DEMOGRAPHICS & PHYSICAL ATTRIBUTES
    # --------------------------------------------------------------------------
    render_section_title("1. User Demographics & Physical Attributes", "Core biological baseline variables", icon="👤")
    
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        age = st.number_input("Age (Years)", min_value=12, max_value=90, value=30, step=1)
        height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=175.0, step=1.0)
    with d_col2:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)

    # Calculated BMI Display
    bmi = weight_kg / ((height_cm / 100) ** 2)
    bmi_category = "Normal" if 18.5 <= bmi < 25 else ("Overweight" if 25 <= bmi < 30 else ("Obese" if bmi >= 30 else "Underweight"))
    
    bmi_badge_html = textwrap.dedent(f"""
        <div style="background:rgba(15,28,48,0.75); border:1px solid rgba(255,255,255,0.08); padding:10px 16px; border-radius:10px; display:flex; justify-content:space-between; align-items:center; margin-top:6px; margin-bottom:1.25rem;">
            <div>
                <span style="font-size:0.82rem; color:#B8C7D9; font-weight:600;">Calculated Body Mass Index (BMI):</span>
                <span style="font-size:1.05rem; font-weight:800; color:#CCFED8; margin-left:8px;">{bmi:.1f} kg/m²</span>
            </div>
            <span style="background:rgba(149,186,254,0.15); color:#95BAFE; border:1px solid rgba(149,186,254,0.3); padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:700;">
                {bmi_category} Category
            </span>
        </div>
    """).strip()
    st.markdown(bmi_badge_html, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 2: WORKOUT ACTIVITY & SESSION
    # --------------------------------------------------------------------------
    render_section_title("2. Workout Activity & Session", "Physical exercise type, duration, and intensity parameters", icon="🏋️")
    
    w_col1, w_col2 = st.columns(2)
    with w_col1:
        activities = list(df["activity_type"].unique()) if "activity_type" in df.columns else ["Running", "Cycling", "HIIT", "Swimming", "Walking"]
        activity_type = st.selectbox("Workout Activity Type", options=activities)
        intensity_options = list(df["workout_intensity"].unique()) if "workout_intensity" in df.columns else ["Low", "Medium", "High", "Extreme"]
        intensity = st.selectbox("Workout Intensity", options=intensity_options)

    with w_col2:
        duration_minutes = st.number_input("Workout Duration (Minutes)", min_value=5, max_value=240, value=45, step=5)
        daily_steps = st.number_input("Daily Step Count", min_value=0, max_value=50000, value=8500, step=500)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SECTION 3: PHYSIOLOGICAL & HEALTH METRICS
    # --------------------------------------------------------------------------
    render_section_title("3. Physiological & Health Metrics", "Heart rate dynamics, sleep, and metabolic stress factors", icon="❤️")
    
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        avg_heart_rate = st.number_input("Average Heart Rate (BPM)", min_value=60, max_value=210, value=145, step=1)
        sleep_hours = st.number_input("Sleep Duration (Hours)", min_value=3.0, max_value=14.0, value=7.5, step=0.5)

    with h_col2:
        resting_heart_rate = st.number_input("Resting Heart Rate (BPM)", min_value=40, max_value=110, value=65, step=1)
        stress_level = st.selectbox("Perceived Stress Level", options=["Low", "Medium", "High"])

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Execute AI Model Inference", use_container_width=True)

# ------------------------------------------------------------------------------
# RIGHT COLUMN (≈35%): LIVE AI PREDICTION PANEL
# ------------------------------------------------------------------------------
with col_right:
    st.markdown("### ⚡ Live AI Prediction Panel")

    target_task = st.radio(
        "Select Prediction Goal Target:",
        ["Active Workout Burn", "Daily Required Calories (TDEE)"]
    )

    is_burned = "Active" in target_task
    model_key = "calories_burned" if is_burned else "required_calories"

    # Defensive Model Results & Metrics Fetching
    if not isinstance(model_results, dict) or model_key not in model_results:
        st.warning("⚠️ No evaluation metrics available for the selected task.")
        task_data = {}
        best_model_name = "Random Forest"
        best_model = None
        metrics = {}
        encoders = {}
        scaler = None
        feature_names = []
    else:
        task_data = model_results.get(model_key, {})
        best_model_name = task_data.get("best_model_name", "Random Forest")
        best_model = task_data.get("best_model")
        
        _, metrics = safe_get_model_metrics(model_results, model_key)
        
        data_block = task_data.get("data", {})
        encoders = data_block.get("encoders", {})
        scaler = data_block.get("scaler")
        feature_names = data_block.get("feature_names", [])

    r2_val = safe_get_metric(metrics, "test_r2", default=None)
    mae_val = safe_get_metric(metrics, "test_mae", default=None)

    r2_display = f"{r2_val*100:.2f}%" if isinstance(r2_val, float) else "N/A"
    mae_display = f"± {mae_val:.2f} kcal" if isinstance(mae_val, float) else "N/A"
    mae = mae_val if isinstance(mae_val, float) else 1.2

    # --------------------------------------------------------------------------
    # STRICT INFERENCE PREPROCESSING PIPELINE
    # --------------------------------------------------------------------------
    if best_model is None or scaler is None:
        predicted_val = 312.4
        st.warning("⚠️ Model estimator or scaler is not initialized properly.")
    else:
        # Numeric stress level mapping fallback if stress_level is in num_cols
        stress_numeric_map = {"Low": 3, "Medium": 5, "High": 8}
        num_cols = data_block.get("num_cols", [])
        stress_num_val = stress_numeric_map.get(stress_level, 5)

        input_dict = {
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "activity_type": activity_type,
            "duration_minutes": duration_minutes,
            "duration_min": duration_minutes,
            "intensity": intensity,
            "workout_intensity": intensity,
            "daily_steps": daily_steps,
            "avg_heart_rate": avg_heart_rate,
            "heart_rate": avg_heart_rate,
            "resting_heart_rate": resting_heart_rate,
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "endurance_level": 5.0,
            "sleep_hours": sleep_hours,
            "stress_level": stress_num_val if "stress_level" in num_cols else stress_level,
            "hydration_level": 2.5,
            "smoking_status": "Never",
            "fitness_level": 0.5
        }

        # Calculate calories_burned for required_calories target if needed
        from dashboard.utils.data_loader import calculate_calories
        calculated_burn = calculate_calories(input_dict)
        input_dict["calories_burned"] = calculated_burn

        input_df = pd.DataFrame([input_dict])

        # 1. Categorical Encoding Step (Target-Specific LabelEncoders)
        for col in list(input_df.columns):
            if col in encoders:
                enc = encoders[col]
                val_str = str(input_df[col].iloc[0])
                if val_str in enc.classes_:
                    input_df[col] = enc.transform([val_str])[0]
                else:
                    # Match gender ('Male' -> 'M') or case-insensitive match
                    matched = False
                    for cls in enc.classes_:
                        if str(cls).lower() == val_str.lower() or str(cls)[0].lower() == val_str[0].lower():
                            input_df[col] = enc.transform([cls])[0]
                            matched = True
                            break
                    if not matched:
                        input_df[col] = 0

        # Convert remaining object/string columns to numeric
        for col in list(input_df.columns):
            if input_df[col].dtype == "object" or isinstance(input_df[col].iloc[0], str):
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0.0)

        # 2. Verify num_cols contains ONLY numeric columns before scaling
        valid_num_cols = [c for c in num_cols if c in input_df.columns]
        non_num_in_num_cols = input_df[valid_num_cols].select_dtypes(include=["object", "string"]).columns.tolist()

        if non_num_in_num_cols:
            st.warning(f"⚠️ Preprocessing Pipeline Alert: Non-numeric column(s) {non_num_in_num_cols} detected before numerical scaling step.")
            predicted_val = 0.0
        else:
            # 3. Numeric Scaling (Target-Specific StandardScaler)
            if scaler is not None and valid_num_cols:
                input_df[valid_num_cols] = scaler.transform(input_df[valid_num_cols])

            # 4. Feature Alignment to Exact Target Training Order
            missing_feats = [f for f in feature_names if f not in input_df.columns]
            for mf in missing_feats:
                input_df[mf] = 0.0

            input_df = input_df[feature_names]

            # 5. Final Validation Check
            str_in_final = input_df.select_dtypes(include=["object", "string"]).columns.tolist()
            if str_in_final:
                st.warning(f"⚠️ Preprocessing Validation Alert: Final feature matrix contains unencoded string column(s) {str_in_final}.")
                predicted_val = 0.0
            else:
                # 6. Execute Model Inference
                predicted_val = float(best_model.predict(input_df)[0])
                predicted_val = max(0, round(predicted_val, 1))

    # Main Output Display Box
    task_label = "Predicted Active Caloric Burn" if is_burned else "Predicted Daily TDEE Target"
    res_card_html = textwrap.dedent(f"""
        <div class="kpi-card" style="text-align:center; padding:1.5rem; margin-bottom:1rem;">
            <div style="font-size:0.78rem; color:#B8C7D9; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">{task_label}</div>
            <div style="font-size:2.4rem; font-weight:800; color:#CCFED8; margin:0.4rem 0;">{predicted_val} kcal</div>
            <div style="font-size:0.8rem; color:#95BAFE;">Confidence: {predicted_val - mae:.1f} &ndash; {predicted_val + mae:.1f} kcal ({mae_display})</div>
        </div>
    """).strip()
    st.markdown(res_card_html, unsafe_allow_html=True)

    # Large Gauge Meter
    max_g = 1200 if is_burned else 4500
    g_title = f"Burn Gauge ({best_model_name})" if is_burned else f"TDEE Gauge ({best_model_name})"
    fig_gauge = plot_gauge_meter(predicted_val, min_val=0, max_val=max_g, title=g_title, unit="kcal")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Small Model KPI Metrics Summary
    p_kpi1, p_kpi2 = st.columns(2)
    with p_kpi1:
        render_glass_kpi_card("Winning Engine", best_model_name, "Optimal Estimator", icon="⚙️", card_type="accent")
    with p_kpi2:
        render_glass_kpi_card("Accuracy R²", r2_display, mae_display, icon="🎯", card_type="secondary")

    # Actionable Health & Metabolic Advisory
    st.markdown("#### 💡 Metabolic Advisory")
    if is_burned:
        if predicted_val > 400:
            rec_text = "🔥 **High Expenditure Session!** Rehydrate with electrolytes and ensure 25-30g protein intake."
        else:
            rec_text = "💪 **Moderate Caloric Burn!** Excellent session for metabolic conditioning and endurance."
    else:
        rec_text = f"🎯 **Daily TDEE Baseline Target:** Maintain daily caloric intake around **{predicted_val:.0f} kcal**."

    st.success(rec_text)

st.markdown('<div class="footer-text">© 2026 National Telecommunications Institute (NTI) &bull; AI & Data Science Division</div>', unsafe_allow_html=True)
