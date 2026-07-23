import os
import sys
import time
import textwrap
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "nti_logo.png")
css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "custom.css")

if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from dashboard.components.cards import render_hero_banner, render_glass_kpi_card, render_sidebar_header
from dashboard.utils.data_loader import load_and_process_data
from dashboard.models.train import train_all_models

# Sidebar Branding
render_sidebar_header(logo_path)

render_hero_banner(
    title="Enterprise Batch Prediction Engine",
    subtitle="Bulk Fitness Log CSV Processing, High-Throughput Inference Pipeline & Results Export",
    badge_text="Batch Processing Pipeline",
    version="v2.4",
    logo_path=logo_path
)

df = load_and_process_data()
model_results = train_all_models(df)

st.markdown("### 📤 Upload Fitness CSV File for Batch Inference")

uploaded_file = st.file_uploader(
    "Drag and drop your fitness logs CSV file here",
    type=["csv"],
    help="Upload a CSV file containing fitness activity records (columns: age, gender, height_cm, weight_kg, activity_type, duration_minutes, intensity, etc.)"
)

# Demo / Sample CSV Generator Option
st.markdown("#### 💡 Need a Sample File to Test?")
sample_df = df.sample(min(len(df), 200), random_state=42).copy()

sample_csv = sample_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Sample Batch Test File (200 Records CSV)",
    data=sample_csv,
    file_name="sample_fitness_batch_input.csv",
    mime="text/csv"
)

st.markdown("<br>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
        st.success(f"✅ File Uploaded Successfully! Found **{len(input_data):,} rows** and **{len(input_data.columns)} columns**.")

        st.markdown("#### 📄 Uploaded Data Preview")
        st.dataframe(input_data.head(10), use_container_width=True)

        target_task = st.radio(
            "Select Prediction Task for Batch Execution:",
            ["Calories Burned Prediction", "Required Daily Calories Prediction"],
            horizontal=True
        )

        if st.button("🚀 Execute High-Speed Batch Inference"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for percent_complete in range(1, 101, 20):
                time.sleep(0.06)
                progress_bar.progress(percent_complete)
                status_text.text(f"Processing batch vectorization... {percent_complete}%")

            model_key = "calories_burned" if "Burned" in target_task else "required_calories"
            task_data = model_results.get(model_key, {})
            best_model_name = task_data.get("best_model_name", "Random Forest")
            best_model = task_data.get("best_model")
            data_block = task_data.get("data", {})
            encoders = data_block.get("encoders", {})
            scaler = data_block.get("scaler")
            feature_names = data_block.get("feature_names", [])

            proc_df = input_data.copy()
            if "bmi" not in proc_df.columns and "weight_kg" in proc_df.columns and "height_cm" in proc_df.columns:
                proc_df["bmi"] = proc_df["weight_kg"] / ((proc_df["height_cm"] / 100) ** 2)

            # Column Schema Aliases
            if "duration_minutes" in proc_df.columns and "duration_min" not in proc_df.columns:
                proc_df["duration_min"] = proc_df["duration_minutes"]
            if "intensity" in proc_df.columns and "workout_intensity" not in proc_df.columns:
                proc_df["workout_intensity"] = proc_df["intensity"]
            if "avg_heart_rate" in proc_df.columns and "heart_rate" not in proc_df.columns:
                proc_df["heart_rate"] = proc_df["avg_heart_rate"]

            # 1. Categorical Encoding Step
            for col in list(proc_df.columns):
                if col in encoders:
                    enc = encoders[col]
                    proc_df[col] = proc_df[col].astype(str).map(
                        lambda x: enc.transform([x])[0] if x in enc.classes_ else 0
                    )

            # Convert remaining object/string columns to numeric
            for col in list(proc_df.columns):
                if proc_df[col].dtype == "object":
                    proc_df[col] = pd.to_numeric(proc_df[col], errors="coerce").fillna(0.0)

            # 2. Scaling
            num_cols = data_block.get("num_cols", [])
            valid_num_cols = [c for c in num_cols if c in proc_df.columns]
            if scaler is not None and valid_num_cols:
                proc_df[valid_num_cols] = scaler.transform(proc_df[valid_num_cols])

            # 3. Feature Alignment
            for col in feature_names:
                if col not in proc_df.columns:
                    proc_df[col] = 0.0

            proc_df = proc_df[feature_names]
            preds = best_model.predict(proc_df)
            preds = [max(0, round(float(p), 1)) for p in preds]

            out_col_name = "predicted_calories_burned" if "Burned" in target_task else "predicted_required_calories"
            output_df = input_data.copy()
            output_df[out_col_name] = preds

            progress_bar.progress(100)
            status_text.text(f"✅ Batch Inference Completed via {best_model_name}!")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Batch Prediction Summary")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_glass_kpi_card("Total Predicted", f"{len(preds):,}", f"Engine: {best_model_name}", icon="📂", card_type="secondary")
            with c2:
                render_glass_kpi_card("Mean Prediction", f"{pd.Series(preds).mean():.1f} kcal", "Cohort Average", icon="🔥", card_type="accent")
            with c3:
                render_glass_kpi_card("Min Prediction", f"{pd.Series(preds).min():.1f} kcal", "Lowest Output", icon="📉", card_type="default")
            with c4:
                render_glass_kpi_card("Max Prediction", f"{pd.Series(preds).max():.1f} kcal", "Highest Output", icon="📈", card_type="secondary")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📥 Download Predictions Output File")

            output_csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Predictions File (CSV)",
                data=output_csv,
                file_name=f"nti_batch_{model_key}_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error processing batch file: {str(e)}")

st.markdown('<div class="footer-text">© 2026 National Telecommunications Institute (NTI) &bull; AI & Data Science Division</div>', unsafe_allow_html=True)
