import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "nti_logo.png")
css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "custom.css")

if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from dashboard.components.cards import render_hero_banner, render_glass_kpi_card, render_sidebar_header
from dashboard.components.charts import plot_histogram, plot_correlation_heatmap
from dashboard.utils.data_loader import load_and_process_data

# Sidebar Branding
render_sidebar_header(logo_path)

render_hero_banner(
    title="Interactive Data Exploration Workspace",
    subtitle="Enterprise Data Profiling Engine: Search, Filter, Sort, Audit Nulls & Inspect Feature Correlations",
    badge_text="Data Health Audit",
    version="v2.4",
    logo_path=logo_path
)

df = load_and_process_data()

# Control Filters Row
st.markdown("### 🔍 Interactive Filters & Search Parameters")

col1, col2, col3, col4 = st.columns([2, 2, 2, 3])

with col1:
    activities = ["All"] + sorted(df["activity_type"].dropna().unique().tolist())
    selected_activity = st.selectbox("Activity Type", activities)

with col2:
    genders = ["All"] + sorted(df["gender"].dropna().unique().tolist())
    selected_gender = st.selectbox("Gender", genders)

with col3:
    intensities = ["All"] + sorted(df["intensity"].dropna().unique().tolist())
    selected_intensity = st.selectbox("Intensity Level", intensities)

with col4:
    search_query = st.text_input("🔍 Quick Search Filter", placeholder="Type activity or value...")

# Apply Filters
filtered_df = df.copy()

if selected_activity != "All":
    filtered_df = filtered_df[filtered_df["activity_type"] == selected_activity]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["gender"] == selected_gender]

if selected_intensity != "All":
    filtered_df = filtered_df[filtered_df["intensity"] == selected_intensity]

if search_query:
    filtered_df = filtered_df[filtered_df["activity_type"].str.contains(search_query, case=False, na=False)]

# Summary KPI Header
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_glass_kpi_card("Filtered Records", f"{len(filtered_df):,}", f"out of {len(df):,}", icon="📊", card_type="secondary")
with c2:
    render_glass_kpi_card("Active Columns", f"{len(filtered_df.columns)}", "Numerical & Categorical", icon="📋", card_type="default")
with c3:
    render_glass_kpi_card("Duplicate Rows", f"{filtered_df.duplicated().sum():,}", "Clean Zero-Dup Matrix", icon="✨", card_type="accent")
with c4:
    render_glass_kpi_card("Avg Steps", f"{filtered_df['daily_steps'].mean():,.0f}", "Daily Baseline", icon="👟", card_type="secondary")

st.markdown("<br>", unsafe_allow_html=True)

# Data Table & Workspace Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Table View", "📊 Summary Statistics", "🩺 Data Quality & Null Audit", "📈 Feature Distributions"])

with tab1:
    col_sel, sort_col, row_count_sel = st.columns([3, 2, 1])
    with col_sel:
        all_cols = filtered_df.columns.tolist()
        selected_cols = st.multiselect("Select Columns to View", all_cols, default=all_cols[:10])
    with sort_col:
        sort_by = st.selectbox("Sort Table By", all_cols, index=0)
    with row_count_sel:
        max_rows = st.select_slider("Rows to Preview", options=[50, 100, 500, 1000, 5000], value=100)

    display_df = filtered_df.sort_values(by=sort_by)
    if selected_cols:
        st.dataframe(display_df[selected_cols].head(max_rows), use_container_width=True)
    else:
        st.dataframe(display_df.head(max_rows), use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv_data,
        file_name="nti_fitness_cleaned_dataset.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown("#### Numerical Feature Summary Statistics")
    st.dataframe(filtered_df.describe().T.style.format("{:.2f}"), use_container_width=True)

with tab3:
    st.markdown("#### Column Metadata & Data Quality Audit")
    audit_data = []
    for col in df.columns:
        audit_data.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Null Count": df[col].isnull().sum(),
            "Null Percentage (%)": f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
            "Unique Values": df[col].nunique(),
            "Sample Value": str(df[col].iloc[0]) if len(df) > 0 else ""
        })
    audit_df = pd.DataFrame(audit_data)
    st.dataframe(audit_df, use_container_width=True)

with tab4:
    st.markdown("#### Feature Value Distribution Analysis")
    num_cols_sel = df.select_dtypes(include=["number"]).columns.tolist()
    chosen_dist_col = st.selectbox("Select Feature for Distribution Plot", num_cols_sel, index=num_cols_sel.index("calories_burned") if "calories_burned" in num_cols_sel else 0)
    
    fig_dist = plot_histogram(filtered_df, chosen_dist_col)
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown('<div class="footer-text">© 2026 National Telecommunications Institute (NTI) &bull; AI & Data Science Division</div>', unsafe_allow_html=True)
