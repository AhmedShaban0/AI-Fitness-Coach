import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "nti_logo.png")
css_path = os.path.join(os.path.dirname(__file__), "..", "styles", "custom.css")

if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from dashboard.components.cards import render_hero_banner, render_glass_kpi_card, render_sidebar_header

# Sidebar Branding
render_sidebar_header(logo_path)

render_hero_banner(
    title="About NTI Fitness Coach AI Platform",
    subtitle="Enterprise AI Analytics & Health Optimization SaaS — National Telecommunications Institute (NTI)",
    badge_text="NTI AI & Data Science Division",
    version="v2.4 Production",
    logo_path=logo_path
)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🏛️ Institution & Project Overview")
    st.markdown(
        """
        <div class="glass-card" style="line-height:1.7; color:#B8C7D9;">
            <p>
                The <b>NTI Fitness Coach AI & Analytics Platform</b> is an enterprise machine learning solution engineered by the 
                <b>National Telecommunications Institute (NTI)</b> AI & Data Science Division.
            </p>
            <p>
                Designed to bridge sports science, metabolic physiology, and modern machine learning engineering, 
                this platform delivers real-time dual-target predictions for active caloric expenditure during workouts and daily required intake targets (TDEE/BMR).
            </p>
            <div style="margin-top:1rem; display:flex; gap:12px; flex-wrap:wrap;">
                <span style="background:rgba(27,117,187,0.2); color:#95BAFE; border:1px solid rgba(27,117,187,0.3); padding:6px 14px; border-radius:8px; font-size:0.85rem; font-weight:700;">🏢 NTI AI Division</span>
                <span style="background:rgba(204,254,216,0.15); color:#CCFED8; border:1px solid rgba(204,254,216,0.3); padding:6px 14px; border-radius:8px; font-size:0.85rem; font-weight:700;">📜 MIT Open Source</span>
                <span style="background:rgba(149,186,254,0.15); color:#D8E8FF; border:1px solid rgba(149,186,254,0.3); padding:6px 14px; border-radius:8px; font-size:0.85rem; font-weight:700;">⚡ Production Build v2.4</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("### 📚 Dataset Provenance & Metadata")
    st.markdown(
        """
        <div class="glass-card" style="line-height:1.7; color:#B8C7D9;">
            <div style="font-size:0.85rem; text-transform:uppercase; color:#95BAFE; font-weight:700; margin-bottom:0.4rem;">Dataset Specifications</div>
            <ul style="padding-left:1.2rem; margin:0;">
                <li><b>Total Observations:</b> 687,701 Records</li>
                <li><b>Feature Count:</b> 22 Variables</li>
                <li><b>Data Cleanliness:</b> 100% Valid (Negative steps removed, MET calibrated)</li>
                <li><b>Target Variables:</b> <code>calories_burned</code> & <code>required_calories</code></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# System Architecture
st.markdown("### 🏗️ Platform System Architecture")
st.markdown(
    """
    <div class="glass-card" style="text-align:center; padding:2rem;">
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px;">
            <div style="background:#0F1C30; border:1px solid rgba(255,255,255,0.08); padding:16px; border-radius:12px;">
                <div style="font-size:1.5rem;">📱 Presentation Layer</div>
                <div style="font-size:0.85rem; color:#95BAFE; margin-top:6px; font-weight:700;">Streamlit & Custom Dark CSS</div>
                <div style="font-size:0.78rem; color:#B8C7D9; margin-top:4px;">Responsive Glassmorphism SaaS UI</div>
            </div>
            <div style="background:#0F1C30; border:1px solid rgba(255,255,255,0.08); padding:16px; border-radius:12px;">
                <div style="font-size:1.5rem;">📊 Data & Charting Layer</div>
                <div style="font-size:0.85rem; color:#95BAFE; margin-top:6px; font-weight:700;">Plotly Express & Pandas</div>
                <div style="font-size:0.78rem; color:#B8C7D9; margin-top:4px;">Dark Theme Interactive Charts & Insights</div>
            </div>
            <div style="background:#0F1C30; border:1px solid rgba(255,255,255,0.08); padding:16px; border-radius:12px;">
                <div style="font-size:1.5rem;">⚙️ Inference & ML Engine</div>
                <div style="font-size:0.78rem; color:#CCFED8; margin-top:6px; font-weight:700;">Scikit-Learn & XGBoost Pipeline</div>
                <div style="font-size:0.78rem; color:#B8C7D9; margin-top:4px;">Dual-Target Cached Training (@st.cache_resource)</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="footer-text">© 2026 National Telecommunications Institute (NTI) &bull; AI & Data Science Division</div>', unsafe_allow_html=True)
