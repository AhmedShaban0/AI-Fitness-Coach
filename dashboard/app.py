# NTI Fitness Coach AI & Business Intelligence Analytics Platform - Final Production Release
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

logo_path = os.path.join(os.path.dirname(__file__), "assets", "nti_logo.png")
pages_dir = os.path.join(os.path.dirname(__file__), "pages")

# Global Streamlit Page Configuration - Executive Full-Width Layout
st.set_page_config(
    page_title="NTI Fitness Coach AI & Analytics Platform",
    page_icon=logo_path if os.path.exists(logo_path) else "🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom SaaS Dark Theme CSS
css_path = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_path):
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Configure Navigation without any group header ("app" label eliminated)
pages = {
    "": [
        st.Page(os.path.join(pages_dir, "1_🏠_Home.py"), title="Home", icon="🏠", default=True),
        st.Page(os.path.join(pages_dir, "2_📊_Dataset_Explorer.py"), title="Dataset Explorer", icon="📊"),
        st.Page(os.path.join(pages_dir, "3_📈_Analytics_Dashboard.py"), title="Analytics Dashboard", icon="📈"),
        st.Page(os.path.join(pages_dir, "4_⚙️_Machine_Learning.py"), title="Machine Learning", icon="⚙️"),
        st.Page(os.path.join(pages_dir, "5_🎯_Prediction.py"), title="Prediction", icon="🎯"),
        st.Page(os.path.join(pages_dir, "6_📂_Batch_Prediction.py"), title="Batch Prediction", icon="📂"),
        st.Page(os.path.join(pages_dir, "7_🧠_Model_Insights.py"), title="Model Insights", icon="🧠"),
        st.Page(os.path.join(pages_dir, "8_ℹ️_About.py"), title="About", icon="ℹ️")
    ]
}

pg = st.navigation(pages)
pg.run()
