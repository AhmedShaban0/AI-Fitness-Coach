import os
import base64
import textwrap
import streamlit as st

def get_base64_logo(logo_path):
    if logo_path and os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as image_file:
                return f'<img src="data:image/png;base64,{base64.b64encode(image_file.read()).decode()}" class="hero-logo-img" alt="NTI Logo" />'
        except Exception:
            pass
    return '<div class="hero-logo-img" style="font-weight:800; color:#1B75BB; display:flex; align-items:center; justify-content:center; width: auto; padding:6px 14px;">🏛️ NTI AI</div>'

def render_sidebar_header(logo_path=None):
    """Renders a prominent, high-impact enterprise sidebar header with logo and NTI accreditation."""
    logo_html = ""
    if logo_path and os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{encoded}" class="sidebar-logo-img" alt="NTI Logo" />'
        except Exception:
            logo_html = '<div style="font-size:2rem; margin-bottom:0.5rem;">🏋️</div>'

    html = textwrap.dedent(f"""
        <div class="sidebar-branding-box">
            {logo_html}
            <div class="sidebar-title">NTI Fitness Coach AI</div>
            <div class="sidebar-subtitle">AI & Data Science Division</div>
            <div style="margin-top:8px; font-size:0.7rem; color:#B8C7D9; background:rgba(255,255,255,0.06); padding:4px 8px; border-radius:6px; display:inline-block;">
                Enterprise SaaS Platform &bull; v2.4
            </div>
        </div>
    """).strip()
    st.sidebar.markdown(html, unsafe_allow_html=True)

def render_hero_banner(title, subtitle, badge_text="NTI Enterprise AI Engine", version="v2.4", logo_path=None):
    """Renders an enterprise SaaS-style Hero Banner with dark theme glow and proportional NTI logo fallback."""
    logo_html = get_base64_logo(logo_path)

    html = textwrap.dedent(f"""
        <div class="hero-banner">
            <div class="hero-logo-box">
                {logo_html}
                <div class="hero-badge">
                    <span>⚡</span> {badge_text} &bull; {version}
                </div>
            </div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)

def render_header(title, subtitle, logo_path=None):
    """API Adapter for render_header."""
    render_hero_banner(title, subtitle, logo_path=logo_path)

def render_glass_kpi_card(title, value, subtitle=None, icon="📊", card_type="default"):
    """
    Renders a standardized KPI card with 22px border radius, top-to-bottom gradient border (#CCFED8 to #95BAFE),
    subtle outer glow, and translateY(-4px) hover effect.
    """
    sub_html = f'<div class="card-desc" title="{subtitle}">{subtitle}</div>' if subtitle else ''
    
    html = textwrap.dedent(f"""
        <div class="kpi-card">
            <div class="card-icon-wrapper">{icon}</div>
            <div class="card-title" title="{title}">{title}</div>
            <div class="card-value" title="{value}">{value}</div>
            {sub_html}
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)

def render_metric_card(title, value, subtitle=None, icon="📊", card_type="default"):
    """API Adapter for render_metric_card."""
    render_glass_kpi_card(title, value, subtitle=subtitle, icon=icon, card_type=card_type)

def render_feature_card(title, description, icon="🚀"):
    """Renders a feature summary card with clean icon deduplication."""
    clean_title = title
    if icon and title.strip().startswith(icon.strip()):
        clean_title = title.strip()[len(icon.strip()):].strip()

    html = textwrap.dedent(f"""
        <div class="feature-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <div class="feature-title" style="margin:0;">{clean_title}</div>
            </div>
            <div class="feature-desc">{description}</div>
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)

def render_model_card(model_name, r2_score, mae_score, rmse_score, is_best=False):
    """Renders a responsive model evaluation metric summary card with 22px gradient border."""
    badge = '<span style="background:rgba(204,254,216,0.2); color:#CCFED8; border:1px solid #CCFED8; border-radius:12px; padding:3px 10px; font-size:0.75rem; font-weight:700; white-space:nowrap;">★ BEST MODEL</span>' if is_best else ''
    
    r2_str = f"{r2_score:.4f}" if isinstance(r2_score, (int, float)) else str(r2_score)
    mae_str = f"{mae_score:.2f}" if isinstance(mae_score, (int, float)) else str(mae_score)
    rmse_str = f"{rmse_score:.2f}" if isinstance(rmse_score, (int, float)) else str(rmse_score)

    html = textwrap.dedent(f"""
        <div class="kpi-card" style="margin-bottom:1.25rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.85rem; flex-wrap:nowrap; gap:8px;">
                <div style="font-size:1.1rem; font-weight:800; color:#FFFFFF; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{model_name}</div>
                {badge}
            </div>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; text-align:center;">
                <div style="background:rgba(15,28,48,0.7); padding:8px 4px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:0.72rem; color:#B8C7D9; font-weight:600; white-space:nowrap;">R² Score</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#95BAFE; white-space:nowrap; margin-top:2px;">{r2_str}</div>
                </div>
                <div style="background:rgba(15,28,48,0.7); padding:8px 4px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:0.72rem; color:#B8C7D9; font-weight:600; white-space:nowrap;">MAE (kcal)</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#CCFED8; white-space:nowrap; margin-top:2px;">{mae_str}</div>
                </div>
                <div style="background:rgba(15,28,48,0.7); padding:8px 4px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:0.72rem; color:#B8C7D9; font-weight:600; white-space:nowrap;">RMSE (kcal)</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#489CE2; white-space:nowrap; margin-top:2px;">{rmse_str}</div>
                </div>
            </div>
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)

def render_section_title(title, subtitle=None, icon="📌"):
    """Renders a section title with optional subtitle."""
    sub_html = f'<div style="font-size:0.9rem; color:var(--text-muted); margin-top:4px;">{subtitle}</div>' if subtitle else ''
    html = textwrap.dedent(f"""
        <div style="margin: 1.5rem 0 1rem 0;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>{icon}</span> {title}
            </div>
            {sub_html}
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)

def render_info_card(title, content, icon="💡", card_type="info"):
    """Renders a structured info card."""
    html = textwrap.dedent(f"""
        <div class="kpi-card">
            <div style="font-weight:700; color:var(--secondary); font-size:1rem; margin-bottom:0.4rem; display:flex; align-items:center; gap:6px;">
                <span>{icon}</span> {title}
            </div>
            <div style="font-size:0.9rem; color:var(--text-muted); line-height:1.6;">{content}</div>
        </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)
