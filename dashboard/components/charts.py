import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

# Strict NTI Dark Theme Color Palette
PRIMARY = "#1B75BB"
PRIMARY_DARK = "#125A93"
PRIMARY_LIGHT = "#489CE2"
SECONDARY = "#95BAFE"
SECONDARY_LIGHT = "#D8E8FF"
ACCENT = "#CCFED8"
ACCENT_DARK = "#A8F3BC"

BG_CANVAS = "#081220"
BG_CARD = "#142540"
BG_SURFACE = "#0F1C30"
TEXT_MAIN = "#FFFFFF"
TEXT_MUTED = "#B8C7D9"
BORDER_COLOR = "rgba(255, 255, 255, 0.08)"

COLOR_SEQUENCE = ["#1B75BB", "#95BAFE", "#CCFED8", "#489CE2", "#10B981", "#F59E0B", "#EC4899"]

def apply_plotly_theme(fig, height=450):
    """Applies high-end SaaS dark mode styling matching palette tokens."""
    if fig is None:
        fig = go.Figure()

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color=TEXT_MUTED),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_SURFACE,
        margin=dict(l=45, r=35, t=50, b=45),
        title_font=dict(size=16, color=TEXT_MAIN, family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color=TEXT_MUTED),
            title_font=dict(color=TEXT_MAIN)
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color=TEXT_MUTED),
            title_font=dict(color=TEXT_MAIN)
        ),
        legend=dict(
            font=dict(color=TEXT_MUTED),
            bgcolor="rgba(15, 28, 48, 0.7)",
            bordercolor=BORDER_COLOR
        ),
        hoverlabel=dict(
            bgcolor=BG_CARD,
            font_size=13,
            font_color=TEXT_MAIN,
            font_family="Plus Jakarta Sans, sans-serif"
        ),
        height=height
    )
    return fig

def render_business_insight(title: str, text: str):
    """Renders a styled business insight box below charts."""
    st.markdown(
        f"""
        <div class="business-insight-box">
            <div class="business-insight-title">💡 Analytical Insight &bull; {title}</div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_correlation_heatmap(df):
    if df is None or len(df) == 0:
        return apply_plotly_theme(go.Figure())

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return apply_plotly_theme(go.Figure())

    corr = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=["#0F1C30", "#1B75BB", "#95BAFE", "#CCFED8"],
        title="Feature Correlation Matrix"
    )
    return apply_plotly_theme(fig, height=520)

def plot_histogram(df, column, nbins=35, title=None):
    if df is None or column not in df.columns:
        return apply_plotly_theme(go.Figure())

    col_name = str(column).replace('_', ' ').title()
    if title is None:
        title = f"Distribution of {col_name}"
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        color_discrete_sequence=[PRIMARY],
        title=title,
        marginal="box"
    )
    fig.update_traces(marker_line_width=1, marker_line_color=BORDER_COLOR)
    return apply_plotly_theme(fig, height=450)

def plot_boxplot(df, x_col, y_col):
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    x_name = str(x_col).replace('_', ' ').title()
    y_name = str(y_col).replace('_', ' ').title()
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=x_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=f"{y_name} Distribution across {x_name}"
    )
    return apply_plotly_theme(fig, height=450)

def plot_bar(df, x_col, y_col, title=None, orientation="v"):
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    if title is None:
        title = f"{str(y_col).replace('_', ' ').title()} by {str(x_col).replace('_', ' ').title()}"
        
    fig = px.bar(
        df,
        x=x_col if orientation == "v" else y_col,
        y=y_col if orientation == "v" else x_col,
        orientation=orientation,
        color=y_col if orientation == "v" else x_col,
        color_continuous_scale=["#1B75BB", "#95BAFE", "#CCFED8"],
        title=title
    )
    return apply_plotly_theme(fig, height=450)

def plot_ranked_horizontal_bar(df, x_col="calories_burned", y_col="activity_type", title="Average Calories Burned by Workout Activity"):
    """Horizontal Ranked Bar Chart sorted descending."""
    if df is None or len(df) == 0:
        return apply_plotly_theme(go.Figure())

    grouped = df.groupby(y_col)[x_col].mean().reset_index()
    grouped.columns = [y_col, x_col]
    grouped = grouped.sort_values(by=x_col, ascending=True)

    fig = px.bar(
        grouped,
        x=x_col,
        y=y_col,
        orientation="h",
        text_auto=".1f",
        color=x_col,
        color_continuous_scale=["#1B75BB", "#95BAFE", "#CCFED8"],
        title=title
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Average Calories (kcal)", yaxis_title="Activity Type")
    return apply_plotly_theme(fig, height=450)

def plot_activity_caloric_efficiency(df, activity_col="activity_type", calories_col="calories_burned", duration_col="duration_minutes", title="Caloric Burn Efficiency (kcal / min)"):
    """Calculates and plots mean calories burned per minute by workout activity."""
    if df is None or len(df) == 0 or activity_col not in df.columns or calories_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    dur_c = duration_col if duration_col and duration_col in df.columns else None
    
    temp_df = df.copy()
    if dur_c:
        temp_df["efficiency"] = temp_df[calories_col] / temp_df[dur_c].replace(0, np.nan)
    else:
        temp_df["efficiency"] = temp_df[calories_col] / 45.0

    grouped = temp_df.groupby(activity_col)["efficiency"].mean().reset_index()
    grouped.columns = [activity_col, "Efficiency"]
    grouped = grouped.sort_values(by="Efficiency", ascending=True)

    fig = px.bar(
        grouped,
        x="Efficiency",
        y=activity_col,
        orientation="h",
        text_auto=".1f",
        color="Efficiency",
        color_continuous_scale=["#1B75BB", "#95BAFE", "#CCFED8"],
        title=title
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Burn Rate (kcal / minute)",
        yaxis_title="Activity Type"
    )
    return apply_plotly_theme(fig, height=450)

def plot_intensity_violin(df, y_col="calories_burned", x_col="workout_intensity", title="Calorie Burn Distribution by Workout Intensity"):
    """Violin plot for Calories distribution by Workout Intensity."""
    if df is None or len(df) == 0 or x_col not in df.columns or y_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    sample_df = df.sample(min(len(df), 15000), random_state=42)
    fig = px.violin(
        sample_df,
        x=x_col,
        y=y_col,
        color=x_col,
        box=True,
        points=False,
        color_discrete_sequence=["#1B75BB", "#489CE2", "#95BAFE", "#CCFED8"],
        title=title
    )
    fig.update_layout(xaxis_title="Workout Intensity Level", yaxis_title="Calories Burned (kcal)", showlegend=False)
    return apply_plotly_theme(fig, height=450)

def plot_density_scatter(df, x_col="heart_rate", y_col="calories_burned", title="Density Heatmap: Heart Rate vs Calories Burned"):
    """2D Density Scatter / Hexbin Density Heatmap to prevent overlapping points."""
    if df is None or len(df) == 0 or x_col not in df.columns or y_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    sample_df = df.sample(min(len(df), 25000), random_state=42)
    fig = px.density_heatmap(
        sample_df,
        x=x_col,
        y=y_col,
        nbinsx=35,
        nbinsy=35,
        color_continuous_scale=["#0F1C30", "#125A93", "#1B75BB", "#489CE2", "#95BAFE", "#CCFED8"],
        title=title
    )
    fig.update_layout(
        xaxis_title="Heart Rate (BPM)",
        yaxis_title="Calories Burned (kcal)",
        coloraxis_colorbar=dict(title="Density Count")
    )
    return apply_plotly_theme(fig, height=450)

def plot_stacked_100_bar(df, group_col="gender", category_col="workout_intensity", title="100% Stacked Distribution Breakdown"):
    """100% Stacked Bar Chart for Gender / Intensity distribution."""
    if df is None or len(df) == 0 or group_col not in df.columns or category_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    ct = pd.crosstab(df[group_col], df[category_col], normalize="index") * 100
    ct_df = ct.reset_index().melt(id_vars=group_col, var_name=category_col, value_name="Percentage")

    fig = px.bar(
        ct_df,
        x=group_col,
        y="Percentage",
        color=category_col,
        barmode="stack",
        text_auto=".1f%",
        color_discrete_sequence=["#1B75BB", "#489CE2", "#95BAFE", "#CCFED8"],
        title=title
    )
    fig.update_layout(yaxis_title="Percentage (%)", xaxis_title=str(group_col).replace('_', ' ').title(), legend_title=str(category_col).replace('_', ' ').title())
    return apply_plotly_theme(fig, height=450)

def plot_bmi_category_distribution(df, title="Demographic Distribution across Standard BMI Categories"):
    """Categorizes BMI into Underweight, Normal, Overweight, Obese and plots distribution bar."""
    if df is None or len(df) == 0 or "bmi" not in df.columns:
        return apply_plotly_theme(go.Figure())

    def categorize_bmi(bmi):
        if pd.isna(bmi):
            return "Unknown"
        if bmi < 18.5:
            return "Underweight (<18.5)"
        elif 18.5 <= bmi < 25.0:
            return "Normal (18.5 - 24.9)"
        elif 25.0 <= bmi < 30.0:
            return "Overweight (25.0 - 29.9)"
        else:
            return "Obese (≥ 30.0)"

    bmi_cats = df["bmi"].apply(categorize_bmi)
    cat_counts = bmi_cats.value_counts().reset_index()
    cat_counts.columns = ["BMI Category", "Count"]
    
    order_map = {"Underweight (<18.5)": 1, "Normal (18.5 - 24.9)": 2, "Overweight (25.0 - 29.9)": 3, "Obese (≥ 30.0)": 4}
    cat_counts["Order"] = cat_counts["BMI Category"].map(order_map)
    cat_counts = cat_counts.sort_values(by="Order")
    
    total = len(df)
    cat_counts["Percentage"] = (cat_counts["Count"] / total * 100).round(1)
    cat_counts["Label"] = cat_counts.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1)

    fig = px.bar(
        cat_counts,
        x="BMI Category",
        y="Count",
        text="Label",
        color="BMI Category",
        color_discrete_sequence=["#95BAFE", "#CCFED8", "#489CE2", "#1B75BB"],
        title=title
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="Participant Count", xaxis_title="BMI Classification")
    return apply_plotly_theme(fig, height=450)

def plot_scatter(df, x_col, y_col, color_col=None):
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    sample_df = df.sample(min(len(df), 4000), random_state=42)
    x_name = str(x_col).replace('_', ' ').title()
    y_name = str(y_col).replace('_', ' ').title()
    
    fig = px.scatter(
        sample_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in sample_df.columns else None,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.65,
        title=f"{y_name} vs {x_name} (Sampled Profile Cluster)"
    )
    return apply_plotly_theme(fig, height=460)

def plot_donut_chart(df, names_col, values_col, title="Distribution Mix"):
    """Renders a modern donut chart for composition analysis."""
    if df is None or names_col not in df.columns:
        return apply_plotly_theme(go.Figure())

    grouped = df.groupby(names_col)[values_col].sum().reset_index() if values_col in df.columns else df[names_col].value_counts().reset_index()
    grouped.columns = [names_col, values_col]

    fig = px.pie(
        grouped,
        names=names_col,
        values=values_col,
        hole=0.5,
        color_discrete_sequence=COLOR_SEQUENCE,
        title=title
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color=BG_CARD, width=2)))
    return apply_plotly_theme(fig, height=450)

def plot_treemap(df, path_cols, values_col, title="Treemap Breakdown"):
    """Renders an executive treemap visualization."""
    if df is None or not any(c in df.columns for c in path_cols):
        return apply_plotly_theme(go.Figure())

    fig = px.treemap(
        df.sample(min(len(df), 10000), random_state=42),
        path=path_cols,
        values=values_col if values_col in df.columns else None,
        color=values_col if values_col in df.columns else None,
        color_continuous_scale=["#0F1C30", "#1B75BB", "#95BAFE", "#CCFED8"],
        title=title
    )
    return apply_plotly_theme(fig, height=480)

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    if y_true is None or y_pred is None or len(y_true) == 0:
        return apply_plotly_theme(go.Figure())

    sample_size = min(len(y_true), 2000)
    np.random.seed(42)
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    
    y_true_s = np.array(y_true)[indices]
    y_pred_s = np.array(y_pred)[indices]
    
    min_val = min(y_true_s.min(), y_pred_s.min())
    max_val = max(y_true_s.max(), y_pred_s.max())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true_s,
        y=y_pred_s,
        mode='markers',
        marker=dict(color=PRIMARY_LIGHT, opacity=0.6, size=6),
        name='Test Predictions'
    ))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color=ACCENT, dash='dash', width=2),
        name='Perfect Prediction Line (y = x)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Values (kcal)",
        yaxis_title="Model Predicted Values (kcal)",
    )
    return apply_plotly_theme(fig, height=480)

def plot_residuals(y_true, y_pred, title="Residual Error Distribution"):
    if y_true is None or y_pred is None or len(y_true) == 0:
        return apply_plotly_theme(go.Figure())

    sample_size = min(len(y_true), 2000)
    np.random.seed(42)
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    
    y_pred_s = np.array(y_pred)[indices]
    residuals = np.array(y_true)[indices] - y_pred_s
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred_s,
        y=residuals,
        mode='markers',
        marker=dict(color=SECONDARY, opacity=0.6, size=6),
        name='Residuals'
    ))
    
    fig.add_trace(go.Scatter(
        x=[y_pred_s.min(), y_pred_s.max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='#EF4444', dash='dash', width=2),
        name='Zero Line'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residual Error (Actual - Predicted)",
    )
    return apply_plotly_theme(fig, height=480)

def plot_feature_importance(importance_series, title="Top Predictive Features"):
    if importance_series is None or len(importance_series) == 0:
        return apply_plotly_theme(go.Figure())

    df_imp = importance_series.reset_index()
    df_imp.columns = ["Feature", "Importance"]
    df_imp["Feature"] = df_imp["Feature"].apply(lambda x: str(x).replace("_", " ").title())
    df_imp = df_imp.sort_values(by="Importance", ascending=True).tail(12)
    
    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
        color="Importance",
        color_continuous_scale=["#1B75BB", "#95BAFE", "#CCFED8"],
        title=title
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Relative Feature Importance Score", yaxis_title="Feature Name")
    return apply_plotly_theme(fig, height=450)

def plot_gauge_meter(value, min_val=0, max_val=1000, title="Predicted Output", unit="kcal"):
    """Renders a high-end gauge meter for individual prediction outputs."""
    safe_val = float(value) if isinstance(value, (int, float, np.number)) else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=safe_val,
        number={'suffix': f" {unit}", 'font': {'color': TEXT_MAIN, 'size': 36}},
        title={'text': title, 'font': {'color': SECONDARY, 'size': 18}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': BORDER_COLOR},
            'bar': {'color': PRIMARY_LIGHT},
            'bgcolor': BG_SURFACE,
            'borderwidth': 1,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [min_val, (max_val - min_val) * 0.35], 'color': 'rgba(27, 117, 187, 0.2)'},
                {'range': [(max_val - min_val) * 0.35, (max_val - min_val) * 0.7], 'color': 'rgba(149, 186, 254, 0.25)'},
                {'range': [(max_val - min_val) * 0.7, max_val], 'color': 'rgba(204, 254, 216, 0.25)'}
            ],
            'threshold': {
                'line': {'color': ACCENT, 'width': 4},
                'thickness': 0.75,
                'value': safe_val
            }
        }
    ))
    return apply_plotly_theme(fig, height=320)
