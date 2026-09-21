import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

# 1. Page Configuration
st.set_page_config(
    page_title="Predictive Analytics Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Universal Dark/Light Adaptive CSS Engine
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    /* Global Typography */
    html, body, .stApp, h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, div[data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', Arial, sans-serif !important;
    }

    /* Core Theme-Driven App Canvas */
    .stApp {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }

    /* Hide Sidebar Completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 22px;
        border-left: 6px solid #F59E0B;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }

    .header-title {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .header-accent {
        color: #F59E0B !important;
    }

    .header-subtitle {
        color: #94A3B8 !important;
        font-size: 0.92rem;
        margin-top: 4px;
        font-weight: 500 !important;
    }

    /* Control Panel & Container Wrappers */
    div[data-testid="stVerticalBlock"] > div:has(div.top-nav-marker) {
        background-color: var(--secondary-background-color) !important;
        padding: 22px 26px !important;
        border-radius: 14px !important;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2)) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 25px !important;
    }

    /* Form Labels & Widget Descriptions */
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stWidgetLabel"] p {
        color: var(--text-color) !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }

    /* Adaptive File Uploader */
    div[data-testid="stFileUploader"] {
        background-color: var(--background-color) !important;
        border: 2px dashed #2563EB !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }

    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] label p {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 6px 18px !important;
    }

    /* Selectboxes and Inputs */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        background-color: var(--background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        border-radius: 8px !important;
        color: var(--text-color) !important;
    }

    div[data-baseweb="select"] span {
        color: var(--text-color) !important;
    }

    /* Metric Cards & Charts */
    div[data-testid="stMetric"], 
    div[data-testid="stPlotlyChart"] {
        background-color: var(--secondary-background-color) !important;
        padding: 22px 26px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.04) !important;
    }

    /* Primary Action Buttons */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 8px 20px !important;
        transition: transform 0.2s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
    }

    /* Tab Headers */
    button[data-baseweb="tab"] p {
        color: var(--text-color) !important;
        font-weight: 700 !important;
    }

    button[aria-selected="true"] p {
        color: #D97706 !important;
    }

    .box-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header Banner
st.markdown("""
<div class="header-banner">
    <div class="header-title">Predictive Analytics <span class="header-accent">& Forecasting Studio</span></div>
    <div class="header-subtitle">Enterprise Trend Analysis & Automated Time-Series Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# 4. Data Generator
@st.cache_data
def generate_default_data():
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    n = len(dates)
    trend = np.linspace(100, 600, n)
    seasonality = 80 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    weekend_bump = 35 * (dates.dayofweek >= 5).astype(int)
    noise = np.random.normal(0, 15, n)
    values = trend + seasonality + weekend_bump + noise
    return pd.DataFrame({"Date": dates, "Target_Value": values})

# 5. File Validation Logic Engine
def validate_uploaded_csv(uploaded_file):
    """Performs structural and data quality checks on uploaded time-series CSVs."""
    validation_results = {
        "is_valid": True,
        "messages": [],
        "df": None,
        "date_cols": [],
        "numeric_cols": []
    }
    
    try:
        uploaded_file.seek(0)
        test_df = pd.read_csv(uploaded_file)
        validation_results["df"] = test_df
    except Exception as e:
        validation_results["is_valid"] = False
        validation_results["messages"].append(f"❌ **File Parsing Error**: Failed to read file as a CSV. Details: `{str(e)}`")
        return validation_results

    # Check minimum row count
    if len(test_df) < 10:
        validation_results["is_valid"] = False
        validation_results["messages"].append(f"❌ **Insufficient Rows**: Dataset contains only {len(test_df)} rows. Minimum 10 rows required for time-series modeling.")

    # Identify potential Date columns
    possible_date_cols = []
    for col in test_df.columns:
        if test_df[col].dtype == 'object' or 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(test_df[col].dropna().head(10))
                possible_date_cols.append(col)
            except Exception:
                pass
        elif pd.api.types.is_datetime64_any_dtype(test_df[col]):
            possible_date_cols.append(col)

    validation_results["date_cols"] = possible_date_cols

    if not possible_date_cols:
        validation_results["is_valid"] = False
        validation_results["messages"].append("❌ **Missing Date Column**: No parseable date/time column detected in the file.")
    else:
        validation_results["messages"].append(f"✅ **Date Column Detected**: Found candidate date column(s): `{', '.join(possible_date_cols)}`.")

    # Identify numeric columns
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    validation_results["numeric_cols"] = numeric_cols

    if not numeric_cols:
        validation_results["is_valid"] = False
        validation_results["messages"].append("❌ **Missing Target Metrics**: No numeric value columns detected for forecasting.")
    else:
        validation_results["messages"].append(f"✅ **Numeric Metrics Detected**: Found candidate metric column(s): `{', '.join(numeric_cols)}`.")

    # Check missing value percentage
    null_pct = test_df.isnull().mean().max() * 100
    if null_pct > 30:
        validation_results["messages"].append(f"⚠️ **High Null Ratio Warning**: Some columns have up to {null_pct:.1f}% missing values.")
    else:
        validation_results["messages"].append(f"✅ **Data Completeness**: Missing values are well within operational bounds ({null_pct:.1f}% max missing).")

    if validation_results["is_valid"]:
        validation_results["messages"].insert(0, "🎉 **Validation Passed**: File structure is valid for predictive modeling!")
    
    return validation_results

# 6. Navigation & Control Panel
st.markdown('<div class="top-nav-marker"></div>', unsafe_allow_html=True)
st.markdown('### 🛠️ Navigation & Control Panel')

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2.8, 2, 2, 2.2])

with nav_col1:
    uploaded_file = st.file_uploader("Upload Time-Series CSV", type=["csv"])
    if uploaded_file is not None:
        if st.button("🔍 Validate Dataset Format", use_container_width=True):
            val_res = validate_uploaded_csv(uploaded_file)
            st.session_state['val_results'] = val_res

if uploaded_file is not None:
    # Display validation feedback if performed
    if 'val_results' in st.session_state and st.session_state['val_results'] is not None:
        res = st.session_state['val_results']
        with st.expander("📌 Dataset Format Validation Report", expanded=True):
            for msg in res["messages"]:
                if "❌" in msg:
                    st.error(msg)
                elif "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.success(msg)

    raw_df = pd.read_csv(uploaded_file)
    with nav_col2:
        date_col = st.selectbox("Date Column:", raw_df.columns)
    with nav_col3:
        target_col = st.selectbox("Target Column:", [c for c in raw_df.columns if c != date_col])
    with nav_col4:
        forecast_horizon = st.slider("Forecast Days:", min_value=7, max_value=180, value=30)
    
    df = raw_df[[date_col, target_col]].copy()
    df.columns = ["Date", "Target_Value"]
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.sort_values("Date").dropna()
    n_estimators = 100
else:
    df = generate_default_data()
    with nav_col2:
        forecast_horizon = st.slider("Forecast Horizon (Days):", min_value=7, max_value=180, value=30)
    with nav_col3:
        n_estimators = st.slider("Model Trees:", min_value=50, max_value=300, value=100, step=50)
    with nav_col4:
        st.info("ℹ️ Running on synthetic 3-year baseline data.")

st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px; border-color: rgba(128, 128, 128, 0.2);'>", unsafe_allow_html=True)

# 7. Machine Learning Pipeline
def create_features(data):
    df_feat = data.copy()
    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Quarter'] = df_feat['Date'].dt.quarter
    df_feat['Year'] = df_feat['Date'].dt.year
    df_feat['DayOfYear'] = df_feat['Date'].dt.dayofyear
    
    df_feat['Lag_1'] = df_feat['Target_Value'].shift(1)
    df_feat['Lag_7'] = df_feat['Target_Value'].shift(7)
    df_feat['Lag_30'] = df_feat['Target_Value'].shift(30)
    df_feat['Rolling_Mean_7'] = df_feat['Target_Value'].shift(1).rolling(window=7).mean()
    df_feat['Rolling_Mean_30'] = df_feat['Target_Value'].shift(1).rolling(window=30).mean()
    return df_feat

df_engineered = create_features(df).dropna().reset_index(drop=True)
feature_cols = ['DayOfWeek', 'Month', 'Quarter', 'Year', 'DayOfYear', 
                'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_Mean_7', 'Rolling_Mean_30']

split_idx = int(len(df_engineered) * 0.80)
train_df = df_engineered.iloc[:split_idx]
test_df = df_engineered.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df['Target_Value']
X_test, y_test = test_df[feature_cols], test_df['Target_Value']

model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Recursive Forecasting
future_df = df.copy()
last_date = future_df['Date'].max()

for i in range(forecast_horizon):
    next_date = last_date + timedelta(days=i + 1)
    temp_feat = create_features(future_df)
    
    latest_row = temp_feat.iloc[-1].copy()
    latest_row['Date'] = next_date
    latest_row['DayOfWeek'] = next_date.dayofweek
    latest_row['Month'] = next_date.month
    latest_row['Quarter'] = next_date.quarter
    latest_row['Year'] = next_date.year
    latest_row['DayOfYear'] = next_date.dayofyear
    
    X_future = pd.DataFrame([latest_row[feature_cols]])
    pred_val = model.predict(X_future)[0]
    
    future_df = pd.concat([future_df, pd.DataFrame({"Date": [next_date], "Target_Value": [pred_val]})], ignore_index=True)

historical_plot = df.copy()
future_plot = future_df.iloc[-forecast_horizon:].copy()

# 8. Main Dashboard Output
tab1, tab2, tab3 = st.tabs(["📈 Demand Forecast Studio", "📊 Feature Diagnostics", "📦 Export Forecast Table"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE Score", f"{mae:.2f}")
    c2.metric("RMSE Deviation", f"{rmse:.2f}")
    c3.metric("R² Model Score", f"{r2:.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_plot['Date'], 
        y=historical_plot['Target_Value'],
        mode='lines',
        name='Historical Actuals',
        line=dict(color='#2563EB', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=future_plot['Date'], 
        y=future_plot['Target_Value'],
        mode='lines',
        name=f'{forecast_horizon}-Day Forecast',
        line=dict(color='#D97706', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title=dict(text="Historical Trends vs Projected Forecast Trajectory", font=dict(size=16, family="Plus Jakarta Sans", weight="bold")),
        xaxis=dict(title="Timeline", gridcolor="rgba(128, 128, 128, 0.2)"),
        yaxis=dict(title="Target Metric Value", gridcolor="rgba(128, 128, 128, 0.2)"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="box-header">Feature Importance Distribution</div>', unsafe_allow_html=True)
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale=['#2563EB', '#D97706']
    )
    
    fig_imp.update_layout(
        title=dict(text="Feature Impact Ranking", font=dict(size=16, family="Plus Jakarta Sans", weight="bold")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)"),
        yaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.markdown(f'<div class="box-header">Generated {forecast_horizon}-Day Projections Table</div>', unsafe_allow_html=True)
    st.dataframe(
        future_plot.rename(columns={"Target_Value": "Predicted_Value"})
        .style.format({"Predicted_Value": "{:.2f}"}),
        use_container_width=True
    )
    
    csv_data = future_plot.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast CSV Data",
        data=csv_data,
        file_name=f"predictive_forecast_{forecast_horizon}_days.csv",
        mime="text/csv"
    )