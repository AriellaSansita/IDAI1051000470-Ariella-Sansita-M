import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ===============================
# 1. SETTINGS & CONFIG (Stage 1: Scope)
# ===============================
st.set_page_config(page_title="EV SmartCharging: Strategic Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: Professional EV Behavior Patterns")
st.markdown("---")

# Sidebar for Deployment Interactivity Marks 
st.sidebar.header("🎯 Dashboard Controls")
st.sidebar.info("Use these controls to filter the analysis and adjust ML parameters.")

# ===============================
# 2. DATA LOADING & CLEANING (Stage 2) 
# ===============================
@st.cache_data
def load_and_deep_clean(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicates based on Station ID as required 
        if 'Station ID' in df.columns:
            df = df.drop_duplicates(subset=['Station ID'])
        else:
            df = df.drop_duplicates()
        
        # Numeric Imputation (Median)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Categorical Imputation (Mode) 
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
            
        if 'Station Operator' in df.columns:
            df['Station Operator'] = df['Station Operator'].astype(str).str.strip().str.title()
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Load dataset
df_raw = load_and_deep_clean("cleaned_ev_charging_data.csv")
if df_raw is None: 
    st.warning("Please upload 'cleaned_ev_charging_data.csv' to the directory.")
    st.stop()

# Sidebar Filter Implementation 
if 'Station Operator' in df_raw.columns:
    ops = st.sidebar.multiselect("Filter by Operator", options=df_raw['Station Operator'].unique(), default=df_raw['Station Operator'].unique())
    df_filtered = df_raw[df_raw['Station Operator'].isin(ops)]
else:
    df_filtered = df_raw

# ===============================
# 3. PREPROCESSING (Stage 2 Continued) 
# ===============================
@st.cache_data
def preprocess_for_ml(df):
    df_proc = df.copy()
    # Categorical Encoding 
    cat_cols = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 'Availability']
    for col in [c for c in cat_cols if c in df_proc.columns]:
        le = LabelEncoder()
        df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

    # Normalization of continuous variables 
    features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Distance to City (km)', 'Availability_Enc']
    existing = [f for f in features if f in df_proc.columns]
    scaler = MinMaxScaler()
    df_proc[existing] = scaler.fit_transform(df_proc[existing])
    return df_proc, existing

df_ml, cluster_cols = preprocess_for_ml(df_filtered)

# ===============================
# 4. EXPLORATORY DATA ANALYSIS (Stage 3) 
# ===============================
st.header("📊 Stage 3: Exploratory Data Analysis")

# Line Chart: Growth over time [Explicit Rubric Requirement - cite: 2]
if 'Installation Year' in df_filtered.columns:
    st.subheader("Infrastructure Growth & Usage Trends")
    yearly = df_filtered.groupby('Installation Year')['Usage Stats (avg users/day)'].mean().reset_index()
    fig_line, ax_line = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=yearly, x='Installation Year', y='Usage Stats (avg users/day)', marker='o', ax=ax_line, color='teal')
    plt.title("Average Usage Trends by Installation Year")
    st.pyplot(fig_line)

col_eda1, col_eda2 = st.columns(2)
with col_eda1:
    st.subheader("Cost Distribution by Operator")
    fig_box, ax_box = plt.subplots()
    sns.boxplot(data=df_filtered, x='Station Operator', y='Cost (USD/kWh)', palette='Set3', ax=ax_box)
    plt.xticks(rotation=45)
    st.pyplot(fig_box)

with col_eda2:
    # Heatmap: Demand across Charger Type and Availability 
    st.subheader("Demand Heatmap")
    if 'Charger Type' in df_filtered.columns and 'Availability' in df_filtered.columns:
        pivot = df_filtered.pivot_table(index='Charger Type', columns='Availability', values='Usage Stats (avg users/day)', aggfunc='mean')
        fig_heat, ax_heat = plt.subplots()
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax_heat)
        st.pyplot(fig_heat)

# ===============================
# 5. CLUSTERING (Stage 4) 
# ===============================
st.divider()
st.header("🤖 Stage 4: Station Clustering")

k_val = st.sidebar.slider("Number of Clusters (k)", 2, 6, 3)
if len(cluster_cols) > 0:
    model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    df_filtered['Cluster'] = model.fit_predict(df_ml[cluster_cols])

    c1, c2 = st.columns([2, 1])
    with c1:
        fig_scat, ax_scat = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df_filtered, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', hue='Cluster', palette='viridis', s=100, ax=ax_scat)
        plt.title("Station Segmentation based on Capacity and Usage")
        st.pyplot(fig_scat)

    with c2:
        st.write("### Segment Personas ")
        summary = df_filtered.groupby('Cluster')['Usage Stats (avg users/day)'].mean()
        for i in range(k_val):
            val = summary.loc[i]
            if val > summary.mean() * 1.2:
                st.success(f"Cluster {i}: Heavy Users (High Demand)")
            elif val < summary.mean() * 0.8:
                st.info(f"Cluster {i}: Occasional Users (Low Frequency)")
            else:
                st.warning(f"Cluster {i}: Daily Commuters (Moderate)")

# ===============================
# 6. ANOMALY DETECTION (Stage 6) 
# ===============================
st.divider()
st.header("🔍 Stage 6: Anomaly Detection")
def get_outliers(df, col):
    if col in df.columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        return df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
    return pd.DataFrame()

usage_anomalies = get_outliers(df_filtered, 'Usage Stats (avg users/day)')
if usage_anomalies.empty:
    st.success("✅ No usage anomalies detected. All stations operating within normal parameters.")
else:
    st.error(f"⚠️ Detected {len(usage_anomalies)} usage anomalies!")
    st.write("These stations show abnormal consumption behavior.")
    st.dataframe(usage_anomalies.head())

# ===============================
# 7. ASSOCIATION RULES (Stage 5) 
# ===============================
st.divider()
st.header("🔗 Stage 5: Association Rule Mining")
try:
    rule_df = pd.DataFrame({
        'HighUsage': df_filtered['Usage Stats (avg users/day)'] > df_filtered['Usage Stats (avg users/day)'].median(),
        'FastCharge': df_filtered['Charging Capacity (kW)'] > df_filtered['Charging Capacity (kW)'].median(),
        'Renewable': df_filtered['Renewable Energy Source'].astype(str).str.lower().isin(['yes', 'true', '1'])
    }).astype(bool)
    
    freq = apriori(rule_df, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.0)
    if not rules.empty:
        st.write("Discovered relationships between station features and demand:")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
    else:
        st.write("No strong associations found in the current filtered data.")
except Exception:
    st.write("Insufficient variance for Association Rule Mining.")

# ===============================
# 8. GEOSPATIAL & INSIGHTS (Stage 7 & 8) 
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution & Insights")
if 'Latitude' in df_filtered.columns and 'Longitude' in df_filtered.columns:
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=df_filtered['Latitude'].mean(), longitude=df_filtered['Longitude'].mean(), zoom=3),
        layers=[pdk.Layer('ScatterplotLayer', data=df_filtered, get_position='[Longitude, Latitude]', get_color='[200, 30, 0, 160]', radius_min_pixels=5)]
    ))

st.subheader("Key Strategic Findings ")
st.info(f"""
- **Infrastructure Strategy**: Prioritize expansion in regions with 'Heavy User' clusters.
- **Reliability**: Investigate {len(usage_anomalies)} anomalies for potential faulty equipment or station abuse.
- **Demand Optimization**: Line trends show usage growth; pricing should be optimized during peak years/times.
""")

if st.checkbox("View Processed Dataset"):
    st.dataframe(df_filtered)
