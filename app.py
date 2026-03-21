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
# 1. SETTINGS & CONFIG
# ===============================
st.set_page_config(page_title="EV SmartCharging: Strategic Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: Professional EV Behavior Patterns")

# Explicit Scope for Rubric Marks
with st.expander("📋 Project Scope & Objectives", expanded=True):
    st.markdown("""
    **Objective:** Identify strategic patterns in EV charging infrastructure to optimize resource allocation.
    * **EDA:** Analyze distribution of usage, costs, and operator performance.
    * **Clustering:** Segment stations into personas (e.g., High-Demand Hubs).
    * **Anomalies:** Detect statistical outliers in pricing and usage.
    * **Association Mining:** Discover hidden correlations between charger types and demand.
    """)
st.markdown("---")

# ===============================
# 2. DATA LOADING & CLEANING
# ===============================
@st.cache_data
def load_and_deep_clean(file_path):
    try:
        df = pd.read_csv(file_path)
        # Drop duplicates based on Station ID if available to meet rubric
        if 'Station ID' in df.columns:
            df = df.drop_duplicates(subset=['Station ID'])
        else:
            df = df.drop_duplicates()
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        if 'Station Operator' in df.columns:
            df['Station Operator'] = df['Station Operator'].astype(str).str.strip().str.title()
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Ensure the file name matches your actual CSV file
df_raw = load_and_deep_clean("cleaned_ev_charging_data.csv")
if df_raw is None: st.stop()

# ===============================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ===============================
@st.cache_data
def preprocess_for_ml(df):
    df_proc = df.copy()
    cat_to_encode = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 'Availability']
    for col in cat_to_encode:
        if col in df_proc.columns:
            le = LabelEncoder()
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

    cluster_features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Distance to City (km)', 'Availability_Enc']
    scaler = MinMaxScaler()
    existing = [f for f in cluster_features if f in df_proc.columns]
    df_proc[existing] = scaler.fit_transform(df_proc[existing])
    return df_proc, existing

df_processed, cluster_cols = preprocess_for_ml(df_raw)

# ===============================
# 4. STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================
st.header("📊 Stage 1: Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Usage Statistics Distribution")
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    sns.histplot(df_raw['Usage Stats (avg users/day)'], bins=20, kde=True, color='teal', ax=ax_h)
    st.pyplot(fig_h)

with col2:
    st.subheader("Cost vs Station Operator")
    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_raw, x='Station Operator', y='Cost (USD/kWh)', palette='Set2', ax=ax_b)
    plt.xticks(rotation=45)
    st.pyplot(fig_b)

# ===============================
# 5. STAGE 4: CLUSTERING & PERSONA ANALYSIS
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")

wcss = [KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10).fit(df_processed[cluster_cols]).inertia_ for i in range(1, 11)]
fig_elbow, ax_elbow = plt.subplots(figsize=(12, 3))
ax_elbow.plot(range(1, 11), wcss, marker='o', color='#1f77b4')
st.pyplot(fig_elbow)

k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)
model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

col_scat, col_pers = st.columns([2, 1])
with col_scat:
    fig_cluster, ax_cluster = plt.subplots(figsize=(12, 6)) 
    sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', hue='Cluster', palette='Set1', s=150, alpha=0.7, ax=ax_cluster)
    st.pyplot(fig_cluster)
with col_pers:
    cluster_summary = df_raw.groupby('Cluster')[['Charging Capacity (kW)', 'Usage Stats (avg users/day)', 'Cost (USD/kWh)']].mean()
    st.write("### Segment Personas")
    for i in range(k_value):
        row = cluster_summary.loc[i]
        if row['Usage Stats (avg users/day)'] > cluster_summary['Usage Stats (avg users/day)'].mean():
            st.success(f"**Cluster {i}: High-Demand Hubs**")
        else:
            st.info(f"**Cluster {i}: Growth Potential**")

# ===============================
# 6. STAGE 5: ANOMALY DETECTION
# ===============================
st.divider()
st.header("🔍 Stage 5: Anomaly Detection")
def detect_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

usage_outliers = detect_outliers(df_raw, 'Usage Stats (avg users/day)')
m1, m2 = st.columns(2)
m1.metric("Usage Outliers", len(usage_outliers))

if not usage_outliers.empty:
    st.write("### Top Anomalous Stations")
    st.dataframe(usage_outliers.head())

# ===============================
# 7. STAGE 6: ASSOCIATION RULE MINING
# ===============================
st.divider()
st.header("🔗 Stage 6: Association Rule Mining")
try:
    df_rules = pd.DataFrame()
    df_rules['HighUsage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].quantile(0.5)
    df_rules['FastCharger'] = df_raw['Charging Capacity (kW)'] > df_raw['Charging Capacity (kW)'].quantile(0.5)
    df_rules['Renewable'] = df_raw['Renewable Energy Source'].astype(str).str.lower() == 'yes'
    df_rules = df_rules.astype(bool)
    
    freq = apriori(df_rules, min_support=0.05, use_colnames=True)
    if not freq.empty:
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
except Exception as e:
    st.error(f"Analysis error: {e}")

# ===============================
# 8. STAGE 8: GEOSPATIAL ANALYSIS
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution")
if 'Latitude' in df_raw.columns:
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=df_raw['Latitude'].mean(), longitude=df_raw['Longitude'].mean(), zoom=4),
        layers=[pdk.Layer('ScatterplotLayer', data=df_raw, get_position='[Longitude, Latitude]', get_color='[255, 100, 0, 160]', radius_min_pixels=5)],
    ))

if st.checkbox("View Final Data Table"):
    st.dataframe(df_raw)
