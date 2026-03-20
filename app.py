import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

# ===============================
# 2. DATA LOADING
# ===============================
try:
    df_raw = pd.read_csv("cleaned_ev_charging_data.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure 'cleaned_ev_charging_data.csv' is in the same folder.")
    st.stop()

# ===============================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ===============================
@st.cache_data
def preprocess_data(df):
    df_proc = df.copy()
    
    # Fill missing values
    if 'Reviews (Rating)' in df_proc.columns:
        df_proc['Reviews (Rating)'] = df_proc['Reviews (Rating)'].fillna(df_proc['Reviews (Rating)'].median())
    
    # Label Encoding for categorical columns
    le = LabelEncoder()
    cat_cols = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 'Availability']
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

    # Features for Clustering (including new requested ones)
    cluster_features = [
        'Cost (USD/kWh)', 
        'Usage Stats (avg users/day)', 
        'Charging Capacity (kW)',
        'Distance to City (km)', 
        'Availability_Enc'
    ]
    
    # Normalize Numeric Features
    scaler = MinMaxScaler()
    existing_features = [f for f in cluster_features if f in df_proc.columns]
    if existing_features:
        df_proc[existing_features] = scaler.fit_transform(df_proc[existing_features])
    
    return df_proc, existing_features

df_processed, cluster_cols = preprocess_data(df_raw)

# ===============================
# 4. STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================
st.divider()
st.header("📊 Stage 1: Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Usage Statistics Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    sns.histplot(df_raw['Usage Stats (avg users/day)'], bins=20, kde=True, color='teal', ax=ax_hist)
    st.pyplot(fig_hist)

with col2:
    st.subheader("Cost vs Station Operator")
    fig_box, ax_box = plt.subplots(figsize=(8, 5))
    # Using raw operator names for better readability
    sns.boxplot(data=df_raw, x='Station Operator', y='Cost (USD/kWh)', palette='Set2', ax=ax_box)
    plt.xticks(rotation=45)
    st.pyplot(fig_box)

st.subheader("Usage Trend by Installation Year")
if 'Installation Year' in df_raw.columns:
    trend_data = df_raw.groupby('Installation Year')['Usage Stats (avg users/day)'].mean().reset_index()
    fig_line, ax_line = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=trend_data, x='Installation Year', y='Usage Stats (avg users/day)', marker='o', ax=ax_line)
    st.pyplot(fig_line)

# ===============================
# 5. STAGE 4: K-MEANS CLUSTERING (IMPROVED)
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")
st.write(f"**Features Used:** {', '.join(cluster_cols)}")

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    km.fit(df_processed[cluster_cols])
    wcss.append(km.inertia_)

st.subheader("1. Finding Optimal Clusters (Elbow Method)")
fig_elbow, ax_elbow = plt.subplots(figsize=(12, 4)) 
ax_elbow.plot(range(1, 11), wcss, marker='o', color='#1f77b4', linewidth=2)
st.pyplot(fig_elbow)

st.subheader("2. Market Segmentation Results")
k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)
model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

fig_cluster, ax_cluster = plt.subplots(figsize=(12, 6)) 
sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                hue='Cluster', palette='Set1', s=150, alpha=0.7, ax=ax_cluster)
st.pyplot(fig_cluster)

# ===============================
# 6. STAGE 5: ANOMALY DETECTION (IQR)
# ===============================
st.divider()
st.header("🔍 Stage 5: Anomaly Detection")
st.write("Detecting outliers in Usage and Cost using the Interquartile Range (IQR) method.")

def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

usage_outliers = detect_outliers(df_raw, 'Usage Stats (avg users/day)')
cost_outliers = detect_outliers(df_raw, 'Cost (USD/kWh)')

c1, c2 = st.columns(2)
c1.metric("Usage Outliers Detected", len(usage_outliers))
c2.metric("Cost Outliers Detected", len(cost_outliers))

if st.checkbox("Show Anomaly Data"):
    st.write("Top Usage Anomalies:")
    st.dataframe(usage_outliers.head())

# ===============================
# 7. STAGE 6: ASSOCIATION RULE MINING
# ===============================
st.divider()
st.header("🔗 Stage 6: Association Rule Mining")

# Discretizing data for Apriori (Binary Encoding)
df_rules = pd.DataFrame()
df_rules['High_Usage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].median()
df_rules['Fast_Charger'] = df_raw['Charging Capacity (kW)'] > df_raw['Charging Capacity (kW)'].median()
df_rules['Renewable'] = df_raw['Renewable Energy Source'] == 1
df_rules['High_Cost'] = df_raw['Cost (USD/kWh)'] > df_raw['Cost (USD/kWh)'].median()

frequent_itemsets = apriori(df_rules, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

st.write("Discovered patterns in station behavior:")
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# ===============================
# 8. STAGE 8: GEOSPATIAL ANALYSIS
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution")
if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    st.pydeck_chart(pdk.Deck(
        map_style=None, 
        initial_view_state=pdk.ViewState(
            latitude=df_raw['Latitude'].mean(), longitude=df_raw['Longitude'].mean(),
            zoom=2, min_zoom=2, pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer', data=df_raw, get_position='[Longitude, Latitude]',
                get_color='[255, 100, 0, 160]', radius_min_pixels=3, radius_max_pixels=10,
            ),
        ],
    ))

# ===============================
# 9. INTERPRETATION & INSIGHTS
# ===============================
st.divider()
st.header("📊 Stage 7: Interpretation & Insights")

fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(df_processed.select_dtypes(include=['number']).corr(), 
            annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8}, ax=ax_corr)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_corr)

st.subheader("Key Findings")
avg_usage = df_raw.groupby('Cluster')['Usage Stats (avg users/day)'].mean()
st.info(f"""
- **Top Performing Group:** Cluster {avg_usage.idxmax()} shows the highest average daily usage.
- **Anomalies:** Identified {len(usage_outliers)} stations with irregular usage patterns.
- **Rules:** Association rules suggest link between '{rules.iloc[0]['antecedents']}' and '{rules.iloc[0]['consequents']}'.
""")

if st.checkbox("View Final Data Table"):
    st.dataframe(df_raw)
