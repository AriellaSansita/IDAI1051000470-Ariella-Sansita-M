import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

# ===============================
# 2. DATA LOADING (LOCAL PATH)
# ===============================
# This follows your requirement to load directly from the repo folder
try:
    df_raw = pd.read_csv("cleaned_ev_charging_data.csv")
    st.success("✅ Dataset 'cleaned_ev_charging_data.csv' loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure 'cleaned_ev_charging_data.csv' is in the same folder as this script.")
    st.stop()

# ===============================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ===============================
@st.cache_data
def preprocess_data(df):
    df_proc = df.copy()
    
    # Fill missing values (Median for ratings, Mode for categories)
    if 'Reviews (Rating)' in df_proc.columns:
        df_proc['Reviews (Rating)'] = df_proc['Reviews (Rating)'].fillna(df_proc['Reviews (Rating)'].median())
    
    # Encoding for Clustering logic
    le = LabelEncoder()
    for col in ['Charger Type', 'Station Operator', 'Renewable Energy Source']:
        if col in df_proc.columns:
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

    # Normalize Numeric Features (Required for K-Means distance calculation)
    scaler = MinMaxScaler()
    features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)']
    existing_features = [f for f in features if f in df_proc.columns]
    
    if existing_features:
        df_proc[existing_features] = scaler.fit_transform(df_proc[existing_features])
    
    return df_proc, existing_features

df_processed, cluster_cols = preprocess_data(df_raw)

# ===============================
# 4. STAGE 4: K-MEANS CLUSTERING & ELBOW METHOD
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")
st.write("Using the Elbow Method to determine the optimal number of clusters based on WCSS (Within-Cluster Sum of Squares).")

c1, c2 = st.columns([1, 1])

with c1:
    # Elbow Method Plot
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        km.fit(df_processed[cluster_cols])
        wcss.append(km.inertia_)
    
    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
    ax_elbow.plot(range(1, 11), wcss, marker='o', color='#1f77b4')
    ax_elbow.set_title('Elbow Method')
    ax_elbow.set_xlabel('Number of Clusters')
    ax_elbow.set_ylabel('WCSS')
    st.pyplot(fig_elbow)

with c2:
    # Cluster Execution
    k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)
    model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
    df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])
    
    # Visualizing Clusters
    fig_cluster, ax_cluster = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                    hue='Cluster', palette='Set1', ax=ax_cluster)
    ax_cluster.set_title(f"K-Means Clustering (k={k_value})")
    st.pyplot(fig_cluster)

import pydeck as pdk

# ===============================
# 5. STAGE 8: GEOSPATIAL ANALYSIS (Attribution Removed)
# ===============================
st.header("📍 Stage 8: Geographic Distribution")
if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    # We use pydeck directly to set a style that minimizes UI elements
    st.pydeck_chart(pdk.Deck(
        map_style=None,  # Setting this to None often removes standard overlays
        initial_view_state=pdk.ViewState(
            latitude=df_raw['Latitude'].mean(),
            longitude=df_raw['Longitude'].mean(),
            zoom=1,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_raw,
                get_position='[Longitude, Latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200000,
            ),
        ],
    ))
else:
    st.warning("Map skipped: 'Latitude' and 'Longitude' columns not found in dataset.")

# ===============================
# 6. CORRELATION & INSIGHTS
# ===============================
st.divider()
st.header("📊 Stage 7: Interpretation & Insights")

# Heatmap Fix (Smaller font, no overlap)
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(df_processed.select_dtypes(include=['number']).corr(), 
            annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8}, ax=ax_corr)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_corr)

# Automatic Insight Generation
st.subheader("Key Findings")
avg_usage = df_raw.groupby('Cluster')['Usage Stats (avg users/day)'].mean()
best_cluster = avg_usage.idxmax()

st.info(f"""
- **Top Performing Group:** Cluster {best_cluster} shows the highest average daily usage.
- **Correlations:** The heatmap indicates how factors like 'Cost' and 'Capacity' influence user behavior.
- **Geographic Trend:** The map reveals concentration of charging infrastructure in specific zones.
""")

if st.checkbox("View Final Data Table"):
    st.dataframe(df_raw)
