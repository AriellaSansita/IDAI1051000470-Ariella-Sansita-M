import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

# ===============================
# 2. DATA LOADING (LOCAL PATH)
# ===============================
try:
    df_raw = pd.read_csv("cleaned_ev_charging_data.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure 'cleaned_ev_charging_data.csv' is in the same folder as this script.")
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
    
    # Encoding categorical columns
    le = LabelEncoder()
    for col in ['Charger Type', 'Station Operator', 'Renewable Energy Source']:
        if col in df_proc.columns:
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

    # Normalize Numeric Features for K-Means math
    scaler = MinMaxScaler()
    features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)']
    existing_features = [f for f in features if f in df_proc.columns]
    
    if existing_features:
        df_proc[existing_features] = scaler.fit_transform(df_proc[existing_features])
    
    return df_proc, existing_features

df_processed, cluster_cols = preprocess_data(df_raw)

# ===============================
# 4. STAGE 4: K-MEANS CLUSTERING (ONE BELOW THE OTHER)
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")

# --- Step 1: Elbow Method (Top) ---
st.subheader("1. Finding Optimal Clusters (Elbow Method)")
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    km.fit(df_processed[cluster_cols])
    wcss.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(12, 4)) 
ax_elbow.plot(range(1, 11), wcss, marker='o', color='#1f77b4', linewidth=2)
ax_elbow.set_title('Elbow Method showing the optimal k')
ax_elbow.set_xlabel('Number of Clusters')
ax_elbow.set_ylabel('WCSS (Error Rate)')
st.pyplot(fig_elbow)

# --- Step 2: Scatter Plot (Bottom) ---
st.subheader("2. Market Segmentation Results")
k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)

model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

fig_cluster, ax_cluster = plt.subplots(figsize=(12, 6)) 
# Using df_raw here ensures the axes show real numbers, not 0-1 decimals
sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                hue='Cluster', palette='Set1', s=150, alpha=0.7, ax=ax_cluster)
ax_cluster.set_title(f"Stations Grouped into {k_value} Clusters")
st.pyplot(fig_cluster)

# ===============================
# 5. STAGE 8: GEOSPATIAL ANALYSIS (Fixed Zoom & Scaling)
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution")
if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    st.pydeck_chart(pdk.Deck(
        map_style=None, 
        initial_view_state=pdk.ViewState(
            latitude=df_raw['Latitude'].mean(),
            longitude=df_raw['Longitude'].mean(),
            zoom=2,
            min_zoom=2, 
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_raw,
                get_position='[Longitude, Latitude]',
                get_color='[255, 100, 0, 160]', 
                radius_min_pixels=3, 
                radius_max_pixels=10,
            ),
        ],
    ))
else:
    st.warning("Map skipped: 'Latitude' and 'Longitude' columns not found.")

# ===============================
# 6. CORRELATION & INSIGHTS
# ===============================
st.divider()
st.header("📊 Stage 7: Interpretation & Insights")

fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
# Filter for numeric only to avoid errors
corr_data = df_processed.select_dtypes(include=['number'])
sns.heatmap(corr_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8}, ax=ax_corr)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_corr)

# Automated Insight Generation
st.subheader("Key Findings")
avg_usage = df_raw.groupby('Cluster')['Usage Stats (avg users/day)'].mean()
best_cluster = avg_usage.idxmax()

st.info(f"""
- **Top Performing Group:** Cluster {best_cluster} shows the highest average daily usage.
- **Correlations:** High correlation between Charging Capacity and Usage confirms demand for fast-chargers.
- **Strategic Recommendation:** Focus maintenance and renewable upgrades on stations in the highest-usage cluster.
""")

if st.checkbox("View Final Data Table"):
    st.dataframe(df_raw)
