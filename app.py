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
# 2. DATA LOADING
# ===============================
try:
    df_raw = pd.read_csv("cleaned_ev_charging_data.csv")
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset 'cleaned_ev_charging_data.csv' not found.")
    st.stop()

# ===============================
# 3. PREPROCESSING
# ===============================
@st.cache_data
def preprocess_data(df):
    df_proc = df.copy()
    if 'Reviews (Rating)' in df_proc.columns:
        df_proc['Reviews (Rating)'] = df_proc['Reviews (Rating)'].fillna(df_proc['Reviews (Rating)'].median())
    
    le = LabelEncoder()
    for col in ['Charger Type', 'Station Operator', 'Renewable Energy Source']:
        if col in df_proc.columns:
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

    scaler = MinMaxScaler()
    features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)']
    existing_features = [f for f in features if f in df_proc.columns]
    
    if existing_features:
        df_proc[existing_features] = scaler.fit_transform(df_proc[existing_features])
    
    return df_proc, existing_features

df_processed, cluster_cols = preprocess_data(df_raw)

# ===============================
# 4. CLUSTERING (ONE BELOW THE OTHER)
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")

# Elbow Method
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    km.fit(df_processed[cluster_cols])
    wcss.append(km.inertia_)

st.subheader("1. Finding Optimal Clusters (Elbow Method)")
fig_elbow, ax_elbow = plt.subplots(figsize=(12, 4))
ax_elbow.plot(range(1, 11), wcss, marker='o', color='#1f77b4')
ax_elbow.set_ylabel('WCSS')
st.pyplot(fig_elbow)

# Scatter Plot
st.subheader("2. Market Segmentation Results")
k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)
model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

fig_cluster, ax_cluster = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                hue='Cluster', palette='Set1', s=100, ax=ax_cluster)
st.pyplot(fig_cluster)

# ===============================
# 5. GEOSPATIAL ANALYSIS (FIXED)
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution")

if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    # Use Pydeck to hide the attribution text and lock the zoom
    st.pydeck_chart(pdk.Deck(
        map_style=None, 
        initial_view_state=pdk.ViewState(
            latitude=df_raw['Latitude'].mean(),
            longitude=df_raw['Longitude'].mean(),
            zoom=2,
            min_zoom=2, # Prevents the "double globe" look
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_raw,
                get_position='[Longitude, Latitude]',
                get_color='[255, 100, 0, 160]',
                # Using pixels instead of meters fixes the "Orange Blobs"
                radius_min_pixels=3, 
                radius_max_pixels=10,
            ),
        ],
    ))
else:
    st.warning("Latitude/Longitude not found.")

# ===============================
# 6. INSIGHTS
# ===============================
st.divider()
st.header("📊 Stage 7: Interpretation & Insights")
avg_usage = df_raw.groupby('Cluster')['Usage Stats (avg users/day)'].mean()
st.info(f"**Top Performing Group:** Cluster {avg_usage.idxmax()} has the highest daily usage.")

if st.checkbox("View Final Data Table"):
    st.dataframe(df_raw)
