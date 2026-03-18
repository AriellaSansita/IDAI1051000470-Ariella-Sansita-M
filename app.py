import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

# --- 1. PAGE SETTINGS ---
st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

# ===============================
# 2. DATA LOADING (LOCAL PATH)
# ===============================
@st.cache_data
def load_data():
    try:
        # Assumes file is in the same GitHub folder
        return pd.read_csv("cleaned_ev_charging_data.csv")
    except FileNotFoundError:
        return None

df_raw = load_data()

if df_raw is None:
    st.error("Dataset 'cleaned_ev_charging_data.csv' not found. Please upload it to your GitHub repository.")
    st.stop()

# ===============================
# 3. PREPROCESSING
# ===============================
def preprocess(df):
    df_p = df.copy()
    # Basic Cleaning
    if 'Reviews (Rating)' in df_p.columns:
        df_p['Reviews (Rating)'] = df_p['Reviews (Rating)'].fillna(df_p['Reviews (Rating)'].median())
    
    # Label Encoding for categorical columns
    le = LabelEncoder()
    for col in ['Charger Type', 'Station Operator', 'Renewable Energy Source']:
        if col in df_p.columns:
            df_p[f'{col}_Enc'] = le.fit_transform(df_p[col].astype(str))

    # Scaling for ML
    scaler = MinMaxScaler()
    features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)']
    df_p[features] = scaler.fit_transform(df_p[features])
    return df_p, features

df_processed, cluster_cols = preprocess(df_raw)

# ===============================
# 4. SIDE-BY-SIDE ANALYTICS (STAGE 4)
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")

# WCSS for Elbow Plot
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    km.fit(df_processed[cluster_cols])
    wcss.append(km.inertia_)

# Layout: Side-by-Side
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Elbow Method")
    st.write("Identifies the optimal number of clusters.")
    # Set figsize to (5, 5) for square consistency
    fig_elbow, ax_elbow = plt.subplots(figsize=(5, 5))
    ax_elbow.plot(range(1, 11), wcss, marker='o', color='#1f77b4', linewidth=2)
    ax_elbow.set_xlabel('Number of Clusters')
    ax_elbow.set_ylabel('WCSS (Error)')
    st.pyplot(fig_elbow)

with col2:
    st.subheader("2. Cluster Results")
    k_value = st.slider("Select k (Clusters)", 2, 6, 3)
    
    # Run KMeans based on slider
    model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
    df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])
    
    # Set same figsize (5, 5) to match the Elbow plot
    fig_cluster, ax_cluster = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                    hue='Cluster', palette='Set1', s=100, ax=ax_cluster)
    ax_cluster.set_title(f"Stations Grouped into {k_value} Clusters")
    st.pyplot(fig_cluster)

# ===============================
# 5. MAP (NO ATTRIBUTION TEXT)
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution")

if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    # Use Pydeck to render the map without the bottom-corner attribution text
    st.pydeck_chart(pdk.Deck(
        map_style=None, # Removes the standard map tile layers/attribution
        initial_view_state=pdk.ViewState(
            latitude=df_raw['Latitude'].mean(),
            longitude=df_raw['Longitude'].mean(),
            zoom=2,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df_raw,
                get_position='[Longitude, Latitude]',
                get_color='[30, 144, 255, 160]', # Dodger Blue
                get_radius=150000,
            ),
        ],
    ))
else:
    st.warning("Latitude/Longitude columns not found.")

# ===============================
# 6. CORRELATION HEATMAP
# ===============================
st.divider()
st.header("📊 Stage 7: Feature Correlation")
fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
sns.heatmap(df_processed.select_dtypes(include=['number']).corr(), 
            annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8}, ax=ax_corr)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_corr)

if st.checkbox("Show Data Table"):
    st.dataframe(df_raw)
