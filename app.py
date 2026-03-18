import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EV SmartCharging Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to make it look even cleaner
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_index=True)

# --- 2. DATA LOADING ---
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_csv("cleaned_ev_charging_data.csv")
        
        # Preprocessing for ML
        df_proc = df.copy()
        if 'Reviews (Rating)' in df_proc.columns:
            df_proc['Reviews (Rating)'] = df_proc['Reviews (Rating)'].fillna(df_proc['Reviews (Rating)'].median())
        
        le = LabelEncoder()
        for col in ['Charger Type', 'Station Operator', 'Renewable Energy Source']:
            if col in df_proc.columns:
                df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

        scaler = MinMaxScaler()
        features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)']
        df_proc[features] = scaler.fit_transform(df_proc[features])
        
        return df, df_proc, features
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

df_raw, df_processed, cluster_cols = load_and_preprocess()

if df_raw is not None:
    # --- 3. SIDEBAR CONTROLS ---
    st.sidebar.title("🛠️ Analytics Settings")
    st.sidebar.markdown("Adjust parameters for real-time data mining.")
    
    k_value = st.sidebar.slider("Select Cluster Count (k)", 2, 6, 3, help="Determines how many groups the AI splits the stations into.")
    
    show_raw = st.sidebar.checkbox("Show Data Table")
    
    st.sidebar.divider()
    st.sidebar.info("This dashboard uses K-Means Clustering to identify patterns in EV charging behavior.")

    # --- 4. HEADER & METRICS ---
    st.title("🚗 SmartCharging Analytics Dashboard")
    st.markdown("### Strategic Insights for EV Infrastructure")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Stations", len(df_raw))
    m2.metric("Avg. Capacity", f"{df_raw['Charging Capacity (kW)'].mean():.1f} kW")
    m3.metric("Avg. Cost", f"${df_raw['Cost (USD/kWh)'].mean():.2f}/kWh")
    m4.metric("Renewable Use", f"{(df_raw['Renewable Energy Source'] == 'Yes').mean()*100:.0f}%")

    # --- 5. CLUSTERING SECTION (STAGE 4) ---
    st.divider()
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("📉 Elbow Method")
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            km.fit(df_processed[cluster_cols])
            wcss.append(km.inertia_)
        
        fig_elbow, ax_elbow = plt.subplots(figsize=(5, 4))
        plt.style.use('ggplot')
        ax_elbow.plot(range(1, 11), wcss, marker='o', color='#3498db', linewidth=2)
        ax_elbow.set_title('Finding Optimal K', fontsize=10)
        st.pyplot(fig_elbow)

    with col_right:
        st.subheader("🎯 Station Segmentations")
        model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
        df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])
        
        fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                        hue='Cluster', palette='viridis', s=100, alpha=0.7, ax=ax_cluster)
        ax_cluster.set_title(f"Market Segments (k={k_value})", fontsize=12)
        st.pyplot(fig_cluster)

    # --- 6. GEOSPATIAL & CORRELATION ---
    st.divider()
    tab1, tab2 = st.tabs(["📍 Geographic Map", "📊 Statistical Correlations"])

    with tab1:
        st.subheader("Station Locations")
        map_df = df_raw.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
        st.map(map_df, color='#2ecc71')

    with tab2:
        st.subheader("Feature Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        corr = df_processed.select_dtypes(include=['number']).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', ax=ax_corr)
        st.pyplot(fig_corr)

    # --- 7. FINAL INSIGHTS ---
    st.divider()
    st.subheader("📝 Automated Analysis Summary")
    
    # Simple logic to generate a conclusion
    top_cluster = df_raw.groupby('Cluster')['Usage Stats (avg users/day)'].mean().idxmax()
    st.success(f"""
    **Conclusion:** Cluster **{top_cluster}** represents your 'High-Value' stations with the highest average traffic. 
    Focus infrastructure maintenance on these coordinates to maximize ROI.
    """)

    if show_raw:
        st.dataframe(df_raw.style.highlight_max(axis=0, subset=['Usage Stats (avg users/day)']))
