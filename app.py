import streamlit as st
import pandas as pd
import matplotlib.pyplot as pd
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
st.markdown("---")

# ===============================
# 2. DATA LOADING & CLEANING (Stage 2)
# ===============================
@st.cache_data
def load_and_deep_clean(file_path):
    try:
        # Source dataset must include columns defined in Task 2 Stage 1 
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

df_raw = load_and_deep_clean("cleaned_ev_charging_data.csv")
if df_raw is None: 
    st.stop()

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
# 4. STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================
st.header("📊 Stage 3: Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Demand & Cost", "Growth Trends", "Correlations"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Usage Statistics Distribution")
        fig_h, ax_h = plt.subplots()
        sns.histplot(df_raw['Usage Stats (avg users/day)'], bins=20, kde=True, color='teal', ax=ax_h)
        st.pyplot(fig_h)
    with col2:
        st.subheader("Cost vs Station Operator")
        fig_b, ax_b = plt.subplots()
        sns.boxplot(data=df_raw, x='Station Operator', y='Cost (USD/kWh)', palette='Set2', ax=ax_b)
        plt.xticks(rotation=45)
        st.pyplot(fig_b)

with tab2:
    st.subheader("Growth Over Time (Usage vs Installation Year)")
    if 'Installation Year' in df_raw.columns:
        yearly_usage = df_raw.groupby('Installation Year')['Usage Stats (avg users/day)'].mean().reset_index()
        fig_line, ax_line = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=yearly_usage, x='Installation Year', y='Usage Stats (avg users/day)', marker='o', ax=ax_line)
        st.pyplot(fig_line)

with tab3:
    st.subheader("Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_processed[cluster_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

# ===============================
# 5. STAGE 4: CLUSTERING ANALYSIS
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")

k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)
model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

col_scat, col_pers = st.columns([2, 1])
with col_scat:
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6)) 
    sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', hue='Cluster', palette='viridis', s=100, ax=ax_cluster)
    st.pyplot(fig_cluster)

with col_pers:
    st.write("### Segment Personas")
    cluster_summary = df_raw.groupby('Cluster')[['Usage Stats (avg users/day)', 'Cost (USD/kWh)']].mean()
    for i in range(k_value):
        avg_usage = cluster_summary.loc[i, 'Usage Stats (avg users/day)']
        if avg_usage > df_raw['Usage Stats (avg users/day)'].mean():
            st.success(f"**Cluster {i}: High-Demand Hubs**")
        else:
            st.info(f"**Cluster {i}: Emerging Stations**")

# ===============================
# 6. STAGE 5: ASSOCIATION RULE MINING
# ===============================
st.divider()
st.header("🔗 Stage 5: Association Rule Mining")
try:
    # Binary encoding for Association Rules 
    df_rules = pd.DataFrame()
    df_rules['HighUsage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].median()
    df_rules['FastCharger'] = df_raw['Charging Capacity (kW)'] > 50
    df_rules['Renewable'] = df_raw['Renewable Energy Source'].apply(lambda x: True if str(x).strip().lower() == 'yes' else False)
    
    freq = apriori(df_rules, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.1)
    
    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Visualization of Rules
        fig_rules, ax_rules = plt.subplots(figsize=(10, 4))
        sns.barplot(data=rules.head(10), x='lift', y='antecedents', hue='consequents', ax=ax_rules)
        st.subheader("Top Rules by Lift")
        st.pyplot(fig_rules)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        st.warning("No strong associations found with current thresholds.")
except Exception as e:
    st.error(f"Rule Mining Error: {e}")

# ===============================
# 7. STAGE 6: ANOMALY DETECTION
# ===============================
st.divider()
st.header("🔍 Stage 6: Anomaly Detection")
def get_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

anomalies = get_outliers(df_raw, 'Usage Stats (avg users/day)')
maintenance_anomalies = get_outliers(df_raw, 'Maintenance Frequency') if 'Maintenance Frequency' in df_raw.columns else pd.DataFrame()

c1, c2 = st.columns(2)
c1.metric("Usage Anomalies", len(anomalies))
c2.metric("Maintenance Outliers", len(maintenance_anomalies))

if not anomalies.empty:
    st.write("### Detailed Anomaly Data")
    st.dataframe(anomalies[['Station Operator', 'Usage Stats (avg users/day)', 'Cost (USD/kWh)']].head(10))

# ===============================
# 8. STAGE 8: DEPLOYMENT & GEOGRAPHIC INSIGHTS
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution")
if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=df_raw['Latitude'].mean(), longitude=df_raw['Longitude'].mean(), zoom=3, pitch=50),
        layers=[pdk.Layer('HexagonLayer', data=df_raw, get_position='[Longitude, Latitude]', radius=200, elevation_scale=4, elevation_range=[0, 1000], pickable=True, extinguished=True)]
    ))

st.subheader("Strategic Recommendations")
st.info("""
- **Expansion:** Prioritize renewable-integrated stations in high-usage clusters.
- **Maintenance:** Investigate the flagged outliers for potential hardware failure.
- **Pricing:** Dynamic pricing models should target 'Emerging Stations' to boost utilization.
""")
