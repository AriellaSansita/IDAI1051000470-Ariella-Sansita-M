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
st.markdown("---")

# ===============================
# 2. DATA LOADING & CLEANING
# ===============================
@st.cache_data
def load_and_deep_clean(file_path):
    try:
        # Note: Ensure 'cleaned_ev_charging_data.csv' exists in your directory
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        
        # Numeric Imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Categorical Imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("Unknown")
        
        if 'Station Operator' in df.columns:
            df['Station Operator'] = df['Station Operator'].astype(str).str.strip().str.title()
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Attempt to load data
df_raw = load_and_deep_clean("cleaned_ev_charging_data.csv")

if df_raw is None:
    st.warning("Please ensure 'cleaned_ev_charging_data.csv' is in the working directory.")
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
    # Only scale features that actually exist in the dataframe
    existing = [f for f in cluster_features if f in df_proc.columns]
    
    if existing:
        scaler = MinMaxScaler()
        df_proc[existing] = scaler.fit_transform(df_proc[existing])
    
    return df_proc, existing

df_processed, cluster_cols = preprocess_for_ml(df_raw)

# ===============================
# 4. STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================
st.header("📊 Stage 1: Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    if 'Usage Stats (avg users/day)' in df_raw.columns:
        st.subheader("Usage Statistics Distribution")
        fig_h, ax_h = plt.subplots(figsize=(8, 5))
        sns.histplot(df_raw['Usage Stats (avg users/day)'], bins=20, kde=True, color='teal', ax=ax_h)
        st.pyplot(fig_h)

with col2:
    if 'Station Operator' in df_raw.columns and 'Cost (USD/kWh)' in df_raw.columns:
        st.subheader("Cost vs Station Operator")
        fig_b, ax_b = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_raw, x='Station Operator', y='Cost (USD/kWh)', palette='Set2', ax=ax_b)
        plt.xticks(rotation=45)
        st.pyplot(fig_b)

# ===============================
# 5. STAGE 4: CLUSTERING
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")

if len(cluster_cols) > 0:
    # Elbow Method
    wcss = []
    max_k = 11 if len(df_processed) > 11 else len(df_processed)
    for i in range(1, max_k):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(df_processed[cluster_cols])
        wcss.append(kmeans.inertia_)
    
    fig_elbow, ax_elbow = plt.subplots(figsize=(12, 3))
    ax_elbow.plot(range(1, max_k), wcss, marker='o', color='#1f77b4')
    ax_elbow.set_title("Elbow Method for Optimal K")
    st.pyplot(fig_elbow)

    k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)
    model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
    df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

    col_scat, col_pers = st.columns([2, 1])
    with col_scat:
        fig_cluster, ax_cluster = plt.subplots(figsize=(12, 6)) 
        sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                        hue='Cluster', palette='Set1', s=150, alpha=0.7, ax=ax_cluster)
        st.pyplot(fig_cluster)

    with col_pers:
        # Summary calculations
        relevant_metrics = [c for c in ['Charging Capacity (kW)', 'Usage Stats (avg users/day)', 'Cost (USD/kWh)'] if c in df_raw.columns]
        cluster_summary = df_raw.groupby('Cluster')[relevant_metrics].mean()
        st.write("### Segment Personas")
        avg_usage = cluster_summary['Usage Stats (avg users/day)'].mean()
        
        for i in range(k_value):
            row = cluster_summary.loc[i]
            if row['Usage Stats (avg users/day)'] >= avg_usage:
                st.success(f"**Cluster {i}: High-Demand Hubs**")
            else:
                st.info(f"**Cluster {i}: Growth Potential**")
else:
    st.error("Not enough numeric features found for clustering.")

# ===============================
# 6. STAGE 5: ANOMALY DETECTION
# ===============================
st.divider()
st.header("🔍 Stage 5: Anomaly Detection")

def detect_outliers(df, col):
    if col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        return outliers
    return pd.DataFrame()

usage_outliers = detect_outliers(df_raw, 'Usage Stats (avg users/day)')
cost_outliers = detect_outliers(df_raw, 'Cost (USD/kWh)')

m1, m2 = st.columns(2)

# Display Metrics with logic for "None"
if len(usage_outliers) > 0:
    m1.metric("Usage Outliers", len(usage_outliers), delta="Action Required", delta_color="inverse")
    st.warning(f"Detected {len(usage_outliers)} unusual usage patterns.")
else:
    m1.metric("Usage Outliers", "None")
    m1.success("✅ Usage patterns are consistent.")

if len(cost_outliers) > 0:
    m2.metric("Cost Outliers", len(cost_outliers), delta="Check Pricing", delta_color="inverse")
    st.warning(f"Detected {len(cost_outliers)} pricing anomalies.")
else:
    m2.metric("Cost Outliers", "None")
    m2.success("✅ Pricing is within normal range.")

# ===============================
# 7. STAGE 6: ASSOCIATION RULE MINING
# ===============================
st.divider()
st.header("🔗 Stage 6: Association Rule Mining")
rules_df = None 

try:
    df_rules = pd.DataFrame()
    df_rules['HighUsage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].quantile(0.5)
    df_rules['FastCharger'] = df_raw['Charging Capacity (kW)'] > df_raw['Charging Capacity (kW)'].quantile(0.5)
    
    if 'Renewable Energy Source' in df_raw.columns:
        # Check if it's already boolean or contains strings like 'Yes'
        df_rules['Renewable'] = df_raw['Renewable Energy Source'].apply(lambda x: True if str(x).lower() in ['yes', 'true', '1.0', '1'] else False)
    
    df_rules['PremiumPrice'] = df_raw['Cost (USD/kWh)'] > df_raw['Cost (USD/kWh)'].quantile(0.5)
    
    # MLxtend requires boolean types specifically
    df_rules = df_rules.astype(bool)
    
    freq = apriori(df_rules, min_support=0.05, use_colnames=True)
    
    if not freq.empty:
        # Updated association_rules call for modern mlxtend
        rules_df = association_rules(freq, metric="lift", min_threshold=1.0)
        if not rules_df.empty:
            rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
            rules_df = rules_df.sort_values('lift', ascending=False)
            st.dataframe(rules_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10), use_container_width=True)
        else:
            st.write("No strong association rules found with current thresholds.")
except Exception as e:
    st.error(f"Association Analysis error: {e}")

# ===============================
# 8. STAGE 8: GEOSPATIAL & SUMMARY
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic & Insights")

if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    # Drop rows with NaN in coordinates for PyDeck
    map_data = df_raw.dropna(subset=['Latitude', 'Longitude'])
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=map_data['Latitude'].mean(), 
            longitude=map_data['Longitude'].mean(), 
            zoom=4
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer', 
                data=map_data, 
                get_position='[Longitude, Latitude]', 
                get_color='[255, 100, 0, 160]', 
                radius_min_pixels=5
            ),
        ],
    ))
else:
    st.info("Geographic data (Latitude/Longitude) not found in dataset.")

st.subheader("Key Findings")
rule_text = "No strong patterns found"
if rules_df is not None and not rules_df.empty:
    rule_text = f"Significant link between '{rules_df.iloc[0]['antecedents']}' and '{rules_df.iloc[0]['consequents']}'"

st.info(f"""
- **Anomalies:** Identified {len(usage_outliers)} stations with irregular usage and {len(cost_outliers)} with irregular pricing.
- **Rules Analysis:** {rule_text}.
""")

if st.checkbox("View Final Data Table"):
    st.dataframe(df_raw)
