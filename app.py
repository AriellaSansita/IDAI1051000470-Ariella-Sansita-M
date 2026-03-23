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

# --- ADD THIS TO STAGE 5 (ANOMALY DETECTION) ---
st.subheader("Strategic Anomalies: High Cost vs. Low Performance")

# Logic to find stations in the top 25% of cost but bottom 25% of usage
high_cost_threshold = df_filtered['Cost (USD/kWh)'].quantile(0.75)
low_usage_threshold = df_filtered['Usage Stats (avg users/day)'].quantile(0.25)

strategic_anomalies = df_filtered[
    (df_filtered['Cost (USD/kWh)'] >= high_cost_threshold) & 
    (df_filtered['Usage Stats (avg users/day)'] <= low_usage_threshold)
]

if not strategic_anomalies.empty:
    st.warning(f"Found {len(strategic_anomalies)} stations with High Cost but Low Usage. These may require pricing adjustments.")
    st.dataframe(strategic_anomalies[['Station ID', 'Station Operator', 'Cost (USD/kWh)', 'Usage Stats (avg users/day)']])
else:
    st.success("✅ No 'High Cost / Low Usage' anomalies found.")

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
# --- ADD THIS TO STAGE 6 (ASSOCIATION RULES) ---
if rules_df is not None and not rules_df.empty:
    st.subheader("Top Rules by Lift (Strength of Association)")
    top_rules = rules_df.sort_values('lift', ascending=False).head(5)
    
    # Create a label combining Antecedents -> Consequents
    top_rules['rule_label'] = top_rules['antecedents'] + " -> " + top_rules['consequents']
    
    fig_rules, ax_rules = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_rules, x='lift', y='rule_label', palette="viridis", ax=ax_rules)
    ax_rules.set_title("Top 5 Strongest Associations")
    st.pyplot(fig_rules)

# ===============================
# 8. STAGE 8: GEOSPATIAL & SUMMARY
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Cluster Distribution")

# Use df_filtered because it contains the 'Cluster' labels from Stage 4
if 'Latitude' in df_filtered.columns and 'Longitude' in df_filtered.columns and 'Cluster' in df_filtered.columns:
    
    # Define a color dictionary for the clusters
    cluster_colors = {
        0: [255, 0, 0, 160],    # Red
        1: [0, 255, 0, 160],    # Green
        2: [0, 0, 255, 160],    # Blue
        3: [255, 165, 0, 160],  # Orange
        4: [128, 0, 128, 160],  # Purple
        5: [0, 255, 255, 160]   # Cyan
    }

    # FIX: Use a lambda or map without fillna(list) to avoid TypeError
    map_df = df_filtered.dropna(subset=['Latitude', 'Longitude']).copy()
    map_df['color'] = map_df['Cluster'].apply(lambda x: cluster_colors.get(x, [200, 200, 200, 160]))

    # Display the Interactive Map as required by Stage 8 
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=map_df['Latitude'].mean(),
            longitude=map_df['Longitude'].mean(),
            zoom=3,
            pitch=45
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=map_df,
                get_position='[Longitude, Latitude]',
                get_color='color',
                get_radius=15000,
                pickable=True
            ),
        ],
        tooltip={"text": "Operator: {Station Operator}\nCluster: {Cluster}"}
    ))
else:
    st.info("Ensure clustering is completed and coordinates are available to view the map.")

# Summary Section 
st.subheader("Key Findings & Strategic Insights")
rule_text = "No strong patterns found"
if rules_df is not None and not rules_df.empty:
    rule_text = f"Significant relationship discovered between '{rules_df.iloc[0]['antecedents']}' and '{rules_df.iloc[0]['consequents']}'"

st.info(f"""
- **Anomalies:** Identified {len(usage_outliers)} stations with irregular usage patterns and {len(cost_outliers)} pricing outliers. 
- **Market Basket Analysis:** {rule_text}. 
- **Infrastructure Growth:** Yearly trends indicate shifts in charging demand since installation. 
""")

if st.checkbox("View Final Data Table"):
    st.dataframe(df_filtered)
