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

# Fix for Matplotlib thread-safety in Streamlit
plt.rcParams.update({'figure.max_open_warning': 0})

# ===============================
# 2. DATA LOADING & CLEANING (Stage 2)
# ===============================
@st.cache_data
def load_and_deep_clean(file_path):
    try:
        df = pd.read_csv(file_path) 
        df = df.drop_duplicates()
        
        # Numeric cleanup: Fill missing with Median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Categorical cleanup: Strip strings and fill with Mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            
        return df
    except FileNotFoundError:
        st.error(f"❌ File '{file_path}' not found. Please ensure it is in the GitHub repository.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# DATASET PATH: Change this if your filename is different
DATA_FILE = "cleaned_ev_charging_data.csv"
df_raw = load_and_deep_clean(DATA_FILE)

if df_raw is None: 
    st.info("💡 Tip: Upload your CSV to the same folder as this script on GitHub.")
    st.stop()

# ===============================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ===============================
@st.cache_data
def preprocess_for_ml(df):
    df_proc = df.copy()
    
    # 1. Label Encoding for categoricals
    cat_to_encode = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 'Availability']
    for col in cat_to_encode:
        if col in df_proc.columns:
            le = LabelEncoder()
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col])

    # 2. Scaling for K-Means (Distance-based)
    cluster_features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Distance to City (km)', 'Availability_Enc']
    existing = [f for f in cluster_features if f in df_proc.columns]
    
    scaler = MinMaxScaler()
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
        fig_h, ax_h = plt.subplots(figsize=(6, 4))
        sns.histplot(df_raw['Usage Stats (avg users/day)'], bins=20, kde=True, color='teal', ax=ax_h)
        st.pyplot(fig_h)
    with col2:
        st.subheader("Cost vs Station Operator")
        fig_b, ax_b = plt.subplots(figsize=(6, 4))
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
    else:
        st.warning("Column 'Installation Year' not found for growth analysis.")

with tab3:
    st.subheader("Feature Correlation Heatmap")
    if len(cluster_cols) > 1:
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_processed[cluster_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

# ===============================
# 5. STAGE 4: CLUSTERING ANALYSIS
# ===============================
st.divider()
st.header("🤖 Stage 4: Machine Learning - Station Clustering")

k_value = st.slider("Select Number of Clusters (k)", 2, 6, 3)
model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

col_scat, col_pers = st.columns([2, 1])
with col_scat:
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 6)) 
    sns.scatterplot(data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                    hue='Cluster', palette='viridis', s=100, ax=ax_cluster, style='Cluster')
    st.pyplot(fig_cluster)

with col_pers:
    st.write("### Segment Personas")
    cluster_summary = df_raw.groupby('Cluster')[['Usage Stats (avg users/day)', 'Cost (USD/kWh)']].mean()
    mean_usage = df_raw['Usage Stats (avg users/day)'].mean()
    
    for cluster_id, row in cluster_summary.iterrows():
        if row['Usage Stats (avg users/day)'] > mean_usage:
            st.success(f"**Cluster {cluster_id}: High-Demand Hubs** (Avg: {row['Usage Stats (avg users/day)']:.1f} users)")
        else:
            st.info(f"**Cluster {cluster_id}: Emerging Stations** (Avg: {row['Usage Stats (avg users/day)']:.1f} users)")

# ===============================
# 6. STAGE 5: ASSOCIATION RULE MINING
# ===============================
st.divider()
st.header("🔗 Stage 5: Association Rule Mining")
try:
    # Creating binary features for Market Basket Analysis logic
    df_rules = pd.DataFrame()
    df_rules['HighUsage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].median()
    df_rules['FastCharger'] = df_raw['Charging Capacity (kW)'] > 50
    # Robust boolean check for Renewable
    df_rules['Renewable'] = df_raw['Renewable Energy Source'].astype(str).str.lower().isin(['yes', 'true', '1'])
    
    # Minimum support of 5% to capture meaningful patterns
    frequent_itemsets = apriori(df_rules, min_support=0.05, use_colnames=True)
    
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
        
        if not rules.empty:
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            fig_rules, ax_rules = plt.subplots(figsize=(10, 4))
            sns.barplot(data=rules.sort_values('lift', ascending=False).head(10), 
                        x='lift', y='antecedents', hue='consequents', ax=ax_rules)
            st.subheader("Top Rules by Lift (Strategic Connections)")
            st.pyplot(fig_rules)
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']], use_container_width=True)
        else:
            st.warning("No association rules met the Lift threshold.")
    else:
        st.warning("No frequent itemsets found. Adjust min_support.")
except Exception as e:
    st.error(f"⚠️ Rule Mining Error: {e}")

# ===============================
# 7. STAGE 6: ANOMALY DETECTION
# ===============================
st.divider()
st.header("🔍 Stage 6: Anomaly Detection")

def get_outliers(df, col):
    if col not in df.columns: return pd.DataFrame()
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

usage_anoms = get_outliers(df_raw, 'Usage Stats (avg users/day)')
maint_anoms = get_outliers(df_raw, 'Maintenance Frequency')

c1, c2 = st.columns(2)
c1.metric("Usage Anomalies", len(usage_anoms))
c2.metric("Maintenance Outliers", len(maint_anoms))

if not usage_anoms.empty:
    with st.expander("View Outlier Station Details"):
        st.write(usage_anoms[['Station Operator', 'Usage Stats (avg users/day)', 'Cluster']].head(10))

# ===============================
# 8. STAGE 8: GEOSPATIAL & DEPLOYMENT
# ===============================
st.divider()
st.header("📍 Stage 8: Geographic Distribution")
if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=df_raw['Latitude'].mean(), 
            longitude=df_raw['Longitude'].mean(), 
            zoom=3, pitch=45
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=df_raw,
                get_position='[Longitude, Latitude]',
                radius=300,
                elevation_scale=10,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ))
else:
    st.info("Latitude/Longitude data missing; skipping map visualization.")

st.info("**Analysis Summary:** Use the findings above to optimize station placement and pricing strategies.")
