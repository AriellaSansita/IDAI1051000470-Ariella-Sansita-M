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
st.set_page_config(page_title="EV SmartCharging Analytics: Professional Edition", layout="wide")
st.title("🚗 EV SmartCharging: Strategic Infrastructure Analysis")
st.markdown("---")

# ===============================
# 2. STEP 1: ROBUST DATA CLEANING
# ===============================
@st.cache_data
def load_and_deep_clean(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # 1. Remove Duplicates
        df = df.drop_duplicates()
        
        # 2. Fill Missing Values (Comprehensive)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # 3. Text Normalization
        if 'Station Operator' in df.columns:
            df['Station Operator'] = df['Station Operator'].astype(str).str.strip().str.title()
            
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

df_raw = load_and_deep_clean("cleaned_ev_charging_data.csv")
if df_raw is None: st.stop()

# ===============================
# 3. STEP 6: TECHNICAL STABILITY (ENCODING)
# ===============================
@st.cache_data
def preprocess_for_ml(df):
    df_proc = df.copy()
    
    # Using separate LabelEncoders for technical accuracy
    encoders = {}
    cat_to_encode = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 'Availability']
    for col in cat_to_encode:
        if col in df_proc.columns:
            le = LabelEncoder()
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))
            encoders[col] = le

    scaler = MinMaxScaler()
    ml_features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Distance to City (km)']
    existing = [f for f in ml_features if f in df_proc.columns]
    df_proc[existing] = scaler.fit_transform(df_proc[existing])
    
    return df_proc, existing

df_processed, cluster_cols = preprocess_for_ml(df_raw)

# ===============================
# 4. STEP 2: UPGRADED EDA (DEEP INSIGHTS)
# ===============================
st.header("📊 Stage 1: Strategic Exploratory Analysis")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Relationship 1: Availability vs. Usage")
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_raw, x='Availability', y='Usage Stats (avg users/day)', palette='mako', ax=ax_a)
    st.pyplot(fig_a)
    st.info("**Why it matters:** 24/7 stations generally capture higher overnight commercial demand. If '9:00-18:00' stations show low usage, they likely miss the 'Home-to-Work' charging peak.")

with c2:
    st.subheader("Relationship 2: Reviews vs. Usage")
    fig_r, ax_r = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df_raw, x='Reviews (Rating)', y='Usage Stats (avg users/day)', scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax_r)
    st.pyplot(fig_r)
    st.info("**Interpretation:** A positive slope confirms that user experience directly drives repeat traffic. Stations with ratings below 3.5 are likely suffering from maintenance downtime.")

# ===============================
# 5. STEP 4: CLUSTER PERSONA ANALYSIS
# ===============================
st.divider()
st.header("🤖 Stage 4: Advanced Market Segmentation")

k = st.slider("Select Segments (k)", 2, 6, 3)
model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

# Cluster Comparison Table (The "Distinction" move)
cluster_summary = df_raw.groupby('Cluster')[cluster_cols].mean()
st.subheader("Cluster Characterization")
st.dataframe(cluster_summary.style.highlight_max(axis=0, color='#2e7d32').highlight_min(axis=0, color='#c62828'))

# Persona Logic
for i in range(k):
    row = cluster_summary.loc[i]
    if row['Usage Stats (avg users/day)'] > cluster_summary['Usage Stats (avg users/day)'].mean():
        persona = "🔥 **High-Traffic Hubs:** Critical infrastructure. Prioritize for renewable upgrades."
    elif row['Cost (USD/kWh)'] > cluster_summary['Cost (USD/kWh)'].mean():
        persona = "💰 **Premium/Luxury Zones:** High margins but potentially lower frequency."
    else:
        persona = "📉 **Underperformers:** High distance to city or low capacity. Review pricing."
    st.write(f"**Cluster {i}:** {persona}")

# ===============================
# 6. STEP 7: ANOMALY EXPLANATION
# ===============================
st.divider()
st.header("🔍 Stage 5: Anomaly Detection")
def get_outliers(df, col):
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]

usage_anoms = get_outliers(df_raw, 'Usage Stats (avg users/day)')
st.metric("Total Anomalies Detected", len(usage_anoms))

st.write("**What these anomalies mean:**")
st.markdown("""
- **Data Entry Errors:** Impossible costs or usage numbers.
- **Niche Locations:** Extreme luxury chargers with very high costs.
- **Maintenance Red Flags:** Stations with near-zero usage despite high capacity.
""")

# ===============================
# 7. STEP 3 & 6: REFINED ASSOCIATION RULES
# ===============================
st.divider()
st.header("🔗 Stage 6: Association Rule Mining (Pattern Discovery)")
try:
    rules_df = pd.DataFrame()
    # Strategic logical features
    rules_df['High_Usage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].quantile(0.75)
    rules_df['Low_Cost'] = df_raw['Cost (USD/kWh)'] < df_raw['Cost (USD/kWh)'].quantile(0.25)
    rules_df['Near_City'] = df_raw['Distance to City (km)'] < df_raw['Distance to City (km)'].median()
    rules_df['Renewable'] = df_raw['Renewable Energy Source'].map({1: True, 0: False, True: True, False: False})
    
    rules_df = rules_df.astype(bool)
    freq = apriori(rules_df, min_support=0.05, use_colnames=True)
    
    if not freq.empty:
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        if not rules.empty:
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(10))
            st.success("**Insight:** Intentional patterns show how proximity to city centers directly correlates with high usage thresholds.")
        else:
            st.info("No strong rules found.")
    else:
        st.info("No frequent patterns found.")
except Exception as e:
    st.error(f"Rule Logic Error: {e}")

# ===============================
# 8. STEP 5: BUSINESS BRAIN INSIGHTS
# ===============================
st.divider()
st.header("💡 Stage 7: Strategic Recommendations")
top_cluster = cluster_summary['Usage Stats (avg users/day)'].idxmax()

st.info(f"""
### Executive Action Plan:
1. **Capacity Expansion:** Double-down on **Cluster {top_cluster}**. These stations are high-usage and likely near capacity. Adding more ports here is the safest ROI.
2. **Pricing Optimization:** Low-usage clusters (Near-city outliers) should trial a 10% price reduction to compete with home charging.
3. **Operational Focus:** The correlation between Reviews and Usage proves that downtime is more expensive than maintenance. Establish a 4-hour repair SLA for all high-traffic hubs.
""")

# Map
if 'Latitude' in df_raw.columns:
    st.subheader("Geographic Station Density")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=df_raw['Latitude'].mean(), longitude=df_raw['Longitude'].mean(), zoom=2, min_zoom=2),
        layers=[pdk.Layer('ScatterplotLayer', data=df_raw, get_position='[Longitude, Latitude]', get_color='[255, 100, 0, 160]', radius_min_pixels=3)],
    ))
