import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="EV SmartCharging: Strategic Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: Professional EV Behavior Patterns")
st.markdown("---")

st.sidebar.header("🎯 Dashboard Controls")

# ===============================
# DATA LOADING
# ===============================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['Station ID']) if 'Station ID' in df.columns else df.drop_duplicates()

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    return df

df_raw = load_data("cleaned_ev_charging_data.csv")

# ===============================
# FILTER
# ===============================
if 'Station Operator' in df_raw.columns:
    ops = st.sidebar.multiselect("Operator", df_raw['Station Operator'].unique(), default=df_raw['Station Operator'].unique())
    df_filtered = df_raw[df_raw['Station Operator'].isin(ops)].copy()
else:
    df_filtered = df_raw.copy()

# ===============================
# PREPROCESSING
# ===============================
@st.cache_data
def preprocess(df):
    df_proc = df.copy()

    cat_cols = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 'Availability']
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[col + "_Enc"] = LabelEncoder().fit_transform(df_proc[col].astype(str))

    features = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Distance to City (km)', 'Availability_Enc']
    features = [f for f in features if f in df_proc.columns]

    scaler = MinMaxScaler()
    df_proc[features] = scaler.fit_transform(df_proc[features])

    return df_proc, features

df_ml, cluster_cols = preprocess(df_filtered)

# ===============================
# EDA
# ===============================
st.header("📊 EDA")

# Histogram
st.subheader("Usage Distribution")
fig, ax = plt.subplots()
sns.histplot(df_filtered['Usage Stats (avg users/day)'], kde=True, ax=ax)
st.pyplot(fig)
st.caption("Most stations cluster around moderate usage levels.")

# Line chart
if 'Installation Year' in df_filtered.columns:
    yearly = df_filtered.groupby('Installation Year')['Usage Stats (avg users/day)'].mean()
    fig, ax = plt.subplots()
    yearly.plot(ax=ax)
    st.pyplot(fig)
    st.caption("Usage trends show infrastructure growth impact.")

# Boxplot
fig, ax = plt.subplots()
sns.boxplot(data=df_filtered, x='Station Operator', y='Cost (USD/kWh)', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df_filtered.select_dtypes(include=[np.number]).corr(), annot=True, ax=ax)
st.pyplot(fig)

# ===============================
# CLUSTERING
# ===============================
st.header("🤖 Clustering")

# Elbow Method
inertia = []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_ml[cluster_cols])
    inertia.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(k_range, inertia, marker='o')
ax.set_title("Elbow Method")
st.pyplot(fig)

k_val = st.sidebar.slider("Clusters", 2, 6, 3)

model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
df_filtered['Cluster'] = model.fit_predict(df_ml[cluster_cols])

fig, ax = plt.subplots()
sns.scatterplot(data=df_filtered, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', hue='Cluster', ax=ax)
st.pyplot(fig)

# ===============================
# ANOMALY DETECTION
# ===============================
st.header("🔍 Anomalies")

def outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

usage_out = outliers(df_raw, 'Usage Stats (avg users/day)')
cost_out = outliers(df_raw, 'Cost (USD/kWh)')

st.metric("Usage Outliers", len(usage_out))
st.metric("Cost Outliers", len(cost_out))

# ===============================
# ASSOCIATION RULES
# ===============================
st.header("🔗 Association Rules")

df_rules = pd.DataFrame()
df_rules['HighUsage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].median()
df_rules['Fast'] = df_raw['Charging Capacity (kW)'] > df_raw['Charging Capacity (kW)'].median()
df_rules['Premium'] = df_raw['Cost (USD/kWh)'] > df_raw['Cost (USD/kWh)'].median()

df_rules = df_rules.astype(bool)

freq = apriori(df_rules, min_support=0.05, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.0)

if not rules.empty:
    rules['antecedents'] = rules['antecedents'].astype(str)
    rules['consequents'] = rules['consequents'].astype(str)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# ===============================
# MAP
# ===============================
st.header("📍 Map")

if 'Latitude' in df_raw.columns:
    map_data = df_raw.dropna(subset=['Latitude', 'Longitude'])
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=map_data['Latitude'].mean(),
            longitude=map_data['Longitude'].mean(),
            zoom=4),
        layers=[pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[Longitude, Latitude]',
            get_color='[255,0,0]',
            radius_min_pixels=5)]
    ))
    
