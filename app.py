import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# --- CONFIG ---
st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

# ===============================
# DATA LOADING (LOCAL PATH)
# ===============================
try:
    # This looks for the file in the same folder as this script
    df_raw = pd.read_csv("cleaned_ev_charging_data.csv")
    st.success("Data loaded successfully from local directory!")
except FileNotFoundError:
    st.error("Dataset 'cleaned_ev_charging_data.csv' not found. Please ensure it is in the same GitHub folder as this script.")
    st.stop()

# ===============================
# PREPROCESSING FUNCTION
# ===============================
def preprocess_data(df):
    df = df.copy()

    # Handle Missing Values
    if 'Reviews (Rating)' in df.columns:
        df['Reviews (Rating)'] = df['Reviews (Rating)'].fillna(df['Reviews (Rating)'].median())
    if 'Renewable Energy Source' in df.columns:
        df['Renewable Energy Source'] = df['Renewable Energy Source'].fillna(df['Renewable Energy Source'].mode()[0])

    # Remove Duplicates
    if 'Station ID' in df.columns:
        df = df.drop_duplicates(subset=['Station ID'])

    # Encode Categorical
    le = LabelEncoder()
    cat_cols = ['Charger Type', 'Station Operator', 'Renewable Energy Source']
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Normalize Numeric
    scaler = MinMaxScaler()
    num_cols = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Distance to City (km)']
    existing_cols = [c for c in num_cols if c in df.columns]
    if existing_cols:
        df[existing_cols] = scaler.fit_transform(df[existing_cols])

    return df

# ===============================
# MAIN DASHBOARD
# ===============================
if st.checkbox("Show Raw Data"):
    st.write(df_raw.head())

df_processed = preprocess_data(df_raw)

st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    if 'Usage Stats (avg users/day)' in df_raw.columns:
        st.subheader("Usage Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_raw['Usage Stats (avg users/day)'], kde=True, ax=ax)
        st.pyplot(fig)

with col2:
    if 'Station Operator' in df_raw.columns and 'Cost (USD/kWh)' in df_raw.columns:
        st.subheader("Cost by Station Operator")
        fig, ax = plt.subplots()
        sns.boxplot(x=df_raw['Station Operator'], y=df_raw['Cost (USD/kWh)'], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- FIXED HEATMAP ---
st.subheader("Correlation Heatmap")
fig_heat, ax_heat = plt.subplots(figsize=(10, 7)) 
sns.heatmap(
    df_processed.corr(numeric_only=True), 
    annot=True, 
    fmt=".2f",           # Limits decimals
    annot_kws={"size": 8}, # Smaller font for numbers
    cmap='coolwarm', 
    ax=ax_heat
)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_heat)

if st.checkbox("Show Processed Data"):
    st.write(df_processed.head())
