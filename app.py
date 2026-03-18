import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

# 🔗 PUT YOUR REAL RAW LINK HERE
GITHUB_RAW_URL = "https://github.com/AriellaSansita/SmartCharging-Analytics/blob/main/detailed_ev_charging_stations.csv"

# Optional upload backup
uploaded_file = st.file_uploader("Upload CSV (optional override)", type=["csv"])

@st.cache_data
def load_data():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv(GITHUB_RAW_URL)

def preprocess_data(df):
    df = df.copy()

    # Missing values
    if 'Reviews (Rating)' in df.columns:
        df['Reviews (Rating)'] = df['Reviews (Rating)'].fillna(df['Reviews (Rating)'].median())

    if 'Renewable Energy Source' in df.columns:
        df['Renewable Energy Source'] = df['Renewable Energy Source'].fillna(
            df['Renewable Energy Source'].mode()[0]
        )

    # Remove duplicates
    if 'Station ID' in df.columns:
        df = df.drop_duplicates(subset=['Station ID'])

    # Encode categorical
    le = LabelEncoder()
    cat_cols = ['Charger Type', 'Station Operator', 'Renewable Energy Source']
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Normalize numeric
    scaler = MinMaxScaler()
    num_cols = [
        'Cost (USD/kWh)',
        'Usage Stats (avg users/day)',
        'Charging Capacity (kW)',
        'Distance to City (km)'
    ]

    existing_cols = [c for c in num_cols if c in df.columns]
    if existing_cols:
        df[existing_cols] = scaler.fit_transform(df[existing_cols])

    return df

# --- MAIN ---
try:
    df_raw = load_data()
    st.success("Data loaded successfully from GitHub (or upload)")

    if st.checkbox("Show Raw Data"):
        st.write(df_raw.head())

    df = preprocess_data(df_raw)

    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    # Graph 1
    with col1:
        if 'Usage Stats (avg users/day)' in df_raw.columns:
            st.subheader("Usage Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df_raw['Usage Stats (avg users/day)'], kde=True, ax=ax)
            st.pyplot(fig)

    # Graph 2
    with col2:
        if 'Station Operator' in df_raw.columns and 'Cost (USD/kWh)' in df_raw.columns:
            st.subheader("Cost by Station Operator")
            fig, ax = plt.subplots()
            sns.boxplot(
                x=df_raw['Station Operator'],
                y=df_raw['Cost (USD/kWh)'],
                ax=ax
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    if st.checkbox("Show Processed Data"):
        st.write(df.head())

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure your GitHub link is RAW and public.")
