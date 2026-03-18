import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# --- CONFIGURATION ---
# Replace 'YourUsername' and 'YourRepo' with your actual GitHub details
GITHUB_RAW_URL = "https://raw.githubusercontent.com/YourUsername/YourRepo/main/detailed_ev_charging_stations.csv"

st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

@st.cache_data
def load_and_preprocess_data(url):
    # Load the dataset from GitHub
    df = pd.read_csv(url)
    
    # --- STAGE 2: DATA CLEANING & PREPROCESSING --- 
    # 1. Handle missing values
    if 'Reviews (Rating)' in df.columns:
        df['Reviews (Rating)'] = df['Reviews (Rating)'].fillna(df['Reviews (Rating)'].median()) [cite: 2]
    
    if 'Renewable Energy Source' in df.columns:
        df['Renewable Energy Source'] = df['Renewable Energy Source'].fillna(df['Renewable Energy Source'].mode()[0]) [cite: 2]

    # 2. Remove duplicates 
    if 'Station ID' in df.columns:
        df = df.drop_duplicates(subset=['Station ID'], keep='first')

    # 3. Normalize continuous variables 
    scaler = MinMaxScaler()
    cont_vars = ['Cost (USD/kWh)', 'Usage Stats (avg users/day)', 'Charging Capacity (kW)', 'Distance to City (km)']
    existing_cont = [v for v in cont_vars if v in df.columns]
    df[existing_cont] = scaler.fit_transform(df[existing_cont])

    # 4. Encode categorical features 
    le = LabelEncoder()
    cat_vars = ['Charger Type', 'Station Operator', 'Renewable Energy Source']
    for col in cat_vars:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

# Main execution
try:
    df = load_and_preprocess_data(GITHUB_RAW_URL)
    st.success("Dataset loaded and preprocessed directly from GitHub!")

    # --- STAGE 3: EXPLORATORY DATA ANALYSIS (EDA) --- 
    st.header("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Station Usage")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Usage Stats (avg users/day)'], kde=True, ax=ax1, color='teal')
        st.pyplot(fig1) [cite: 2]

    with col2:
        if 'Station Operator' in df.columns:
            st.subheader("Cost Distribution by Operator")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='Station Operator', y='Cost (USD/kWh)', data=df, ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2) [cite: 2]

    # Show raw data preview 
    if st.checkbox("Show Cleaned & Normalized Data Preview"):
        st.write(df.head(10))

except Exception as e:
    st.error(f"Error: Could not retrieve data from GitHub. {e}")
    st.info("Check if your GitHub link is the 'Raw' version and the filename is correct.")
