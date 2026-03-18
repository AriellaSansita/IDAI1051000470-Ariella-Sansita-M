import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. Load the dataset
# Replace 'ev_charging_data.csv' with your actual file path
df = pd.read_csv('ev_charging_data.csv')

# 2. Handle Missing Values 
# Fill numerical ratings with the median to avoid outlier distortion
if 'Reviews (Rating)' in df.columns:
    df['Reviews (Rating)'] = df['Reviews (Rating)'].fillna(df['Reviews (Rating)'].median())

# Fill categorical missing values with 'Unknown' or the most frequent value
cols_to_fix = ['Renewable Energy Source', 'Connector Types']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col].fillna(df['Renewable Energy Source'].mode()[0])

# 3. Remove Duplicates 
# Using 'Station ID' as the unique identifier
df.drop_duplicates(subset=['Station ID'], keep='first', inplace=True)

# 4. Normalize Continuous Variables 
# Scales values to a range between 0 and 1 for clustering accuracy
scaler = MinMaxScaler()
continuous_vars = [
    'Cost (USD/kWh)', 
    'Usage Stats (avg users/day)', 
    'Charging Capacity (kW)', 
    'Distance to City (km)'
]

# Ensure columns exist before scaling
existing_cont_vars = [col for col in continuous_vars if col in df.columns]
df[existing_cont_vars] = scaler.fit_transform(df[existing_cont_vars])

# 5. Encode Categorical Features 
# Converts text labels into numbers for machine learning models
le = LabelEncoder()
categorical_vars = ['Charger Type', 'Station Operator', 'Renewable Energy Source']

for col in categorical_vars:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Display the cleaned data preview
print("Data Cleaning Complete. Scaled and Encoded Head:")
print(df.head())

# Save the cleaned data for the next stage (EDA)
# df.to_csv('cleaned_ev_charging_data.csv', index=False)
