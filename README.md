# IDAI1051000470-Ariella-Sansita-M

# Candidate Name - Ariella Sansita M

# Candidate Registration Number - 1000470

# CRS Name: Artificial Intelligence

# Course Name - Data Mining

# School name - Birla Open Minds International School, Kollur

# Summative Assessment

# EV SmartCharging Analytics
---


## 📋 Project Scope

**Scenario:** SmartCharging Analytics — Uncovering EV Behavior Patterns

As part of the SmartEnergy Data Lab team, this project analyses a global dataset of EV charging stations to improve station utilisation and customer experience. The analysis explores charging behaviour across station types, operators, locations, and usage levels to support smarter infrastructure planning.

**Objectives:**
- Understand usage trends by charger type, operator, and geography
- Segment charging stations using K-Means clustering
- Discover hidden feature associations using the Apriori algorithm
- Detect anomalous stations with abnormal costs, usage, or reviews
- Deploy an interactive Streamlit dashboard for stakeholder exploration

**Dataset:** `cleaned_ev_charging_data.csv`  
One row per charging station. Key columns: `Station ID`, `Latitude`, `Longitude`, `Charger Type`, `Cost (USD/kWh)`, `Availability`, `Distance to City (km)`, `Usage Stats (avg users/day)`, `Station Operator`, `Charging Capacity (kW)`, `Installation Year`, `Renewable Energy Source`, `Reviews (Rating)`.

---

## 🧹 Data Preparation & Preprocessing

| Step | Detail |
|------|--------|
| Missing value imputation | `Reviews (Rating)` filled with column median |
| Duplicate removal | Rows deduplicated on `Station ID` |
| Label encoding | `Charger Type`, `Station Operator`, `Renewable Energy Source`, `Availability` converted to numeric |
| Min-Max normalisation | `Cost (USD/kWh)`, `Usage Stats (avg users/day)`, `Charging Capacity (kW)`, `Distance to City (km)`, `Availability_Enc` scaled to [0, 1] |
| Feature engineering | `Location Type` derived: stations ≤ median distance to city = Urban, else Rural |

---

## 📊 EDA & Visualisations

The following charts were produced to understand the dataset before modelling:

| Chart | Purpose |
|-------|---------|
| **Usage Statistics Histogram** | Distribution of average daily users across all stations |
| **Cost Boxplot by Operator** | Spread of pricing across different station operators |
| **Usage Trend by Installation Year** | How average daily usage has changed over time |
| **Demand Heatmap (Charger Type × Availability)** | Which charger/availability combinations attract the most users |
| **Reviews vs Usage Scatter** | Whether higher-rated stations attract more users (with trend line and Pearson r) |
| **Charger Type Popularity Bar** | Average daily users by charger type |
| **Operator Rating vs Usage Bubble Chart** | Operator performance: bubble size = number of stations |
| **Urban vs Rural Demand & Cost** | Comparison of usage and pricing between city-proximate and rural stations |
| **Correlation Heatmap** | Pairwise correlations between all numeric features |

**Key EDA Finding:** DC Fast Chargers consistently show higher average daily usage than AC Level 1/2. Urban stations (closer to city centres) record significantly more daily users than rural ones.

---

## 🤖 Clustering Analysis (K-Means)

**Algorithm:** K-Means with `k-means++` initialisation  
**Features used:** `Cost (USD/kWh)`, `Usage Stats (avg users/day)`, `Charging Capacity (kW)`, `Distance to City (km)`, `Availability_Enc`  
**Optimal k:** Determined using the Elbow Method (WCSS plotted for k = 1–10)

**Cluster Profiles (example at k=3):**

| Cluster | Label | Characteristics |
|---------|-------|----------------|
| 0 | Heavy Users | High capacity, high daily usage, city-proximate |
| 1 | Daily Commuters | Moderate usage, mid-range cost, good availability |
| 2 | Occasional Users | Low usage, higher distance from city, lower capacity |

**Visualisations produced:**
- Scatter plot: Charging Capacity vs Usage, coloured by cluster
- Bar chart: Mean feature values per cluster
- Geographic map: Cluster-coloured scatter plot on a world map (pydeck)

---

## 🔗 Association Rule Mining (Apriori)

**Algorithm:** Apriori (via `mlxtend`)  
**Binary features created:**

| Feature | Definition |
|---------|-----------|
| `High_Usage` | Usage > median daily users |
| `Fast_Charger` | Charging capacity > median kW |
| `High_Cost` | Cost > median USD/kWh |
| `Renewable` | Renewable energy source = Yes |
| `Type_*` | One-hot encoded charger type columns |

**Parameters:** Minimum support = 0.05, minimum lift = 1.0 (adjustable via sliders in app)

**Example Rules Discovered:**
- `Fast_Charger → High_Usage` — Fast chargers tend to attract above-average daily demand
- `Renewable, Fast_Charger → High_Usage` — Renewable fast chargers show the strongest usage lift
- `High_Cost → Low_Usage` — Stations with above-median pricing attract fewer users

**Visualisations produced:**
- Sortable rules table (support, confidence, lift)
- Horizontal bar chart: Top 10 rules by lift
- Network diagram: Nodes = features, edge thickness/colour = lift score

---

## 🔍 Anomaly Detection (IQR Method)

**Method:** Interquartile Range (IQR) — flags values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR

**Anomaly types detected:**

| Type | Description |
|------|-------------|
| Usage outliers | Stations with abnormally high or low daily users |
| Cost outliers | Stations with unusually high or low pricing |
| High cost + low usage | Stations in top 25% cost but bottom 25% usage — likely overpriced or underperforming |
| High cost + low reviews | Stations in top 25% cost but bottom 25% rating — poor value for money |
| Maintenance anomalies | Stations with abnormal maintenance frequency (if column present) |

**Visualisation:** Scatter plot of Cost vs Usage with colour-coded anomaly types overlaid.

---

## 🗺️ Geospatial Analysis

Three map modes available in the app:

| Mode | Description |
|------|-------------|
| Raw Stations | All stations plotted as teal scatter points |
| Demand Heatmap | HeatmapLayer weighted by Usage Stats — highlights high-demand regions |
| Cluster Map | Stations colour-coded by K-Means cluster assignment |

---
## Screenshot

<img width="1402" height="712" alt="Screenshot 2026-03-24 at 6 04 34 PM" src="https://github.com/user-attachments/assets/28af923a-1319-4f3f-bb5e-0f0fbe1d2f97" />
<img width="1351" height="660" alt="Screenshot 2026-03-24 at 6 06 04 PM" src="https://github.com/user-attachments/assets/96d352f8-6ee8-4323-ad7e-fabc2307655f" />
<img width="1334" height="431" alt="Screenshot 2026-03-24 at 6 06 34 PM" src="https://github.com/user-attachments/assets/52b6e572-60cd-4f50-a9b4-55cbd0fc69b1" />
<img width="1366" height="579" alt="Screenshot 2026-03-24 at 6 06 45 PM" src="https://github.com/user-attachments/assets/374189cf-e612-4e94-839c-ceed6d46d428" />
<img width="1337" height="730" alt="Screenshot 2026-03-24 at 6 07 13 PM" src="https://github.com/user-attachments/assets/f53e6bf0-ebcb-4641-b830-bd1e05ea1dc5" />
<img width="1378" height="583" alt="Screenshot 2026-03-24 at 6 07 24 PM" src="https://github.com/user-attachments/assets/a1dbb5d6-1534-4523-be4b-cba52ab79ee0" />



## 💡 Key Findings & Insights

1. **DC Fast Chargers** attract the highest average daily usage — should be prioritised in expansion plans.
2. **Urban stations** outperform rural stations on usage; rural stations tend to charge more per kWh.
3. **High-cost stations with low usage and poor reviews** represent the clearest operational inefficiency — 
   these are candidates for pricing review or service improvement.
4. **Renewable energy + fast charging** is the strongest association pattern — a signal for bundling 
   infrastructure investment.
5. **Cluster analysis** reveals three distinct station archetypes: heavy urban fast-charge hubs, moderate 
   commuter stations, and underutilised rural stations.

---

## 🚀 Streamlit Deployment

**Live App:** https://daychargingcapacitykwdistancetocitykmensurecolumnsexistbefores.streamlit.app/

The app is structured into 7 tabs:

| Tab | Content |
|-----|---------|
| 📋 Overview | Project scope, preprocessing pipeline, dataset preview |
| 📊 EDA | All exploratory charts |
| 🤖 Clustering | Elbow method, scatter, profile bar chart, cluster map |
| 🔗 Assoc. Rules | Apriori table, top-rules bar chart, network diagram |
| 🔍 Anomalies | IQR scatter, metric cards, detailed anomaly tables |
| 💡 Insights | Correlation heatmap, operator bubbles, urban vs rural, key takeaways |
| 🗺️ Map | Three switchable map layers |

---

## 📦 Dependencies

See `requirements.txt`. Key libraries:

| Library | Purpose |
|---------|---------|
| `streamlit` | Dashboard framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | K-Means clustering, MinMaxScaler |
| `mlxtend` | Apriori and association rules |
| `pydeck` | Geospatial map layers |
| `networkx` | Association rule network diagram |

---

## 📚 References

- https://dicecamp.com/insights/association-mining-rules-combined-with-clustering/
- https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html
- https://scikit-learn.org/stable/
- https://likuyani.cdf.go.ke/uploaded-files/5P8049/HomePages/PythonForGeospatialDataAnalysis.pdf
- https://365datascience.com/tutorials/python-tutorials/k-means-clustering/
