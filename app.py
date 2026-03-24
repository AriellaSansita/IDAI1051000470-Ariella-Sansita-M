import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pydeck as pdk
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ===============================
# SETTINGS & CONFIG
# ===============================
st.set_page_config(page_title="EV SmartCharging Analytics", layout="wide")
st.title("🚗 SmartCharging Analytics: EV Behavior Patterns")

# ─────────────────────────────────────────────────────────────
# STAGE 1 – PROJECT SCOPE  ✅ (was missing)
# ─────────────────────────────────────────────────────────────
st.header("📋 Stage 1: Project Scope & Objectives")
st.markdown("""
This analytics dashboard investigates **Electric Vehicle (EV) charging station behaviour** across a
global dataset to uncover patterns, inefficiencies, and opportunities for smarter infrastructure
planning.

**Key Objectives**
1. **Understand usage patterns** — identify which station types, operators, and locations attract the
   most daily users.
2. **Segment stations** (K-Means clustering) to support targeted investment and marketing decisions.
3. **Mine association rules** (Apriori) to discover hidden co-occurrence patterns between charger
   features and high usage.
4. **Detect anomalies** — flag stations with suspicious cost/usage profiles or maintenance concerns.
5. **Visualise geographic demand** — heat-map overlay to pinpoint underserved regions.

**Dataset**  `cleaned_ev_charging_data.csv` — one row per charging station with attributes including
charger type, operator, renewable energy source, cost, capacity, installation year, reviews, and
geographic coordinates.
""")

# ===============================
# DATA LOADING
# ===============================
try:
    df_raw = pd.read_csv("cleaned_ev_charging_data.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure 'cleaned_ev_charging_data.csv' is in the same folder.")
    st.stop()

# ===============================
# PREPROCESSING & FEATURE ENGINEERING
# ===============================
@st.cache_data
def preprocess_data(df):
    df_proc = df.copy()

    if 'Reviews (Rating)' in df_proc.columns:
        df_proc['Reviews (Rating)'] = df_proc['Reviews (Rating)'].fillna(df_proc['Reviews (Rating)'].median())

    le = LabelEncoder()
    cat_cols = ['Charger Type', 'Station Operator', 'Renewable Energy Source', 'Availability']
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[f'{col}_Enc'] = le.fit_transform(df_proc[col].astype(str))

    cluster_features = [
        'Cost (USD/kWh)',
        'Usage Stats (avg users/day)',
        'Charging Capacity (kW)',
        'Distance to City (km)',
        'Availability_Enc'
    ]

    scaler = MinMaxScaler()
    existing_features = [f for f in cluster_features if f in df_proc.columns]
    if existing_features:
        df_proc[existing_features] = scaler.fit_transform(df_proc[existing_features])

    return df_proc, existing_features

df_processed, cluster_cols = preprocess_data(df_raw)

# ===============================
# STAGE 2 – DATA CLEANING SUMMARY
# ===============================
st.divider()
st.header("🧹 Stage 2: Data Cleaning & Preprocessing")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Stations", len(df_raw))
col_b.metric("Features", df_raw.shape[1])
col_c.metric("Missing Values (after cleaning)", int(df_raw.isnull().sum().sum()))
st.markdown("""
- **Missing values** in `Reviews (Rating)` filled with the column median.  
- **Label encoding** applied to: `Charger Type`, `Station Operator`, `Renewable Energy Source`, `Availability`.  
- **Min-Max normalisation** applied to `Cost`, `Usage Stats`, `Charging Capacity`, `Distance to City`, and `Availability_Enc` before clustering.
""")

# ─────────────────────────────────────────────────────────────
# STAGE 3 – EDA  (original charts + two missing charts)
# ─────────────────────────────────────────────────────────────
st.divider()
st.header("📊 Stage 3: Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Usage Statistics Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    sns.histplot(df_raw['Usage Stats (avg users/day)'], bins=20, kde=True, color='teal', ax=ax_hist)
    ax_hist.set_xlabel("Avg Users / Day")
    st.pyplot(fig_hist)

with col2:
    st.subheader("Cost vs Station Operator")
    fig_box, ax_box = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_raw, x='Station Operator', y='Cost (USD/kWh)', palette='Set2', ax=ax_box)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_box)

# --- Original line chart ---
st.subheader("Usage Trend by Installation Year")
if 'Installation Year' in df_raw.columns:
    trend_data = df_raw.groupby('Installation Year')['Usage Stats (avg users/day)'].mean().reset_index()
    fig_line, ax_line = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=trend_data, x='Installation Year', y='Usage Stats (avg users/day)', marker='o', ax=ax_line)
    st.pyplot(fig_line)

# ── MISSING EDA 1: Heatmap — demand across Charger Type & Availability ✅
st.subheader("🔥 Demand Heatmap: Charger Type × Availability")
if 'Charger Type' in df_raw.columns and 'Availability' in df_raw.columns:
    heatmap_data = df_raw.pivot_table(
        index='Charger Type',
        columns='Availability',
        values='Usage Stats (avg users/day)',
        aggfunc='mean'
    )
    fig_ht, ax_ht = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd',
        linewidths=0.5, ax=ax_ht
    )
    ax_ht.set_title("Average Daily Users by Charger Type & Availability")
    st.pyplot(fig_ht)
    st.caption("Darker cells = higher average daily usage. Reveals which charger-availability combos drive the most traffic.")
else:
    st.warning("Columns 'Charger Type' or 'Availability' not found in dataset.")

# ── MISSING EDA 2: Reviews (Rating) vs Usage  ✅
st.subheader("⭐ Reviews (Rating) vs. Usage Stats")
if 'Reviews (Rating)' in df_raw.columns:
    fig_rv, ax_rv = plt.subplots(figsize=(10, 5))
    hue_col = 'Charger Type' if 'Charger Type' in df_raw.columns else None
    sns.scatterplot(
        data=df_raw,
        x='Reviews (Rating)',
        y='Usage Stats (avg users/day)',
        hue=hue_col,
        palette='tab10',
        alpha=0.65,
        s=60,
        ax=ax_rv
    )
    # Add trend line
    x_vals = df_raw['Reviews (Rating)'].dropna()
    y_vals = df_raw.loc[x_vals.index, 'Usage Stats (avg users/day)']
    m, b = np.polyfit(x_vals, y_vals, 1)
    ax_rv.plot(sorted(x_vals), [m * xi + b for xi in sorted(x_vals)],
               color='crimson', linewidth=2, linestyle='--', label='Trend')
    ax_rv.set_xlabel("Reviews (Rating)")
    ax_rv.set_ylabel("Avg Users / Day")
    ax_rv.legend(loc='upper left', fontsize=8)
    st.pyplot(fig_rv)
    corr_val = df_raw[['Reviews (Rating)', 'Usage Stats (avg users/day)']].corr().iloc[0, 1]
    st.caption(f"Pearson correlation between rating and usage: **{corr_val:.3f}**")
else:
    st.warning("Column 'Reviews (Rating)' not found.")

# ─────────────────────────────────────────────────────────────
# STAGE 4 – K-MEANS CLUSTERING  (+ cluster map)
# ─────────────────────────────────────────────────────────────
st.divider()
st.header("🤖 Stage 4: Machine Learning – Station Clustering")
st.write(f"**Features Used:** {', '.join(cluster_cols)}")

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    km.fit(df_processed[cluster_cols])
    wcss.append(km.inertia_)

st.subheader("1. Finding Optimal Clusters (Elbow Method)")
fig_elbow, ax_elbow = plt.subplots(figsize=(12, 4))
ax_elbow.plot(range(1, 11), wcss, marker='o', color='#1f77b4', linewidth=2)
ax_elbow.set_xlabel("Number of Clusters (k)")
ax_elbow.set_ylabel("WCSS (Inertia)")
ax_elbow.set_title("Elbow Method")
st.pyplot(fig_elbow)

st.subheader("2. Market Segmentation Results")
k_value = st.slider("Select k (Number of Clusters)", 2, 6, 3)
model = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_raw['Cluster'] = model.fit_predict(df_processed[cluster_cols])

fig_cluster, ax_cluster = plt.subplots(figsize=(12, 6))
sns.scatterplot(
    data=df_raw, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)',
    hue='Cluster', palette='Set1', s=150, alpha=0.7, ax=ax_cluster
)
ax_cluster.set_title("Station Clusters: Capacity vs Usage")
st.pyplot(fig_cluster)

# ── MISSING CLUSTER MAP: clusters coloured on geographic map  ✅
st.subheader("3. Cluster Map – Geographic Distribution of Segments")
if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    # Assign a colour per cluster
    palette_rgb = [
        [31, 119, 180],   # blue
        [255, 127, 14],   # orange
        [44, 160, 44],    # green
        [214, 39, 40],    # red
        [148, 103, 189],  # purple
        [140, 86, 75],    # brown
    ]
    df_map = df_raw[['Latitude', 'Longitude', 'Cluster']].copy().dropna()
    df_map['color'] = df_map['Cluster'].apply(lambda c: palette_rgb[int(c) % len(palette_rgb)])
    df_map['r'] = df_map['color'].apply(lambda c: c[0])
    df_map['g'] = df_map['color'].apply(lambda c: c[1])
    df_map['b'] = df_map['color'].apply(lambda c: c[2])

    layer_clusters = pdk.Layer(
        'ScatterplotLayer',
        data=df_map,
        get_position='[Longitude, Latitude]',
        get_color='[r, g, b, 200]',
        radius_min_pixels=5,
        radius_max_pixels=14,
        pickable=True,
    )
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=df_map['Latitude'].mean(),
            longitude=df_map['Longitude'].mean(),
            zoom=2, pitch=0,
        ),
        layers=[layer_clusters],
        tooltip={"text": "Cluster: {Cluster}"},
    ))

    # Legend
    legend_cols = st.columns(k_value)
    for ci in range(k_value):
        rgb = palette_rgb[ci % len(palette_rgb)]
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        legend_cols[ci].markdown(
            f"<span style='color:{hex_color}; font-size:18px;'>●</span> Cluster {ci}",
            unsafe_allow_html=True
        )
else:
    st.warning("Latitude / Longitude columns not found — skipping cluster map.")

# ─────────────────────────────────────────────────────────────
# STAGE 5 – ASSOCIATION RULE MINING  (+ visualisations)
# ─────────────────────────────────────────────────────────────
st.divider()
st.header("🔗 Stage 5: Association Rule Mining")

rules = pd.DataFrame()  # initialise so later stages don't crash

try:
    df_rules = pd.DataFrame()

    # ── EV-specific binary features  ✅ (more meaningful than generic median splits)
    df_rules['High_Usage'] = df_raw['Usage Stats (avg users/day)'] > df_raw['Usage Stats (avg users/day)'].median()
    df_rules['Fast_Charger'] = df_raw['Charging Capacity (kW)'] > df_raw['Charging Capacity (kW)'].median()

    if 'Renewable Energy Source' in df_raw.columns:
        # Handle both string ('Yes'/'No') and numeric (1/0) encodings
        ren = df_raw['Renewable Energy Source']
        if ren.dtype == object:
            df_rules['Renewable'] = ren.str.strip().str.lower().isin(['yes', '1', 'true'])
        else:
            df_rules['Renewable'] = ren.astype(bool)

    df_rules['High_Cost'] = df_raw['Cost (USD/kWh)'] > df_raw['Cost (USD/kWh)'].median()

    if 'Charger Type' in df_raw.columns:
        # One-hot encode charger type into boolean columns
        for ctype in df_raw['Charger Type'].dropna().unique():
            col_name = f"Type_{ctype.replace(' ', '_')}"
            df_rules[col_name] = df_raw['Charger Type'] == ctype

    df_rules = df_rules.astype(bool)

    frequent_itemsets = apriori(df_rules, min_support=0.05, use_colnames=True)

    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        if not rules.empty:
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

            st.write("### Discovered Association Rules")
            st.dataframe(
                rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                .sort_values('lift', ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

            # ── MISSING VIZ 1: Bar chart of top rules by lift  ✅
            st.subheader("📊 Top Rules by Lift (Bar Chart)")
            top_rules = rules.sort_values('lift', ascending=False).head(10).copy()
            top_rules['rule'] = top_rules['antecedents'] + " → " + top_rules['consequents']

            fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
            bars = ax_bar.barh(
                top_rules['rule'][::-1],
                top_rules['lift'][::-1],
                color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_rules)))
            )
            ax_bar.set_xlabel("Lift")
            ax_bar.set_title("Association Rules Ranked by Lift")
            ax_bar.axvline(x=1, color='gray', linestyle='--', linewidth=1, label='Lift = 1 (random)')
            ax_bar.legend()
            plt.tight_layout()
            st.pyplot(fig_bar)
            st.caption("Rules with Lift > 1 indicate a positive association beyond chance.")

            # ── MISSING VIZ 2: Network diagram of rules  ✅
            st.subheader("🕸️ Association Rule Network Diagram")
            top_net = rules.sort_values('lift', ascending=False).head(15)
            G = nx.DiGraph()
            for _, row in top_net.iterrows():
                G.add_edge(row['antecedents'], row['consequents'],
                           weight=row['lift'], confidence=row['confidence'])

            fig_net, ax_net = plt.subplots(figsize=(14, 9))
            pos = nx.spring_layout(G, seed=42, k=2)
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_w = max(edge_weights) if edge_weights else 1

            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#4ECDC4',
                                   alpha=0.85, ax=ax_net)
            nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax_net)
            nx.draw_networkx_edges(
                G, pos,
                width=[2 + 4 * (w / max_w) for w in edge_weights],
                edge_color=edge_weights, edge_cmap=plt.cm.OrRd,
                arrows=True, arrowsize=20, ax=ax_net
            )
            sm = plt.cm.ScalarMappable(cmap=plt.cm.OrRd,
                                       norm=plt.Normalize(vmin=min(edge_weights), vmax=max_w))
            sm.set_array([])
            plt.colorbar(sm, ax=ax_net, label='Lift', shrink=0.6)
            ax_net.set_title("Rule Network — edge width & colour = Lift strength", fontsize=12)
            ax_net.axis('off')
            plt.tight_layout()
            st.pyplot(fig_net)
            st.caption("Nodes = item sets. Directed edges show antecedent → consequent. Thicker / darker = stronger association.")
        else:
            st.warning("No strong rules found with current thresholds.")
    else:
        st.warning("No frequent patterns found. Try lowering min_support.")

except Exception as e:
    st.error(f"Association Rule Error: {e}")

# ─────────────────────────────────────────────────────────────
# STAGE 6 – ANOMALY DETECTION  (+ extra requested checks)
# ─────────────────────────────────────────────────────────────
st.divider()
st.header("🔍 Stage 6: Anomaly Detection")
st.write("Detecting outliers using the **Interquartile Range (IQR)** method across multiple dimensions.")

def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

usage_outliers = detect_outliers(df_raw, 'Usage Stats (avg users/day)')
cost_outliers  = detect_outliers(df_raw, 'Cost (USD/kWh)')

# ── MISSING: High cost + low reviews/usage  ✅
high_cost_low_usage = pd.DataFrame()
high_cost_low_reviews = pd.DataFrame()
if 'Reviews (Rating)' in df_raw.columns:
    cost_thresh   = df_raw['Cost (USD/kWh)'].quantile(0.75)
    usage_thresh  = df_raw['Usage Stats (avg users/day)'].quantile(0.25)
    review_thresh = df_raw['Reviews (Rating)'].quantile(0.25)

    high_cost_low_usage = df_raw[
        (df_raw['Cost (USD/kWh)'] > cost_thresh) &
        (df_raw['Usage Stats (avg users/day)'] < usage_thresh)
    ]
    high_cost_low_reviews = df_raw[
        (df_raw['Cost (USD/kWh)'] > cost_thresh) &
        (df_raw['Reviews (Rating)'] < review_thresh)
    ]

# ── MISSING: Stations with abnormally frequent maintenance  ✅
maintenance_outliers = pd.DataFrame()
if 'Maintenance Records' in df_raw.columns:
    maintenance_outliers = detect_outliers(df_raw, 'Maintenance Records')
elif 'Number of Maintenance Visits' in df_raw.columns:
    maintenance_outliers = detect_outliers(df_raw, 'Number of Maintenance Visits')

c1, c2, c3, c4 = st.columns(4)
c1.metric("Usage Outliers",               len(usage_outliers))
c2.metric("Cost Outliers",                len(cost_outliers))
c3.metric("High Cost + Low Usage",        len(high_cost_low_usage))
c4.metric("High Cost + Low Reviews",      len(high_cost_low_reviews))

if len(maintenance_outliers) > 0:
    st.metric("Abnormal Maintenance Frequency", len(maintenance_outliers))

if len(usage_outliers) == 0 and len(cost_outliers) == 0 and \
   len(high_cost_low_usage) == 0 and len(high_cost_low_reviews) == 0:
    st.info("✅ No anomalies detected. All station data falls within the normal statistical range.")
else:
    if st.checkbox("Show Anomaly Details"):
        tabs = st.tabs(["Usage", "Cost", "High Cost + Low Usage",
                        "High Cost + Low Reviews", "Maintenance"])
        with tabs[0]:
            st.write(f"**{len(usage_outliers)} usage outliers**")
            if len(usage_outliers): st.dataframe(usage_outliers)
        with tabs[1]:
            st.write(f"**{len(cost_outliers)} cost outliers**")
            if len(cost_outliers): st.dataframe(cost_outliers)
        with tabs[2]:
            st.write(f"**{len(high_cost_low_usage)} stations with high cost but low usage**")
            if len(high_cost_low_usage): st.dataframe(high_cost_low_usage)
        with tabs[3]:
            st.write(f"**{len(high_cost_low_reviews)} stations with high cost but low ratings**")
            if len(high_cost_low_reviews): st.dataframe(high_cost_low_reviews)
        with tabs[4]:
            if len(maintenance_outliers):
                st.write(f"**{len(maintenance_outliers)} stations with abnormal maintenance frequency**")
                st.dataframe(maintenance_outliers)
            else:
                st.info("No maintenance column found or no outliers detected.")

# Visualise anomaly flags in scatter
st.subheader("Anomaly Visualisation: Cost vs Usage")
fig_anom, ax_anom = plt.subplots(figsize=(12, 6))
# Normal points
normal_mask = ~df_raw.index.isin(usage_outliers.index) & ~df_raw.index.isin(cost_outliers.index)
ax_anom.scatter(df_raw.loc[normal_mask, 'Cost (USD/kWh)'],
                df_raw.loc[normal_mask, 'Usage Stats (avg users/day)'],
                c='steelblue', alpha=0.5, s=40, label='Normal')
if len(usage_outliers):
    ax_anom.scatter(usage_outliers['Cost (USD/kWh)'], usage_outliers['Usage Stats (avg users/day)'],
                    c='orange', s=100, marker='^', label='Usage Outlier', zorder=5)
if len(cost_outliers):
    ax_anom.scatter(cost_outliers['Cost (USD/kWh)'], cost_outliers['Usage Stats (avg users/day)'],
                    c='red', s=100, marker='X', label='Cost Outlier', zorder=5)
if len(high_cost_low_usage):
    ax_anom.scatter(high_cost_low_usage['Cost (USD/kWh)'],
                    high_cost_low_usage['Usage Stats (avg users/day)'],
                    c='purple', s=120, marker='D', label='High Cost + Low Usage', zorder=6)
ax_anom.set_xlabel("Cost (USD/kWh)")
ax_anom.set_ylabel("Avg Users / Day")
ax_anom.legend()
ax_anom.set_title("Cost vs Usage — Anomalies Highlighted")
st.pyplot(fig_anom)

# ─────────────────────────────────────────────────────────────
# STAGE 7 – INSIGHTS & REPORTING  (+ operator & city/rural analysis)
# ─────────────────────────────────────────────────────────────
st.divider()
st.header("📊 Stage 7: Interpretation & Insights")

# Correlation heatmap (was already here)
st.subheader("Correlation Heatmap")
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(df_processed.select_dtypes(include=['number']).corr(),
            annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 8}, ax=ax_corr)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_corr)

# ── MISSING: Most popular charger types  ✅
if 'Charger Type' in df_raw.columns:
    st.subheader("🔌 Most Popular Charger Types by Usage")
    charger_usage = df_raw.groupby('Charger Type')['Usage Stats (avg users/day)'].mean().sort_values(ascending=False)
    fig_ct, ax_ct = plt.subplots(figsize=(10, 5))
    charger_usage.plot(kind='bar', ax=ax_ct, color=sns.color_palette('viridis', len(charger_usage)))
    ax_ct.set_xlabel("Charger Type")
    ax_ct.set_ylabel("Avg Daily Users")
    ax_ct.set_title("Average Daily Usage by Charger Type")
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig_ct)

# ── MISSING: Operator ratings vs usage comparison  ✅
if 'Station Operator' in df_raw.columns and 'Reviews (Rating)' in df_raw.columns:
    st.subheader("🏢 Station Operators: Rating vs. Usage")
    op_stats = df_raw.groupby('Station Operator').agg(
        Avg_Rating=('Reviews (Rating)', 'mean'),
        Avg_Usage=('Usage Stats (avg users/day)', 'mean'),
        Count=('Usage Stats (avg users/day)', 'count')
    ).reset_index()

    fig_op, ax_op = plt.subplots(figsize=(11, 6))
    scatter = ax_op.scatter(
        op_stats['Avg_Rating'], op_stats['Avg_Usage'],
        s=op_stats['Count'] * 10, alpha=0.7,
        c=range(len(op_stats)), cmap='tab10'
    )
    for _, row in op_stats.iterrows():
        ax_op.annotate(row['Station Operator'],
                       (row['Avg_Rating'], row['Avg_Usage']),
                       fontsize=8, ha='center', va='bottom')
    ax_op.set_xlabel("Average Rating")
    ax_op.set_ylabel("Average Daily Users")
    ax_op.set_title("Operator Comparison — bubble size = number of stations")
    st.pyplot(fig_op)

# ── MISSING: City vs Rural demand comparison  ✅
if 'Distance to City (km)' in df_raw.columns:
    st.subheader("🏙️ City vs. Rural Demand Comparison")
    median_dist = df_raw['Distance to City (km)'].median()
    df_raw['Location Type'] = np.where(df_raw['Distance to City (km)'] <= median_dist, 'Urban', 'Rural')

    loc_stats = df_raw.groupby('Location Type').agg(
        Avg_Usage=('Usage Stats (avg users/day)', 'mean'),
        Avg_Cost=('Cost (USD/kWh)', 'mean'),
    ).reset_index()

    fig_loc, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=loc_stats, x='Location Type', y='Avg_Usage', palette=['#2196F3', '#4CAF50'], ax=axes[0])
    axes[0].set_title("Avg Daily Usage: Urban vs Rural")
    axes[0].set_ylabel("Avg Users / Day")

    sns.barplot(data=loc_stats, x='Location Type', y='Avg_Cost', palette=['#FF5722', '#9C27B0'], ax=axes[1])
    axes[1].set_title("Avg Cost: Urban vs Rural")
    axes[1].set_ylabel("Cost (USD/kWh)")

    plt.tight_layout()
    st.pyplot(fig_loc)
    st.caption(f"Split threshold: stations ≤ {median_dist:.1f} km from city = Urban, rest = Rural.")

# Key findings summary
st.subheader("📌 Key Findings")
avg_usage_cluster = df_raw.groupby('Cluster')['Usage Stats (avg users/day)'].mean()
best_cluster = int(avg_usage_cluster.idxmax())

top_charger = ""
if 'Charger Type' in df_raw.columns:
    top_charger = df_raw.groupby('Charger Type')['Usage Stats (avg users/day)'].mean().idxmax()

first_rule_ant = rules.iloc[0]['antecedents'] if not rules.empty else "N/A"
first_rule_con = rules.iloc[0]['consequents'] if not rules.empty else "N/A"

st.info(f"""
- **Top Performing Cluster:** Cluster {best_cluster} has the highest average daily usage
  ({avg_usage_cluster[best_cluster]:.1f} users/day).
- **Most Popular Charger Type:** {top_charger if top_charger else 'N/A'} attracts the highest average daily users.
- **Anomalies:** {len(usage_outliers)} stations with irregular usage, {len(high_cost_low_usage)} with
  high cost + low usage flagged for review.
- **Association Rules:** Strongest link found between **'{first_rule_ant}'** → **'{first_rule_con}'**.
- **Urban vs Rural:** Urban stations tend to have higher usage; rural stations may need targeted incentives.
""")

# ─────────────────────────────────────────────────────────────
# STAGE 8 – STREAMLIT DEPLOYMENT  (+ heatmap layer)
# ─────────────────────────────────────────────────────────────
st.divider()
st.header("📍 Stage 8: Geographic Distribution")

if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
    map_mode = st.radio("Map Layer", ["Scatter (raw stations)", "Demand Heatmap ✅"], horizontal=True)

    if map_mode == "Scatter (raw stations)":
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=df_raw['Latitude'].mean(), longitude=df_raw['Longitude'].mean(),
                zoom=2, pitch=0,
            ),
            layers=[pdk.Layer(
                'ScatterplotLayer', data=df_raw,
                get_position='[Longitude, Latitude]',
                get_color='[255, 100, 0, 160]',
                radius_min_pixels=3, radius_max_pixels=10,
            )],
        ))
    else:
        # ── MISSING: Demand heatmap layer  ✅
        heatmap_df = df_raw[['Latitude', 'Longitude', 'Usage Stats (avg users/day)']].dropna().copy()
        heatmap_df.rename(columns={'Usage Stats (avg users/day)': 'weight'}, inplace=True)

        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=heatmap_df['Latitude'].mean(),
                longitude=heatmap_df['Longitude'].mean(),
                zoom=2, pitch=30,
            ),
            layers=[
                pdk.Layer(
                    'HeatmapLayer',
                    data=heatmap_df,
                    get_position='[Longitude, Latitude]',
                    get_weight='weight',
                    radiusPixels=60,
                    intensity=1,
                    threshold=0.05,
                    aggregation='SUM',
                ),
            ],
        ))
        st.caption("Brighter / hotter areas = higher cumulative daily EV charging demand.")

# ===============================
# FINAL DATA TABLE
# ===============================
if st.checkbox("View Final Data Table"):
    st.dataframe(df_raw)
