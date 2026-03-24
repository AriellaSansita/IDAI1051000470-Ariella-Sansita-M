# ══════════════════════════════════════════════════════════════════════════════
#  EV SmartCharging Analytics  —  Full Streamlit App
#  Tabs: Overview | EDA | Clustering | Association Rules | Anomalies | Insights | Map
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules as arm_rules

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV SmartCharging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b0f1a; color: #e2e8f0; }

.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #0d3d4a 50%, #113a2e 100%);
    border: 1px solid #1e4d5a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "⚡";
    position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 9rem; opacity: 0.06;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.3rem; font-weight: 700;
    color: #38f5c8; margin: 0 0 0.4rem 0; letter-spacing: -1px;
}
.hero-sub { font-size: 1rem; color: #94a3b8; margin: 0; font-weight: 300; }

.metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
.metric-card {
    background: #111827; border: 1px solid #1e3a4a;
    border-radius: 12px; padding: 1.1rem 1.5rem;
    flex: 1; min-width: 150px; transition: border-color .2s;
}
.metric-card:hover { border-color: #38f5c8; }
.metric-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1.5px; color: #64748b; margin-bottom: 4px; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 1.9rem; font-weight: 700; color: #38f5c8; }
.metric-sub   { font-size: 0.72rem; color: #475569; margin-top: 2px; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; letter-spacing: 3px; text-transform: uppercase;
    color: #38f5c8; margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3a4a;
}

.finding-box {
    background: #0f1f2e; border-left: 3px solid #38f5c8;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.3rem; margin: 0.5rem 0;
    font-size: 0.9rem; color: #cbd5e1;
}
.finding-box strong { color: #38f5c8; }

.stTabs [data-baseweb="tab-list"] {
    background: #0f1622; border-radius: 12px;
    padding: 4px; gap: 2px; border: 1px solid #1e3a4a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px;
    color: #64748b; font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 0.85rem;
    padding: 0.45rem 1rem; border: none; transition: all .2s;
}
.stTabs [aria-selected="true"] {
    background: #0d3d4a !important; color: #38f5c8 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
TEAL    = "#38f5c8"
PALETTE = ["#38f5c8", "#f97316", "#818cf8", "#f43f5e", "#facc15", "#34d399"]

plt.rcParams.update({
    "figure.facecolor": "#111827", "axes.facecolor":  "#111827",
    "axes.edgecolor":   "#1e3a4a", "axes.labelcolor": "#94a3b8",
    "axes.titlecolor":  "#e2e8f0", "axes.titlesize":  13,
    "axes.labelsize":   11,        "xtick.color":     "#64748b",
    "ytick.color":      "#64748b", "text.color":      "#e2e8f0",
    "grid.color":       "#1e293b", "grid.linestyle":  "--",
    "grid.linewidth":   0.6,       "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#334155", "legend.fontsize":  9,
    "font.family":      "monospace",
})
sns.set_palette(PALETTE)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process():
    try:
        df = pd.read_csv("cleaned_ev_charging_data.csv")
    except FileNotFoundError:
        return None, None, None

    d = df.copy()

    # Fill missing reviews
    if "Reviews (Rating)" in d.columns:
        d["Reviews (Rating)"] = d["Reviews (Rating)"].fillna(d["Reviews (Rating)"].median())

    # Label encode categoricals
    le = LabelEncoder()
    for col in ["Charger Type", "Station Operator", "Renewable Energy Source", "Availability"]:
        if col in d.columns:
            d[f"{col}_Enc"] = le.fit_transform(d[col].astype(str))

    # Normalise for clustering
    feats = ["Cost (USD/kWh)", "Usage Stats (avg users/day)",
             "Charging Capacity (kW)", "Distance to City (km)", "Availability_Enc"]
    exist = [f for f in feats if f in d.columns]
    if exist:
        d[exist] = MinMaxScaler().fit_transform(d[exist])

    return df, d, exist


df_raw, df_proc, cluster_cols = load_and_process()

if df_raw is None:
    st.error("❌ `cleaned_ev_charging_data.csv` not found. Place it in the same folder as this script.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">⚡ EV SmartCharging Analytics</p>
  <p class="hero-sub">Data Mining · K-Means Clustering · Apriori Rules · Anomaly Detection · Geospatial Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Top KPI row
avg_usage = df_raw["Usage Stats (avg users/day)"].mean() if "Usage Stats (avg users/day)" in df_raw.columns else 0
avg_cost  = df_raw["Cost (USD/kWh)"].mean()              if "Cost (USD/kWh)" in df_raw.columns else 0
n_missing = int(df_raw.isnull().sum().sum())

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Total Stations</div>
    <div class="metric-value">{len(df_raw):,}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Features</div>
    <div class="metric-value">{df_raw.shape[1]}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Missing Values</div>
    <div class="metric-value">{n_missing}</div>
    <div class="metric-sub">after cleaning</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Avg Daily Users</div>
    <div class="metric-value">{avg_usage:.1f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Avg Cost / kWh</div>
    <div class="metric-value">${avg_cost:.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
(tab_overview, tab_eda, tab_cluster,
 tab_arm, tab_anomaly, tab_insights, tab_map) = st.tabs([
    "📋 Overview",
    "📊 EDA",
    "🤖 Clustering",
    "🔗 Assoc. Rules",
    "🔍 Anomalies",
    "💡 Insights",
    "🗺️ Map",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<p class="section-header">Stage 1 & 2 — Project Scope & Data Preparation</p>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
**Purpose**  
This dashboard investigates Electric Vehicle (EV) charging station behaviour across a global
dataset to uncover usage patterns, inefficiencies, and opportunities for smarter infrastructure
planning.

**Objectives**
- 📈 Understand usage trends by charger type, operator, and geography
- 🤖 Segment stations with K-Means to support investment decisions
- 🔗 Mine association rules (Apriori) to find hidden feature co-occurrences
- 🚨 Flag anomalous stations — high cost, irregular usage, poor reviews
- 🗺️ Visualise geographic demand with an interactive heatmap

**Dataset**  
`cleaned_ev_charging_data.csv` — one row per charging station with charger type, operator,
renewable energy source, cost per kWh, capacity, installation year, reviews, and GPS coordinates.
        """)

    with col_r:
        st.markdown('<p class="section-header">Preprocessing Pipeline</p>', unsafe_allow_html=True)
        steps = {
            "Missing value imputation": "Reviews (Rating) → median fill",
            "Label encoding": "Charger Type, Operator, Renewable, Availability",
            "Min-Max normalisation": "Cost, Usage, Capacity, Distance, Availability_Enc",
            "Feature engineering": "Location Type via distance-to-city median split",
            "Cluster assignment": "K-Means on 5 normalised features",
        }
        for k, v in steps.items():
            st.markdown(f"""
<div class="finding-box">
  <strong>{k}</strong><br>
  <span style="font-size:0.8rem;color:#64748b;">{v}</span>
</div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header" style="margin-top:2rem;">Raw Dataset Preview</p>',
                unsafe_allow_html=True)
    st.dataframe(df_raw.head(50), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<p class="section-header">Stage 3 — Exploratory Data Analysis</p>',
                unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Usage Statistics Distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_raw["Usage Stats (avg users/day)"], bins=22, kde=True,
                     color=TEAL, ax=ax, alpha=0.75)
        ax.set_xlabel("Avg Users / Day"); ax.set_ylabel("Count"); ax.grid(True, axis="y")
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown("**Cost per kWh by Station Operator**")
        fig, ax = plt.subplots(figsize=(7, 4))
        if "Station Operator" in df_raw.columns:
            sns.boxplot(data=df_raw, x="Station Operator", y="Cost (USD/kWh)",
                        palette=PALETTE, ax=ax, linewidth=0.8)
            plt.xticks(rotation=38, ha="right", fontsize=8)
        ax.grid(True, axis="y")
        st.pyplot(fig, use_container_width=True)

    # Row 2 — line chart
    if "Installation Year" in df_raw.columns:
        st.markdown("**Usage Trend by Installation Year**")
        trend = df_raw.groupby("Installation Year")["Usage Stats (avg users/day)"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(trend["Installation Year"], trend["Usage Stats (avg users/day)"],
                marker="o", color=TEAL, linewidth=2, markersize=6)
        ax.fill_between(trend["Installation Year"], trend["Usage Stats (avg users/day)"],
                        alpha=0.12, color=TEAL)
        ax.set_xlabel("Installation Year"); ax.set_ylabel("Avg Users / Day"); ax.grid(True, axis="y")
        st.pyplot(fig, use_container_width=True)

    # Row 3 — Demand heatmap + Rating scatter
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**🔥 Demand Heatmap: Charger Type × Availability**")
        if "Charger Type" in df_raw.columns and "Availability" in df_raw.columns:
            piv = df_raw.pivot_table(
                index="Charger Type", columns="Availability",
                values="Usage Stats (avg users/day)", aggfunc="mean"
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd",
                        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.75})
            ax.set_title("Avg Daily Users")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("Darker = higher average usage.")
        else:
            st.warning("Charger Type or Availability column not found.")

    with c4:
        st.markdown("**⭐ Reviews (Rating) vs. Daily Usage**")
        if "Reviews (Rating)" in df_raw.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            hue_col = "Charger Type" if "Charger Type" in df_raw.columns else None
            sns.scatterplot(data=df_raw, x="Reviews (Rating)",
                            y="Usage Stats (avg users/day)",
                            hue=hue_col, palette=PALETTE, alpha=0.6, s=40, ax=ax)
            xv = df_raw["Reviews (Rating)"].dropna()
            yv = df_raw.loc[xv.index, "Usage Stats (avg users/day)"]
            m, b = np.polyfit(xv, yv, 1)
            ax.plot(sorted(xv), [m * xi + b for xi in sorted(xv)],
                    color="#f97316", linewidth=2, linestyle="--", label="Trend")
            ax.legend(fontsize=7, loc="upper left")
            ax.set_xlabel("Rating"); ax.set_ylabel("Avg Users / Day"); ax.grid(True, axis="y")
            st.pyplot(fig, use_container_width=True)
            corr = df_raw[["Reviews (Rating)", "Usage Stats (avg users/day)"]].corr().iloc[0, 1]
            st.caption(f"Pearson r = **{corr:.3f}**")
        else:
            st.warning("Reviews (Rating) column not found.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab_cluster:
    st.markdown('<p class="section-header">Stage 4 — K-Means Station Segmentation</p>',
                unsafe_allow_html=True)
    st.markdown(f"**Features used:** `{'`, `'.join(cluster_cols)}`")

    # Elbow
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
        km.fit(df_proc[cluster_cols])
        wcss.append(km.inertia_)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("**Elbow Method — Optimal k**")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(range(1, 11), wcss, marker="o", color=TEAL, linewidth=2, markersize=7)
        ax.fill_between(range(1, 11), wcss, alpha=0.1, color=TEAL)
        ax.set_xlabel("k (Number of Clusters)"); ax.set_ylabel("WCSS"); ax.grid(True, axis="y")
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        k_value = st.slider("Select k", 2, 6, 3)
        st.caption("Choose the 'elbow' point where inertia stops dropping sharply.")

    # Fit model and assign clusters
    model = KMeans(n_clusters=k_value, init="k-means++", random_state=42, n_init=10)
    df_raw["Cluster"] = model.fit_predict(df_proc[cluster_cols]).astype(str)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Cluster Scatter: Capacity vs Usage**")
        fig, ax = plt.subplots(figsize=(7, 5))
        for ci, grp in df_raw.groupby("Cluster"):
            ax.scatter(grp["Charging Capacity (kW)"], grp["Usage Stats (avg users/day)"],
                       label=f"Cluster {ci}", alpha=0.7, s=55,
                       color=PALETTE[int(ci) % len(PALETTE)])
        ax.set_xlabel("Charging Capacity (kW)"); ax.set_ylabel("Avg Users / Day")
        ax.legend(); ax.grid(True)
        st.pyplot(fig, use_container_width=True)

    with c4:
        st.markdown("**Cluster Profile (Mean Values)**")
        p_cols = [c for c in ["Charging Capacity (kW)", "Usage Stats (avg users/day)",
                               "Cost (USD/kWh)"] if c in df_raw.columns]
        profile = df_raw[p_cols + ["Cluster"]].groupby("Cluster").mean().round(2)
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(profile.columns))
        bw = 0.8 / k_value
        for i, (idx, row) in enumerate(profile.iterrows()):
            ax.bar(x + i * bw, row.values, bw,
                   label=f"Cluster {idx}", color=PALETTE[int(idx) % len(PALETTE)], alpha=0.85)
        ax.set_xticks(x + bw * (k_value - 1) / 2)
        ax.set_xticklabels(profile.columns, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, axis="y")
        st.pyplot(fig, use_container_width=True)

    # Cluster map
    st.markdown("**Cluster Map — Geographic Segments**")
    if "Latitude" in df_raw.columns and "Longitude" in df_raw.columns:
        PAL_RGB = [[56,245,200],[249,115,22],[129,140,248],[244,63,94],[250,204,21],[52,211,153]]
        dm = df_raw[["Latitude","Longitude","Cluster"]].dropna().copy()
        dm["r"] = dm["Cluster"].apply(lambda c: PAL_RGB[int(c) % len(PAL_RGB)][0])
        dm["g"] = dm["Cluster"].apply(lambda c: PAL_RGB[int(c) % len(PAL_RGB)][1])
        dm["b"] = dm["Cluster"].apply(lambda c: PAL_RGB[int(c) % len(PAL_RGB)][2])

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=pdk.ViewState(
                latitude=dm["Latitude"].mean(), longitude=dm["Longitude"].mean(), zoom=2, pitch=0),
            layers=[pdk.Layer("ScatterplotLayer", data=dm,
                               get_position="[Longitude, Latitude]", get_color="[r, g, b, 210]",
                               radius_min_pixels=5, radius_max_pixels=14, pickable=True)],
            tooltip={"text": "Cluster {Cluster}"},
        ), use_container_width=True)

        leg_cols = st.columns(k_value)
        for ci in range(k_value):
            rgb = PAL_RGB[ci % len(PAL_RGB)]
            hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
            leg_cols[ci].markdown(
                f"<span style='color:{hex_c};font-size:1.3rem;'>●</span> Cluster {ci}",
                unsafe_allow_html=True)
    else:
        st.warning("Latitude / Longitude columns missing.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
with tab_arm:
    st.markdown('<p class="section-header">Stage 5 — Association Rule Mining (Apriori)</p>',
                unsafe_allow_html=True)

    rules_df = pd.DataFrame()

    try:
        ar = pd.DataFrame()

        ar["High_Usage"]   = df_raw["Usage Stats (avg users/day)"] > df_raw["Usage Stats (avg users/day)"].median()
        ar["Fast_Charger"] = df_raw["Charging Capacity (kW)"] > df_raw["Charging Capacity (kW)"].median()
        ar["High_Cost"]    = df_raw["Cost (USD/kWh)"] > df_raw["Cost (USD/kWh)"].median()

        if "Renewable Energy Source" in df_raw.columns:
            ren = df_raw["Renewable Energy Source"]
            ar["Renewable"] = (
                ren.str.strip().str.lower().isin(["yes","1","true"])
                if ren.dtype == object else ren.astype(bool)
            )

        # ── KEY FIX: cast ctype to str before .replace() ──
        if "Charger Type" in df_raw.columns:
            for ctype in df_raw["Charger Type"].dropna().unique():
                ctype_str = str(ctype)
                col_name  = f"Type_{ctype_str.replace(' ', '_')}"
                ar[col_name] = df_raw["Charger Type"].astype(str) == ctype_str

        ar = ar.astype(bool)

        c_sup, c_lft = st.columns(2)
        min_sup = c_sup.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
        min_lft = c_lft.slider("Minimum Lift",    0.5,  3.0, 1.0,  0.1)

        freq_sets = apriori(ar, min_support=min_sup, use_colnames=True)

        if not freq_sets.empty:
            rules_df = arm_rules(freq_sets, metric="lift", min_threshold=min_lft)

            if not rules_df.empty:
                # ── KEY FIX: use str(i) for i in x in join ──
                rules_df["antecedents"] = rules_df["antecedents"].apply(
                    lambda x: ", ".join(str(i) for i in x))
                rules_df["consequents"] = rules_df["consequents"].apply(
                    lambda x: ", ".join(str(i) for i in x))
                rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)

                st.success(f"✅ {len(rules_df)} rules found at support ≥ {min_sup}, lift ≥ {min_lft}")
                st.dataframe(
                    rules_df[["antecedents","consequents","support","confidence","lift"]].head(15),
                    use_container_width=True, height=280)

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Top Rules by Lift**")
                    top10 = rules_df.head(10).copy()
                    top10["rule"] = top10["antecedents"] + " → " + top10["consequents"]
                    fig, ax = plt.subplots(figsize=(7, 5))
                    colors = plt.cm.YlGn(np.linspace(0.35, 0.9, len(top10)))
                    ax.barh(top10["rule"][::-1], top10["lift"][::-1], color=colors[::-1])
                    ax.axvline(1, color="#f97316", linestyle="--", linewidth=1.2, label="Lift = 1")
                    ax.set_xlabel("Lift"); ax.legend(); ax.grid(True, axis="x")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                with c2:
                    st.markdown("**Association Network**")
                    G = nx.DiGraph()
                    for _, row in rules_df.head(15).iterrows():
                        G.add_edge(row["antecedents"], row["consequents"],
                                   weight=row["lift"])
                    fig, ax = plt.subplots(figsize=(7, 5))
                    pos = nx.spring_layout(G, seed=42, k=2.5)
                    ew  = [G[u][v]["weight"] for u, v in G.edges()]
                    mw  = max(ew) if ew else 1
                    nx.draw_networkx_nodes(G, pos, node_size=1600,
                                           node_color=TEAL, alpha=0.85, ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=6.5,
                                            font_color="#0b0f1a", font_weight="bold", ax=ax)
                    nx.draw_networkx_edges(G, pos,
                        width=[1.5 + 4*(w/mw) for w in ew],
                        edge_color=ew, edge_cmap=plt.cm.OrRd,
                        arrows=True, arrowsize=18, ax=ax)
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.OrRd,
                                               norm=plt.Normalize(min(ew), mw))
                    sm.set_array([])
                    plt.colorbar(sm, ax=ax, label="Lift", shrink=0.65)
                    ax.set_title("Edge thickness & colour = Lift", fontsize=9)
                    ax.axis("off"); plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
            else:
                st.warning("No rules found — try lowering Lift threshold.")
        else:
            st.warning("No frequent patterns — try lowering Support.")

    except Exception as e:
        st.error(f"Association Rule Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<p class="section-header">Stage 6 — Anomaly Detection (IQR Method)</p>',
                unsafe_allow_html=True)

    def iqr_out(df, col):
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

    usage_out = iqr_out(df_raw, "Usage Stats (avg users/day)")
    cost_out  = iqr_out(df_raw, "Cost (USD/kWh)")

    hc_lu = hc_lr = pd.DataFrame()
    if "Reviews (Rating)" in df_raw.columns:
        ch = df_raw["Cost (USD/kWh)"].quantile(0.75)
        ul = df_raw["Usage Stats (avg users/day)"].quantile(0.25)
        rl = df_raw["Reviews (Rating)"].quantile(0.25)
        hc_lu = df_raw[(df_raw["Cost (USD/kWh)"] > ch) & (df_raw["Usage Stats (avg users/day)"] < ul)]
        hc_lr = df_raw[(df_raw["Cost (USD/kWh)"] > ch) & (df_raw["Reviews (Rating)"] < rl)]

    maint_out = pd.DataFrame()
    for mc in ["Maintenance Records", "Number of Maintenance Visits"]:
        if mc in df_raw.columns:
            maint_out = iqr_out(df_raw, mc)
            break

    st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Usage Outliers</div>
    <div class="metric-value" style="color:#f97316;">{len(usage_out)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Cost Outliers</div>
    <div class="metric-value" style="color:#f43f5e;">{len(cost_out)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">High Cost + Low Usage</div>
    <div class="metric-value" style="color:#818cf8;">{len(hc_lu)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">High Cost + Low Reviews</div>
    <div class="metric-value" style="color:#facc15;">{len(hc_lr)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Maintenance Anomalies</div>
    <div class="metric-value" style="color:#34d399;">{len(maint_out)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("**Cost vs Usage — Anomalies Highlighted**")
    fig, ax = plt.subplots(figsize=(12, 5))
    normal = ~df_raw.index.isin(usage_out.index) & ~df_raw.index.isin(cost_out.index)
    ax.scatter(df_raw.loc[normal, "Cost (USD/kWh)"],
               df_raw.loc[normal, "Usage Stats (avg users/day)"],
               c="#1e3a4a", alpha=0.5, s=30, label="Normal")
    if len(usage_out):
        ax.scatter(usage_out["Cost (USD/kWh)"], usage_out["Usage Stats (avg users/day)"],
                   c="#f97316", s=90, marker="^", label="Usage Outlier", zorder=5)
    if len(cost_out):
        ax.scatter(cost_out["Cost (USD/kWh)"], cost_out["Usage Stats (avg users/day)"],
                   c="#f43f5e", s=90, marker="X", label="Cost Outlier", zorder=5)
    if len(hc_lu):
        ax.scatter(hc_lu["Cost (USD/kWh)"], hc_lu["Usage Stats (avg users/day)"],
                   c="#818cf8", s=110, marker="D", label="High Cost + Low Usage", zorder=6)
    ax.set_xlabel("Cost (USD/kWh)"); ax.set_ylabel("Avg Users / Day")
    ax.legend(); ax.grid(True)
    st.pyplot(fig, use_container_width=True)

    st.markdown("**Detailed Anomaly Records**")
    a1, a2, a3, a4, a5 = st.tabs(["Usage", "Cost", "High Cost+Low Usage",
                                    "High Cost+Low Reviews", "Maintenance"])
    with a1:
        st.write(f"{len(usage_out)} records")
        if len(usage_out): st.dataframe(usage_out, use_container_width=True)
    with a2:
        st.write(f"{len(cost_out)} records")
        if len(cost_out): st.dataframe(cost_out, use_container_width=True)
    with a3:
        st.write(f"{len(hc_lu)} records")
        if len(hc_lu): st.dataframe(hc_lu, use_container_width=True)
    with a4:
        st.write(f"{len(hc_lr)} records")
        if len(hc_lr): st.dataframe(hc_lr, use_container_width=True)
    with a5:
        if len(maint_out):
            st.write(f"{len(maint_out)} records")
            st.dataframe(maint_out, use_container_width=True)
        else:
            st.info("No maintenance column found or no outliers detected.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown('<p class="section-header">Stage 7 — Interpretation & Key Findings</p>',
                unsafe_allow_html=True)

    st.markdown("**Feature Correlation Matrix**")
    fig, ax = plt.subplots(figsize=(11, 6))
    corr_m = df_proc.select_dtypes(include="number").corr()
    mask   = np.triu(np.ones_like(corr_m, dtype=bool))
    sns.heatmap(corr_m, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                annot_kws={"size": 7}, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.7})
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(fontsize=8); plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🔌 Charger Type Popularity**")
        if "Charger Type" in df_raw.columns:
            ct = (df_raw.groupby("Charger Type")["Usage Stats (avg users/day)"]
                  .mean().sort_values(ascending=True))
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(ct.index, ct.values,
                           color=[PALETTE[i % len(PALETTE)] for i in range(len(ct))])
            ax.set_xlabel("Avg Daily Users"); ax.grid(True, axis="x")
            ax.bar_label(bars, fmt="%.1f", padding=3, color="#94a3b8", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown("**🏢 Operator: Rating vs Usage**")
        if "Station Operator" in df_raw.columns and "Reviews (Rating)" in df_raw.columns:
            op = df_raw.groupby("Station Operator").agg(
                Avg_Rating=("Reviews (Rating)", "mean"),
                Avg_Usage=("Usage Stats (avg users/day)", "mean"),
                Count=("Usage Stats (avg users/day)", "count"),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(op["Avg_Rating"], op["Avg_Usage"],
                       s=op["Count"] * 12, alpha=0.75,
                       c=[PALETTE[i % len(PALETTE)] for i in range(len(op))])
            for _, row in op.iterrows():
                ax.annotate(row["Station Operator"],
                            (row["Avg_Rating"], row["Avg_Usage"]),
                            fontsize=7, ha="center", va="bottom", color="#94a3b8")
            ax.set_xlabel("Avg Rating"); ax.set_ylabel("Avg Daily Users")
            ax.set_title("Bubble size = station count"); ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    if "Distance to City (km)" in df_raw.columns:
        st.markdown("**🏙️ Urban vs Rural Demand**")
        med_d = df_raw["Distance to City (km)"].median()
        df_raw["Location Type"] = np.where(df_raw["Distance to City (km)"] <= med_d, "Urban", "Rural")
        loc = df_raw.groupby("Location Type").agg(
            Avg_Usage=("Usage Stats (avg users/day)", "mean"),
            Avg_Cost=("Cost (USD/kWh)", "mean"),
        ).reset_index()
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(loc["Location Type"], loc["Avg_Usage"],
                   color=["#38f5c8", "#f97316"], width=0.5)
            ax.set_ylabel("Avg Users / Day"); ax.set_title("Usage"); ax.grid(True, axis="y")
            st.pyplot(fig, use_container_width=True)
        with c4:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(loc["Location Type"], loc["Avg_Cost"],
                   color=["#818cf8", "#f43f5e"], width=0.5)
            ax.set_ylabel("Cost (USD/kWh)"); ax.set_title("Cost"); ax.grid(True, axis="y")
            st.pyplot(fig, use_container_width=True)
        st.caption(f"Split: ≤ {med_d:.1f} km = Urban, else Rural.")

    # Summary
    st.markdown('<p class="section-header" style="margin-top:2rem;">📌 Key Takeaways</p>',
                unsafe_allow_html=True)

    avg_by_cl = df_raw.groupby("Cluster")["Usage Stats (avg users/day)"].mean()
    best_cl   = avg_by_cl.idxmax()
    top_ct    = (df_raw.groupby("Charger Type")["Usage Stats (avg users/day)"].mean().idxmax()
                 if "Charger Type" in df_raw.columns else "N/A")

    # Re-derive anomaly counts for summary (in case anomaly tab not run yet)
    def iqr_out2(df, col):
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]

    u_out_cnt = len(iqr_out2(df_raw, "Usage Stats (avg users/day)"))
    hclu_cnt  = (len(df_raw[
        (df_raw["Cost (USD/kWh)"] > df_raw["Cost (USD/kWh)"].quantile(0.75)) &
        (df_raw["Usage Stats (avg users/day)"] < df_raw["Usage Stats (avg users/day)"].quantile(0.25))
    ]) if "Cost (USD/kWh)" in df_raw.columns else 0)

    top_rule  = (f"{rules_df.iloc[0]['antecedents']} → {rules_df.iloc[0]['consequents']}"
                 if not rules_df.empty else "Run Association Rules tab first")

    findings = [
        (f"Top cluster: Cluster {best_cl}",
         f"Averages {avg_by_cl[best_cl]:.1f} users/day — highest across all segments."),
        (f"Most popular charger type: {top_ct}",
         "Attracts the highest average daily usage — prioritise for expansion."),
        (f"Anomalies flagged: {u_out_cnt} usage + {hclu_cnt} high-cost/low-usage",
         "These stations need pricing or operational review."),
        (f"Strongest rule: {top_rule}",
         "Actionable pattern for infrastructure bundling and marketing."),
        ("Urban stations lead in daily usage",
         "Rural stations may benefit from subsidised pricing or renewable integration."),
    ]
    for title, detail in findings:
        st.markdown(f"""
<div class="finding-box">
  <strong>{title}</strong><br>
  <span style="font-size:0.82rem;">{detail}</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown('<p class="section-header">Stage 8 — Geographic Distribution</p>',
                unsafe_allow_html=True)

    if "Latitude" not in df_raw.columns or "Longitude" not in df_raw.columns:
        st.warning("Latitude / Longitude columns not found — map unavailable.")
    else:
        map_mode = st.radio(
            "Select Map Layer",
            ["📍 Raw Stations", "🌡️ Demand Heatmap", "🎨 Cluster Map"],
            horizontal=True,
        )

        base_view = pdk.ViewState(
            latitude=df_raw["Latitude"].mean(),
            longitude=df_raw["Longitude"].mean(),
            zoom=2, pitch=0,
        )

        if map_mode == "📍 Raw Stations":
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=base_view,
                layers=[pdk.Layer(
                    "ScatterplotLayer", data=df_raw,
                    get_position="[Longitude, Latitude]",
                    get_color="[56, 245, 200, 170]",
                    radius_min_pixels=3, radius_max_pixels=10,
                )],
            ), use_container_width=True)

        elif map_mode == "🌡️ Demand Heatmap":
            hdf = df_raw[["Latitude","Longitude","Usage Stats (avg users/day)"]].dropna().copy()
            hdf.rename(columns={"Usage Stats (avg users/day)": "weight"}, inplace=True)
            base_view.pitch = 35
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=base_view,
                layers=[pdk.Layer(
                    "HeatmapLayer", data=hdf,
                    get_position="[Longitude, Latitude]", get_weight="weight",
                    radiusPixels=65, intensity=1.2, threshold=0.04, aggregation="SUM",
                )],
            ), use_container_width=True)
            st.caption("🌡️ Hotter / brighter = higher cumulative daily EV demand.")

        else:  # Cluster Map
            PAL_RGB = [[56,245,200],[249,115,22],[129,140,248],[244,63,94],[250,204,21],[52,211,153]]
            dc = df_raw[["Latitude","Longitude","Cluster"]].dropna().copy()
            dc["r"] = dc["Cluster"].apply(lambda c: PAL_RGB[int(c) % len(PAL_RGB)][0])
            dc["g"] = dc["Cluster"].apply(lambda c: PAL_RGB[int(c) % len(PAL_RGB)][1])
            dc["b"] = dc["Cluster"].apply(lambda c: PAL_RGB[int(c) % len(PAL_RGB)][2])
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=base_view,
                layers=[pdk.Layer(
                    "ScatterplotLayer", data=dc,
                    get_position="[Longitude, Latitude]", get_color="[r, g, b, 210]",
                    radius_min_pixels=5, radius_max_pixels=14, pickable=True,
                )],
                tooltip={"text": "Cluster {Cluster}"},
            ), use_container_width=True)
            k_now = int(df_raw["Cluster"].nunique())
            leg   = st.columns(k_now)
            for ci in range(k_now):
                rgb   = PAL_RGB[ci % len(PAL_RGB)]
                hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
                leg[ci].markdown(
                    f"<span style='color:{hex_c};font-size:1.3rem;'>●</span> Cluster {ci}",
                    unsafe_allow_html=True)

        n_geo = df_raw["Latitude"].notna().sum()
        st.markdown(f"""
<div class="finding-box" style="margin-top:1.2rem;">
  <strong>Dataset coverage</strong> &nbsp;·&nbsp;
  {n_geo:,} geo-located stations plotted across the globe.
</div>""", unsafe_allow_html=True)
