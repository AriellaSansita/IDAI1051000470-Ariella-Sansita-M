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

st.set_page_config(page_title="EV Smart Charging Analytics", page_icon="⚡", layout="wide")
st.title("🚗 Smart Charging Analytics: EV Behavior Patterns")

# ── Data ────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    try:
        df = pd.read_csv("cleaned_ev_charging_data.csv")
    except FileNotFoundError:
        return None, None, None
    d = df.copy()
    if "Reviews (Rating)" in d.columns:
        d["Reviews (Rating)"] = d["Reviews (Rating)"].fillna(d["Reviews (Rating)"].median())
    le = LabelEncoder()
    for c in ["Charger Type", "Station Operator", "Renewable Energy Source", "Availability"]:
        if c in d.columns:
            d[f"{c}_Enc"] = le.fit_transform(d[c].astype(str))
    feats = ["Cost (USD/kWh)", "Usage Stats (avg users/day)", "Charging Capacity (kW)",
             "Distance to City (km)", "Availability_Enc"]
    exist = [f for f in feats if f in d.columns]
    if exist:
        d[exist] = MinMaxScaler().fit_transform(d[exist])
    return df, d, exist

df, dp, ccols = load()
if df is None:
    st.error("❌ cleaned_ev_charging_data.csv not found."); st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📋 Overview", "📊 EDA", "🤖 Clustering",
    "🔗 Assoc. Rules", "🔍 Anomalies", "💡 Insights", "🗺️ Map"
])

# ── Tab 1: Overview ──────────────────────────────────────────────────────────
with t1:
    st.header("Project Scope & Objectives")
    c1, c2 = st.columns(2)
    c1.markdown("""
**Goal:** Uncover EV charging station behaviour to support smarter infrastructure planning.

**Objectives**
1. Identify usage patterns by charger type, operator & location
2. Segment stations via K-Means clustering
3. Mine association rules  for hidden co-occurrences
4. Flag anomalous stations (high cost, low usage/reviews)
5. Visualise geographic demand with an interactive heatmap
""")
    c2.markdown("""
**Preprocessing steps**
- Missing `Reviews (Rating)` → median imputation
- Label encoding: Charger Type, Operator, Renewable, Availability
- Min-Max normalisation on 5 cluster features
- Urban/Rural split via median distance-to-city
""")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stations", len(df))
    m2.metric("Features", df.shape[1])
    m3.metric("Missing (after clean)", int(df.isnull().sum().sum()))
    m4.metric("Avg Daily Users", f"{df['Usage Stats (avg users/day)'].mean():.1f}")
    st.dataframe(df.head(30), use_container_width=True)

# ── Tab 2: EDA ───────────────────────────────────────────────────────────────
with t2:
    st.header("Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(); sns.histplot(df["Usage Stats (avg users/day)"], bins=20, kde=True, color="teal", ax=ax)
        ax.set_title("Usage Distribution"); st.pyplot(fig, use_container_width=True)
    with c2:
        fig, ax = plt.subplots()
        if "Station Operator" in df.columns:
            sns.boxplot(data=df, x="Station Operator", y="Cost (USD/kWh)", palette="Set2", ax=ax)
            plt.xticks(rotation=38, ha="right", fontsize=8)
        ax.set_title("Cost by Operator"); st.pyplot(fig, use_container_width=True)

    if "Installation Year" in df.columns:
        trend = df.groupby("Installation Year")["Usage Stats (avg users/day)"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.lineplot(data=trend, x="Installation Year", y="Usage Stats (avg users/day)", marker="o", ax=ax)
        ax.set_title("Usage Trend by Year"); st.pyplot(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "Charger Type" in df.columns and "Availability" in df.columns:
            piv = df.pivot_table(index="Charger Type", columns="Availability",
                                  values="Usage Stats (avg users/day)", aggfunc="mean")
            fig, ax = plt.subplots()
            sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title("Demand: Charger Type × Availability"); st.pyplot(fig, use_container_width=True)
    with c4:
        if "Reviews (Rating)" in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x="Reviews (Rating)", y="Usage Stats (avg users/day)",
                            hue="Charger Type" if "Charger Type" in df.columns else None,
                            palette="tab10", alpha=0.6, s=40, ax=ax)
            xv = df["Reviews (Rating)"].dropna()
            m, b = np.polyfit(xv, df.loc[xv.index, "Usage Stats (avg users/day)"], 1)
            ax.plot(sorted(xv), [m*x+b for x in sorted(xv)], "r--", lw=1.5, label="Trend")
            ax.legend(fontsize=7); ax.set_title("Reviews vs Usage"); st.pyplot(fig, use_container_width=True)
            st.caption(f"Pearson r = {df[['Reviews (Rating)','Usage Stats (avg users/day)']].corr().iloc[0,1]:.3f}")

# ── Tab 3: Clustering ────────────────────────────────────────────────────────
with t3:
    st.header("K-Means Station Segmentation")
    wcss = [KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10).fit(dp[ccols]).inertia_ for i in range(1, 11)]
    k = st.slider("Number of Clusters (k)", 2, 6, 3)
    model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(dp[ccols]).astype(str)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(); ax.plot(range(1,11), wcss, marker="o", color="#1f77b4")
        ax.set_title("Elbow Method"); ax.set_xlabel("k"); ax.set_ylabel("WCSS"); st.pyplot(fig, use_container_width=True)
    with c2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Charging Capacity (kW)", y="Usage Stats (avg users/day)",
                        hue="Cluster", palette="Set1", s=80, alpha=0.7, ax=ax)
        ax.set_title("Clusters: Capacity vs Usage"); st.pyplot(fig, use_container_width=True)

    if "Latitude" in df.columns:
        PAL = [[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75]]
        dm = df[["Latitude","Longitude","Cluster"]].dropna().copy()
        for i, ch in enumerate(["r","g","b"]):
            dm[ch] = dm["Cluster"].apply(lambda c: PAL[int(c)%len(PAL)][i])
        st.subheader("Cluster Map")
        st.pydeck_chart(pdk.Deck(map_style=None,
            initial_view_state=pdk.ViewState(latitude=dm.Latitude.mean(), longitude=dm.Longitude.mean(), zoom=2),
            layers=[pdk.Layer("ScatterplotLayer", data=dm, get_position="[Longitude,Latitude]",
                               get_color="[r,g,b,200]", radius_min_pixels=5, radius_max_pixels=14)],
        ), use_container_width=True)

# ── Tab 4: Association Rules ──────────────────────────────────────────────────
with t4:
    st.header("Association Rule Mining")
    rules_df = pd.DataFrame()
    try:
        ar = pd.DataFrame({
            "High_Usage":   df["Usage Stats (avg users/day)"] > df["Usage Stats (avg users/day)"].median(),
            "Fast_Charger": df["Charging Capacity (kW)"] > df["Charging Capacity (kW)"].median(),
            "High_Cost":    df["Cost (USD/kWh)"] > df["Cost (USD/kWh)"].median(),
        })
        if "Renewable Energy Source" in df.columns:
            ren = df["Renewable Energy Source"]
            ar["Renewable"] = ren.str.strip().str.lower().isin(["yes","1","true"]) if ren.dtype==object else ren.astype(bool)
        if "Charger Type" in df.columns:
            for ct in df["Charger Type"].dropna().unique():
                ar[f"Type_{str(ct).replace(' ','_')}"] = df["Charger Type"].astype(str) == str(ct)
        ar = ar.astype(bool)

        c1, c2 = st.columns(2)
        sup = c1.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
        lft = c2.slider("Min Lift", 0.5, 3.0, 1.0, 0.1)
        freq = apriori(ar, min_support=sup, use_colnames=True)
        if not freq.empty:
            rules_df = arm_rules(freq, metric="lift", min_threshold=lft)
            if not rules_df.empty:
                rules_df["antecedents"] = rules_df["antecedents"].apply(lambda x: ", ".join(str(i) for i in x))
                rules_df["consequents"] = rules_df["consequents"].apply(lambda x: ", ".join(str(i) for i in x))
                rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
                st.success(f"✅ {len(rules_df)} rules found")
                st.dataframe(rules_df[["antecedents","consequents","support","confidence","lift"]].head(12), use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    top = rules_df.head(10).copy(); top["rule"] = top["antecedents"] + " → " + top["consequents"]
                    fig, ax = plt.subplots()
                    ax.barh(top["rule"][::-1], top["lift"][::-1], color=plt.cm.YlGn(np.linspace(0.4,0.9,10)))
                    ax.axvline(1, color="red", linestyle="--", lw=1); ax.set_xlabel("Lift")
                    ax.set_title("Top Rules by Lift"); plt.tight_layout(); st.pyplot(fig, use_container_width=True)
                with c2:
                    G = nx.DiGraph()
                    for _, row in rules_df.head(15).iterrows():
                        G.add_edge(row["antecedents"], row["consequents"], weight=row["lift"])
                    fig, ax = plt.subplots()
                    pos = nx.spring_layout(G, seed=42, k=2)
                    ew = [G[u][v]["weight"] for u,v in G.edges()]
                    nx.draw_networkx(G, pos, node_color="#4ECDC4", node_size=1400, font_size=6,
                                     width=[1+3*(w/max(ew)) for w in ew], edge_color=ew,
                                     edge_cmap=plt.cm.OrRd, arrows=True, arrowsize=15, ax=ax)
                    ax.set_title("Rule Network"); ax.axis("off"); plt.tight_layout(); st.pyplot(fig, use_container_width=True)
            else:
                st.warning("No rules found — lower Lift threshold.")
        else:
            st.warning("No frequent patterns — lower Support.")
    except Exception as e:
        st.error(f"Error: {e}")

# ── Tab 5: Anomaly Detection ──────────────────────────────────────────────────
with t5:
    st.header("Anomaly Detection (IQR Method)")
    
    # Logic: Define IQR function for numerical outliers
    def iqr(d, c):
        Q1, Q3 = d[c].quantile(0.25), d[c].quantile(0.75)
        IQR = Q3 - Q1
        return d[(d[c] < Q1 - 1.5 * IQR) | (d[c] > Q3 + 1.5 * IQR)]

    # 1. Identify Outliers
    uout = iqr(df, "Usage Stats (avg users/day)")
    cout = iqr(df, "Cost (USD/kWh)")
    
    # 2. Identify Multi-variable Business Anomalies
    hclu = hclr = pout = pd.DataFrame()
    if "Reviews (Rating)" in df.columns:
        # Define threshold as the "Expensive" (75th percentile) and "Poor" (25th percentile)
        ch = df["Cost (USD/kWh)"].quantile(0.75)
        ul = df["Usage Stats (avg users/day)"].quantile(0.25)
        rl = df["Reviews (Rating)"].quantile(0.25)
        
        # High Cost but Low Usage
        hclu = df[(df["Cost (USD/kWh)"] > ch) & (df["Usage Stats (avg users/day)"] < ul)]
        # High Cost but Low Reviews
        hclr = df[(df["Cost (USD/kWh)"] > ch) & (df["Reviews (Rating)"] < rl)]

    # 3. Check for Parking Capacity Outliers (Replacing the missing Maintenance column)
    if "Parking Spots" in df.columns:
        pout = iqr(df, "Parking Spots")

    # Display Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Usage Outliers", len(uout))
    m2.metric("Cost Outliers", len(cout))
    m3.metric("High Cost + Low Usage", len(hclu))
    m4.metric("High Cost + Low Reviews", len(hclr))
    m5.metric("Parking Anomalies", len(pout))

    # Visualization: Anomaly Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Filter for points that are NOT anomalies for the background
    norm = ~df.index.isin(uout.index) & ~df.index.isin(cout.index)
    
    # Plot normal points
    ax.scatter(df.loc[norm, "Cost (USD/kWh)"], df.loc[norm, "Usage Stats (avg users/day)"], 
               c="steelblue", alpha=0.3, s=30, label="Normal")
    
    # Highlight specific anomaly groups
    anomalies = [
        (uout, "orange", "^", "Usage Outlier"),
        (cout, "red", "X", "Cost Outlier"),
        (hclu, "purple", "D", "High Cost + Low Usage")
    ]
    
    for data, col, mk, lbl in anomalies:
        if not data.empty:
            ax.scatter(data["Cost (USD/kWh)"], data["Usage Stats (avg users/day)"], 
                       c=col, s=100, marker=mk, label=lbl, zorder=5)
            
    ax.set_xlabel("Cost (USD/kWh)")
    ax.set_ylabel("Avg Users/Day")
    ax.legend(loc="upper right", fontsize="small")
    ax.set_title("Detection of High-Priority Station Anomalies")
    st.pyplot(fig, use_container_width=True)
    
# ── Tab 6: Insights ───────────────────────────────────────────────────────────
with t6:
    st.header("Insights & Reporting")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(dp.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm",
                annot_kws={"size":7}, ax=ax); plt.xticks(rotation=45, ha="right"); st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if "Charger Type" in df.columns:
            ct = df.groupby("Charger Type")["Usage Stats (avg users/day)"].mean().sort_values()
            fig, ax = plt.subplots(); ax.barh(ct.index, ct.values, color=sns.color_palette("viridis", len(ct)))
            ax.set_title("Charger Type Popularity"); ax.set_xlabel("Avg Daily Users"); st.pyplot(fig, use_container_width=True)
    with c2:
        if "Station Operator" in df.columns and "Reviews (Rating)" in df.columns:
            op = df.groupby("Station Operator").agg(R=("Reviews (Rating)","mean"), U=("Usage Stats (avg users/day)","mean"), N=("Usage Stats (avg users/day)","count")).reset_index()
            fig, ax = plt.subplots()
            ax.scatter(op.R, op.U, s=op.N*10, alpha=0.7, c=range(len(op)), cmap="tab10")
            for _, r in op.iterrows(): ax.annotate(r["Station Operator"], (r.R, r.U), fontsize=7, ha="center", va="bottom")
            ax.set_xlabel("Avg Rating"); ax.set_ylabel("Avg Daily Users"); ax.set_title("Operator: Rating vs Usage"); st.pyplot(fig, use_container_width=True)

    if "Distance to City (km)" in df.columns:
        df["Loc"] = np.where(df["Distance to City (km)"] <= df["Distance to City (km)"].median(), "Urban","Rural")
        loc = df.groupby("Loc").agg(U=("Usage Stats (avg users/day)","mean"), C=("Cost (USD/kWh)","mean")).reset_index()
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(); ax.bar(loc.Loc, loc.U, color=["#2196F3","#4CAF50"]); ax.set_title("Usage: Urban vs Rural"); st.pyplot(fig, use_container_width=True)
        with c4:
            fig, ax = plt.subplots(); ax.bar(loc.Loc, loc.C, color=["#FF5722","#9C27B0"]); ax.set_title("Cost: Urban vs Rural"); st.pyplot(fig, use_container_width=True)

    avg_cl = df.groupby("Cluster")["Usage Stats (avg users/day)"].mean(); best = avg_cl.idxmax()
    top_ct = df.groupby("Charger Type")["Usage Stats (avg users/day)"].mean().idxmax() if "Charger Type" in df.columns else "N/A"
    top_r  = f"{rules_df.iloc[0]['antecedents']} → {rules_df.iloc[0]['consequents']}" if not rules_df.empty else "Run Assoc. Rules tab first"
    st.info(f"""
- **Top Cluster:** Cluster {best} → {avg_cl[best]:.1f} avg users/day
- **Most Popular Charger:** {top_ct}
- **Strongest Rule:** {top_r}
- **Urban stations** typically show higher usage than rural ones.
""")

# ── Tab 7: Map ────────────────────────────────────────────────────────────────
with t7:
    st.header("Geographic Distribution")
    if "Latitude" not in df.columns:
        st.warning("No Latitude/Longitude columns found.")
    else:
        mode = st.radio("Layer", ["📍 Stations", "🌡️ Demand Heatmap", "🎨 Cluster Map"], horizontal=True)
        view = pdk.ViewState(latitude=df.Latitude.mean(), longitude=df.Longitude.mean(), zoom=2, pitch=0)
        if mode == "📍 Stations":
            layer = pdk.Layer("ScatterplotLayer", data=df, get_position="[Longitude,Latitude]",
                               get_color="[255,100,0,160]", radius_min_pixels=3, radius_max_pixels=10)
        elif mode == "🌡️ Demand Heatmap":
            hdf = df[["Latitude","Longitude","Usage Stats (avg users/day)"]].dropna().rename(columns={"Usage Stats (avg users/day)":"weight"})
            view.pitch = 30
            layer = pdk.Layer("HeatmapLayer", data=hdf, get_position="[Longitude,Latitude]",
                               get_weight="weight", radiusPixels=60, intensity=1, threshold=0.05, aggregation="SUM")
        else:
            PAL = [[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75]]
            dc = df[["Latitude","Longitude","Cluster"]].dropna().copy()
            for i,ch in enumerate(["r","g","b"]): dc[ch] = dc["Cluster"].apply(lambda c: PAL[int(c)%len(PAL)][i])
            layer = pdk.Layer("ScatterplotLayer", data=dc, get_position="[Longitude,Latitude]",
                               get_color="[r,g,b,210]", radius_min_pixels=5, radius_max_pixels=14, pickable=True)
        st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view, layers=[layer],
                                  tooltip={"text":"{Cluster}"}), use_container_width=True)
        if mode == "🌡️ Demand Heatmap": st.caption("Brighter areas = higher cumulative daily EV demand.")
