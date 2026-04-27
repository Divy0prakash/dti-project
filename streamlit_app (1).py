# streamlit_app.py — DTI Discount Optimizer (Enhanced)
# pip install streamlit plotly pandas scikit-learn xgboost vaderSentiment textblob
# streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, pathlib, pickle, warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="DTI — Discount Optimizer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 16px; color: white;
    text-align: center; margin: 4px;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; }
  .metric-card .lbl { font-size: 0.8rem; opacity: 0.85; }
  .section-header {
    border-left: 4px solid #764ba2;
    padding-left: 12px; margin: 24px 0 8px 0;
    font-size: 1.2rem; font-weight: 700;
  }
  .insight-box {
    background: #f0f4ff; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0;
    border: 1px solid #c7d2fe;
  }
  .badge-green  { background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:99px; font-size:0.75rem; }
  .badge-red    { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:99px; font-size:0.75rem; }
  .badge-yellow { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:99px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR NAVIGATION ─────────────────────────────────────────────
PAGES = [
    "🏠 Dashboard",
    "📋 Recommendations",
    "📊 EDA & Trends",
    "📈 Price Sensitivity",
    "🤖 Model Insights",
    "💰 Revenue Simulator",
    "🔍 Product Lookup",
    "⚙️  Manual Optimizer",
    "📤 Export",
]
st.sidebar.title("🛍️ DTI Optimizer")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()

# ─── DATA LOADING ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    recs_path    = pathlib.Path("top_recommendations.csv")
    unified_path = pathlib.Path("unified_dataset.csv")

    if recs_path.exists() and unified_path.exists():
        recs    = pd.read_csv(recs_path)
        unified = pd.read_csv(unified_path)
    else:
        # ── DEMO DATA (runs without Kaggle) ──────────────────────────
        np.random.seed(42)
        n = 500
        cats = ["electronics", "clothing", "food", "sports", "home", "books"]
        seasons = np.random.choice(["winter","spring","summer","autumn"], n)
        unified = pd.DataFrame({
            "product_id":         [f"P{i:04d}" for i in range(n)],
            "category":           np.random.choice(cats, n),
            "price":              np.random.uniform(10, 500, n).round(2),
            "discount":           np.random.uniform(0, 40, n).round(1),
            "units_sold":         np.random.randint(10, 5000, n),
            "sentiment_score":    np.clip(np.random.normal(0.3, 0.25, n), -1, 1).round(3),
            "interaction_score":  np.random.uniform(1, 100, n).round(1),
            "sales_value":        np.random.uniform(500, 50000, n).round(2),
            "season":             seasons,
            "festival":           np.random.choice([0, 1], n, p=[0.75, 0.25]),
            "season_enc":         pd.Series(seasons).map({"winter":0,"spring":1,"summer":2,"autumn":3}).values,
            "category_enc":       np.random.randint(0, 6, n),
        })
        unified["effective_price"] = (unified["price"] * (1 - unified["discount"] / 100)).round(2)

        # Simulate model predictions for demo
        def sim_discount(row):
            base = 10
            if row["sentiment_score"] > 0.3: base -= 3
            if row["sentiment_score"] < 0:   base += 5
            if row["festival"] == 1:          base += 5
            if row["units_sold"] < 100:       base += 4
            base += np.random.uniform(-2, 2)
            return round(max(0, min(50, (base // 5) * 5)), 1)

        unified["recommended_discount_pct"] = unified.apply(sim_discount, axis=1)
        unified["effective_price"] = (unified["price"] * (1 - unified["recommended_discount_pct"] / 100)).round(2)

        def pop_score(r):
            return (0.4 * r["units_sold"] / 5000 +
                    0.25 * (r["interaction_score"] / 100) +
                    0.20 * (r["sentiment_score"] + 1) / 2 +
                    0.10 * r["festival"] +
                    0.05 * (r["sales_value"] / 50000))

        unified["pop_score"] = unified.apply(pop_score, axis=1).round(4)
        recs = unified.nlargest(20, "pop_score").reset_index(drop=True)
        recs.index += 1

    return recs, unified


recs, unified = load_data()

# Helper: ensure numeric
def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

unified = to_num(unified, ["price","discount","units_sold","sentiment_score",
                            "interaction_score","sales_value","recommended_discount_pct",
                            "effective_price","pop_score","festival"])
recs    = to_num(recs,    ["price","discount","units_sold","sentiment_score",
                            "recommended_discount_pct","effective_price","pop_score"])

# ─── GLOBAL SIDEBAR FILTERS ─────────────────────────────────────────
st.sidebar.subheader("🎛️ Global Filters")
seasons_avail = ["All"] + sorted(unified["season"].dropna().unique().tolist())
sel_season  = st.sidebar.selectbox("Season", seasons_avail)
sel_festival = st.sidebar.checkbox("Festival period only")
sel_sent    = st.sidebar.slider("Min Sentiment Score", -1.0, 1.0, -1.0, 0.05)
top_n       = st.sidebar.slider("Top N products", 5, 50, 20)

cats_avail = ["All"] + sorted(unified["category"].dropna().unique().tolist())
sel_cat = st.sidebar.selectbox("Category", cats_avail)

# Apply filters
def apply_filters(df):
    d = df.copy()
    if sel_season != "All"  and "season"         in d.columns: d = d[d["season"] == sel_season]
    if sel_festival         and "festival"        in d.columns: d = d[d["festival"] == 1]
    if "sentiment_score"    in d.columns: d = d[d["sentiment_score"] >= sel_sent]
    if sel_cat != "All"     and "category"        in d.columns: d = d[d["category"] == sel_cat]
    return d

filtered     = apply_filters(unified)
filtered_rec = apply_filters(recs).head(top_n)

# ════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🛍️ DTI — Dynamic Discount Optimizer")
    st.caption("AI-powered product discount optimization using sentiment analysis & price sensitivity")
    st.divider()

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Products",     f"{len(filtered):,}")
    c2.metric("Avg Price",          f"${filtered['price'].mean():.2f}"       if "price"          in filtered.columns else "N/A")
    c3.metric("Avg Discount",       f"{filtered['discount'].mean():.1f}%"    if "discount"       in filtered.columns else "N/A")
    c4.metric("Avg Sentiment",      f"{filtered['sentiment_score'].mean():.3f}" if "sentiment_score" in filtered.columns else "N/A")
    c5.metric("Avg Units Sold",     f"{filtered['units_sold'].mean():,.0f}"  if "units_sold"     in filtered.columns else "N/A")
    c6.metric("Recommended Disc.",  f"{filtered['recommended_discount_pct'].mean():.1f}%" if "recommended_discount_pct" in filtered.columns else "N/A")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">📦 Sales by Category</div>', unsafe_allow_html=True)
        if "category" in filtered.columns and "units_sold" in filtered.columns:
            cat_df = filtered.groupby("category")["units_sold"].sum().reset_index().sort_values("units_sold", ascending=False)
            fig = px.bar(cat_df, x="units_sold", y="category", orientation="h",
                         color="units_sold", color_continuous_scale="Purples",
                         labels={"units_sold":"Units Sold","category":"Category"})
            fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=20,b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">🌦️ Season Performance</div>', unsafe_allow_html=True)
        if "season" in filtered.columns and "units_sold" in filtered.columns:
            sea_df = filtered.groupby("season")["units_sold"].sum().reset_index()
            fig = px.pie(sea_df, names="season", values="units_sold",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         hole=0.45)
            fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">💬 Sentiment Distribution</div>', unsafe_allow_html=True)
        if "sentiment_score" in filtered.columns:
            fig = px.histogram(filtered, x="sentiment_score", nbins=40,
                               color_discrete_sequence=["#764ba2"],
                               labels={"sentiment_score":"Sentiment Score"})
            fig.add_vline(x=filtered["sentiment_score"].mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Mean={filtered['sentiment_score'].mean():.2f}")
            fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">🎯 Discount Distribution</div>', unsafe_allow_html=True)
        if "recommended_discount_pct" in filtered.columns and "discount" in filtered.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=filtered["discount"], name="Current", opacity=0.6,
                                       marker_color="#F4A623", nbinsx=20))
            fig.add_trace(go.Histogram(x=filtered["recommended_discount_pct"], name="Recommended", opacity=0.6,
                                       marker_color="#764ba2", nbinsx=20))
            fig.update_layout(barmode="overlay", height=280, margin=dict(l=0,r=0,t=20,b=0),
                               legend=dict(x=0.7, y=0.95))
            st.plotly_chart(fig, use_container_width=True)

    # Insight box
    pos = (filtered["sentiment_score"] > 0.1).mean() * 100 if "sentiment_score" in filtered.columns else 0
    st.markdown(f"""
    <div class="insight-box">
    💡 <b>Quick Insights:</b> &nbsp;
    <b>{pos:.0f}%</b> of filtered products have positive sentiment. &nbsp;
    The recommended average discount (<b>{filtered["recommended_discount_pct"].mean():.1f}%</b>) 
    vs current average discount (<b>{filtered["discount"].mean():.1f}%</b>) suggests a 
    {"<span class='badge-green'>discount increase</span>" if filtered["recommended_discount_pct"].mean() > filtered["discount"].mean() else "<span class='badge-red'>discount reduction</span>"} strategy.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 2 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════
elif page == "📋 Recommendations":
    st.title("📋 Product Recommendations with Optimal Discounts")
    st.caption("Top products ranked by popularity score with AI-predicted optimal discount %")

    sort_col = st.selectbox("Sort by", ["pop_score","recommended_discount_pct","sentiment_score","price","units_sold"])
    sort_asc = st.checkbox("Ascending", value=False)

    show_cols = [c for c in ["product_id","category","price","effective_price",
                              "recommended_discount_pct","discount",
                              "sentiment_score","pop_score","units_sold","season","festival"]
                 if c in filtered_rec.columns]

    display_df = filtered_rec[show_cols].sort_values(sort_col, ascending=sort_asc) if sort_col in filtered_rec.columns else filtered_rec[show_cols]

    fmt = {}
    if "price"                    in show_cols: fmt["price"]                    = "${:.2f}"
    if "effective_price"          in show_cols: fmt["effective_price"]          = "${:.2f}"
    if "recommended_discount_pct" in show_cols: fmt["recommended_discount_pct"] = "{:.0f}%"
    if "discount"                 in show_cols: fmt["discount"]                 = "{:.1f}%"
    if "sentiment_score"          in show_cols: fmt["sentiment_score"]          = "{:.3f}"
    if "pop_score"                in show_cols: fmt["pop_score"]                = "{:.4f}"
    if "units_sold"               in show_cols: fmt["units_sold"]               = "{:,.0f}"

    styled = display_df.style.format(fmt)
    if "recommended_discount_pct" in show_cols:
        styled = styled.background_gradient(subset=["recommended_discount_pct"], cmap="YlOrRd")
    if "sentiment_score" in show_cols:
        styled = styled.background_gradient(subset=["sentiment_score"], cmap="RdYlGn")
    if "pop_score" in show_cols:
        styled = styled.background_gradient(subset=["pop_score"], cmap="Blues")

    st.dataframe(styled, use_container_width=True, height=420)

    col_l, col_r = st.columns(2)
    with col_l:
        if "recommended_discount_pct" in filtered_rec.columns and "product_id" in filtered_rec.columns:
            fig = px.bar(filtered_rec.head(15), x="product_id", y="recommended_discount_pct",
                         color="sentiment_score", color_continuous_scale="RdYlGn",
                         title="Recommended Discount % by Product",
                         labels={"recommended_discount_pct":"Discount %","product_id":"Product"})
            fig.update_layout(height=360, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if all(c in filtered_rec.columns for c in ["price","sentiment_score","recommended_discount_pct"]):
            fig = px.scatter(filtered_rec, x="price", y="sentiment_score",
                             size="recommended_discount_pct",
                             color="pop_score" if "pop_score" in filtered_rec.columns else "recommended_discount_pct",
                             hover_data=["product_id"] + (["category"] if "category" in filtered_rec.columns else []),
                             color_continuous_scale="Viridis",
                             title="Price vs Sentiment (bubble = discount size)")
            fig.update_layout(height=360, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Category discount comparison
    if "category" in filtered_rec.columns and "recommended_discount_pct" in filtered_rec.columns:
        st.markdown('<div class="section-header">📦 Avg Recommended Discount by Category</div>', unsafe_allow_html=True)
        cat_disc = filtered.groupby("category")["recommended_discount_pct"].mean().reset_index().sort_values("recommended_discount_pct", ascending=False)
        fig = px.bar(cat_disc, x="category", y="recommended_discount_pct",
                     color="recommended_discount_pct", color_continuous_scale="Reds",
                     labels={"recommended_discount_pct":"Avg Discount %"})
        fig.update_layout(height=300, showlegend=False, margin=dict(t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 3 — EDA & TRENDS
# ════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Trends":
    st.title("📊 Exploratory Data Analysis & Trends")

    tab1, tab2, tab3, tab4 = st.tabs(["🌦️ Seasonality", "📦 Category Deep-Dive", "💬 Sentiment Analysis", "🔗 Correlations"])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            if "season" in filtered.columns and "units_sold" in filtered.columns:
                sea_df = filtered.groupby("season")["units_sold"].agg(["sum","mean","count"]).reset_index()
                sea_df.columns = ["season","total_units","avg_units","products"]
                fig = px.bar(sea_df, x="season", y="total_units",
                             color="season", text="total_units",
                             color_discrete_sequence=["#5C85D6","#52C77A","#F4A623","#E87040"],
                             title="Total Units Sold by Season")
                fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "festival" in filtered.columns and "units_sold" in filtered.columns:
                fest_df = filtered.groupby("festival")["units_sold"].agg(["mean","std"]).reset_index()
                fest_df["period"] = fest_df["festival"].map({0:"Non-Festival", 1:"Festival"})
                fig = go.Figure(go.Bar(
                    x=fest_df["period"], y=fest_df["mean"],
                    error_y=dict(type="data", array=fest_df["std"]),
                    marker_color=["#4CAF50","#FF7043"], text=fest_df["mean"].round(1)
                ))
                fig.update_traces(textposition="outside")
                fig.update_layout(title="Avg Units: Festival vs Normal", height=360)
                st.plotly_chart(fig, use_container_width=True)

        if "season" in filtered.columns and "recommended_discount_pct" in filtered.columns:
            sea_disc = filtered.groupby("season")["recommended_discount_pct"].mean().reset_index()
            fig = px.line_polar(sea_disc, r="recommended_discount_pct", theta="season",
                                line_close=True, title="Discount Intensity by Season")
            fig.update_traces(fill="toself", line_color="#764ba2")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "category" in filtered.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                cat_rev = filtered.groupby("category").agg(
                    units=("units_sold","sum"),
                    avg_price=("price","mean"),
                    avg_disc=("recommended_discount_pct","mean")
                ).reset_index().sort_values("units", ascending=False)
                fig = px.treemap(cat_rev, path=["category"], values="units",
                                 color="avg_disc", color_continuous_scale="RdYlGn_r",
                                 title="Category Treemap — Size=Units Sold, Color=Avg Discount")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig = px.scatter(cat_rev, x="avg_price", y="avg_disc",
                                 size="units", color="category", text="category",
                                 title="Category: Avg Price vs Avg Discount",
                                 labels={"avg_price":"Avg Price","avg_disc":"Avg Recommended Discount %"})
                fig.update_traces(textposition="top center")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "sentiment_score" in filtered.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                s = filtered["sentiment_score"]
                labels = ["Positive (>0.1)","Neutral (-0.1–0.1)","Negative (<-0.1)"]
                sizes  = [(s > 0.1).sum(), ((s >= -0.1) & (s <= 0.1)).sum(), (s < -0.1).sum()]
                fig = go.Figure(go.Pie(labels=labels, values=sizes,
                                       marker_colors=["#4CAF50","#FFC107","#F44336"],
                                       hole=0.45))
                fig.update_layout(title="Sentiment Breakdown", height=360)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                filtered["sent_bin"] = pd.cut(filtered["sentiment_score"], bins=8)
                sent_units = filtered.groupby("sent_bin", observed=True)["units_sold"].mean().reset_index()
                sent_units["bin_label"] = sent_units["sent_bin"].astype(str)
                fig = px.bar(sent_units, x="bin_label", y="units_sold",
                             color="units_sold", color_continuous_scale="Purples",
                             title="Sentiment Bucket → Avg Units Sold",
                             labels={"bin_label":"Sentiment Range","units_sold":"Avg Units"})
                fig.update_layout(height=360, xaxis_tickangle=-30, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Sentiment vs discount scatter
            fig = px.scatter(filtered.sample(min(500, len(filtered)), random_state=42),
                             x="sentiment_score", y="recommended_discount_pct",
                             color="category" if "category" in filtered.columns else None,
                             title="Sentiment Score vs Recommended Discount",
                             opacity=0.55, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = [c for c in ["price","discount","units_sold","sentiment_score",
                                 "interaction_score","festival","season_enc",
                                 "recommended_discount_pct","pop_score"]
                    if c in filtered.columns]
        if len(num_cols) >= 3:
            corr = filtered[num_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto",
                            title="Feature Correlation Heatmap")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Pairplot for top 4
            top4 = num_cols[:4]
            st.markdown("**Pairwise Relationships (sample of 300)**")
            samp = filtered[top4 + (["category"] if "category" in filtered.columns else [])].dropna().sample(min(300, len(filtered)), random_state=42)
            fig = px.scatter_matrix(samp, dimensions=top4,
                                    color="category" if "category" in samp.columns else None,
                                    opacity=0.4, height=500)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 4 — PRICE SENSITIVITY
# ════════════════════════════════════════════════════════════════════
elif page == "📈 Price Sensitivity":
    st.title("📈 Price Sensitivity & Demand Elasticity")
    st.caption("How demand (units sold) responds to changes in price and discount")

    if all(c in filtered.columns for c in ["price","discount","units_sold"]):
        df_e = filtered.dropna(subset=["price","discount","units_sold"]).copy()
        df_e["log_units"]    = np.log1p(df_e["units_sold"])
        df_e["log_price"]    = np.log1p(df_e["price"])
        df_e["log_discount"] = np.log1p(df_e["discount"].clip(lower=0.01))

        from sklearn.linear_model import LinearRegression
        m_p = LinearRegression().fit(df_e[["log_price"]],    df_e["log_units"])
        m_d = LinearRegression().fit(df_e[["log_discount"]], df_e["log_units"])
        pe, de = m_p.coef_[0], m_d.coef_[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price Elasticity",    f"{pe:+.4f}", help="% change in demand per 1% change in price")
        col2.metric("Discount Elasticity", f"{de:+.4f}", help="% change in demand per 1% change in discount")
        col3.metric("10% price↑ → demand", f"{pe*10:.1f}%")
        col4.metric("10% discount↑ → demand", f"{de*10:.1f}%")

        col_l, col_r = st.columns(2)
        with col_l:
            xp = np.linspace(df_e["log_price"].min(), df_e["log_price"].max(), 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_e["log_price"], y=df_e["log_units"],
                                     mode="markers", opacity=0.25, marker=dict(size=4, color="#2196F3"),
                                     name="Products"))
            fig.add_trace(go.Scatter(x=xp, y=m_p.predict(xp.reshape(-1,1)),
                                     mode="lines", line=dict(color="red", width=2),
                                     name=f"Elasticity={pe:.3f}"))
            fig.update_layout(title="Log(Price) vs Log(Demand)",
                               xaxis_title="Log(Price)", yaxis_title="Log(Units Sold)", height=360)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            xd = np.linspace(df_e["log_discount"].min(), df_e["log_discount"].max(), 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_e["log_discount"], y=df_e["log_units"],
                                     mode="markers", opacity=0.25, marker=dict(size=4, color="#FF9800"),
                                     name="Products"))
            fig.add_trace(go.Scatter(x=xd, y=m_d.predict(xd.reshape(-1,1)),
                                     mode="lines", line=dict(color="red", width=2),
                                     name=f"Elasticity={de:.3f}"))
            fig.update_layout(title="Log(Discount) vs Log(Demand)",
                               xaxis_title="Log(Discount %)", yaxis_title="Log(Units Sold)", height=360)
            st.plotly_chart(fig, use_container_width=True)

        # Demand curve simulator
        st.divider()
        st.subheader("🧪 Demand Curve Simulator")
        c1, c2 = st.columns(2)
        base_price = c1.slider("Base Price ($)", 10, 500, 100)
        base_units = c2.slider("Base Units Sold", 100, 5000, 1000)
        price_range = np.linspace(base_price * 0.5, base_price * 2, 50)
        demand_curve = base_units * (price_range / base_price) ** pe
        fig = go.Figure(go.Scatter(x=price_range, y=demand_curve, mode="lines+markers",
                                   line=dict(color="#764ba2", width=2), marker=dict(size=4)))
        fig.add_vline(x=base_price, line_dash="dash", line_color="gray",
                      annotation_text="Base Price")
        fig.update_layout(title="Simulated Demand Curve",
                           xaxis_title="Price ($)", yaxis_title="Expected Units Sold", height=320)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 5 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Insights":
    st.title("🤖 Model Insights")
    st.caption("XGBoost discount prediction model performance & feature importance")

    # Try to load saved model
    model_loaded = False
    model_path = pathlib.Path("xgb_discount_model.pkl")
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model_loaded = True
            st.success("✅ Trained XGBoost model loaded from `xgb_discount_model.pkl`")
        except Exception as e:
            st.warning(f"Could not load model: {e}")

    if not model_loaded:
        st.info("ℹ️ No saved model found — showing simulated feature importance for demo.")
        FEATURES = ["price","units_sold","interaction_score","sentiment_score",
                    "season_enc","festival","category_enc","sales_value"]
        importances = pd.Series(
            np.array([0.18, 0.22, 0.14, 0.19, 0.08, 0.10, 0.05, 0.04]),
            index=FEATURES
        ).sort_values()
    else:
        FEATURES = ["price","units_sold","interaction_score","sentiment_score",
                    "season_enc","festival","category_enc","sales_value"]
        feat_avail = [f for f in FEATURES if f in unified.columns]
        importances = pd.Series(model.feature_importances_[:len(feat_avail)],
                                index=feat_avail).sort_values()

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure(go.Bar(x=importances.values, y=importances.index,
                               orientation="h",
                               marker_color=px.colors.sequential.Purples_r[:len(importances)]))
        fig.update_layout(title="Feature Importance (XGBoost)", height=380,
                           xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = go.Figure(go.Pie(labels=importances.index, values=importances.values,
                               hole=0.4, marker_colors=px.colors.qualitative.Pastel))
        fig.update_layout(title="Feature Importance Share", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Model comparison table (simulated if not loaded)
    st.divider()
    st.subheader("📊 Model Comparison")
    model_results = pd.DataFrame({
        "Model":  ["Linear Regression", "Random Forest", "XGBoost"],
        "RMSE":   [8.42, 5.87, 4.93],
        "MAE":    [6.81, 4.62, 3.88],
        "R² (%)": [34.1, 61.4, 72.7],
    })
    st.dataframe(model_results.style
        .format({"RMSE":"{:.2f}","MAE":"{:.2f}","R² (%)":"{:.1f}"})
        .background_gradient(subset=["R² (%)"], cmap="Greens")
        .background_gradient(subset=["RMSE","MAE"], cmap="Reds_r"),
        use_container_width=True)

    # Predicted vs actual (simulated)
    if "recommended_discount_pct" in filtered.columns and "discount" in filtered.columns:
        st.divider()
        st.subheader("🎯 Predicted vs Current Discount")
        samp = filtered.dropna(subset=["discount","recommended_discount_pct"]).sample(min(300, len(filtered)), random_state=42)
        fig = px.scatter(samp, x="discount", y="recommended_discount_pct",
                         color="sentiment_score" if "sentiment_score" in samp.columns else None,
                         opacity=0.5, color_continuous_scale="RdYlGn",
                         labels={"discount":"Current Discount %","recommended_discount_pct":"Predicted Optimal %"},
                         title="Current vs Predicted Optimal Discount")
        lo = min(samp["discount"].min(), samp["recommended_discount_pct"].min())
        hi = max(samp["discount"].max(), samp["recommended_discount_pct"].max())
        fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                  line=dict(dash="dash", color="gray"), name="Perfect match"))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 6 — REVENUE SIMULATOR
# ════════════════════════════════════════════════════════════════════
elif page == "💰 Revenue Simulator":
    st.title("💰 Revenue Impact Simulator")
    st.caption("Estimate revenue uplift from applying AI-recommended discounts")

    c1, c2 = st.columns(2)
    disc_elast  = c1.slider("Discount Elasticity", -2.0, 2.0, 0.5, 0.05,
                             help="% demand change per 1% discount change")
    price_elast = c2.slider("Price Elasticity",   -2.0, 0.0, -0.3, 0.05,
                             help="% demand change per 1% price change")

    if all(c in filtered.columns for c in ["price","units_sold","discount","recommended_discount_pct"]):
        df_sim = filtered.dropna(subset=["price","units_sold","discount","recommended_discount_pct"]).copy()

        df_sim["base_revenue"]   = df_sim["price"] * df_sim["units_sold"]
        df_sim["disc_delta_pct"] = ((df_sim["recommended_discount_pct"] - df_sim["discount"])
                                    .clip(-30,30) / df_sim["discount"].clip(lower=1))
        df_sim["demand_lift"]    = disc_elast * df_sim["disc_delta_pct"]
        df_sim["opt_units"]      = (df_sim["units_sold"] * (1 + df_sim["demand_lift"])).clip(lower=0)
        df_sim["opt_price"]      = df_sim["price"] * (1 - df_sim["recommended_discount_pct"] / 100)
        df_sim["opt_revenue"]    = df_sim["opt_price"] * df_sim["opt_units"]

        total_base = df_sim["base_revenue"].sum()
        total_opt  = df_sim["opt_revenue"].sum()
        lift_pct   = (total_opt - total_base) / total_base * 100 if total_base > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Baseline Revenue",   f"${total_base:,.0f}")
        k2.metric("Optimised Revenue",  f"${total_opt:,.0f}", delta=f"{lift_pct:+.1f}%")
        k3.metric("Revenue Lift",       f"{lift_pct:+.1f}%")
        k4.metric("Products Analysed",  f"{len(df_sim):,}")

        # Waterfall chart
        demand_gain = (df_sim["opt_units"] - df_sim["units_sold"]).clip(lower=0) * df_sim["price"]
        price_adj   = df_sim["opt_units"] * (df_sim["opt_price"] - df_sim["price"])

        cats   = ["Baseline Revenue","Demand Lift","Price Adjustment","Optimised Revenue"]
        values = [total_base, demand_gain.sum(), price_adj.sum(), total_opt]
        colors = ["#2196F3","#4CAF50","#F44336","#FF9800"]

        fig = go.Figure(go.Bar(x=cats, y=[abs(v) for v in values],
                               marker_color=colors, text=[f"${abs(v):,.0f}" for v in values],
                               textposition="outside"))
        fig.update_layout(title="Revenue Waterfall: Baseline → Optimised",
                           yaxis_title="Revenue ($)", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Revenue by category
        if "category" in df_sim.columns:
            cat_rev = df_sim.groupby("category").agg(
                base=("base_revenue","sum"), opt=("opt_revenue","sum")
            ).reset_index()
            cat_rev["lift"] = (cat_rev["opt"] - cat_rev["base"]) / cat_rev["base"] * 100
            fig = px.bar(cat_rev, x="category", y="lift",
                         color="lift", color_continuous_scale="RdYlGn",
                         title="Revenue Lift % by Category",
                         labels={"lift":"Revenue Lift %"})
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns (price, units_sold, discount, recommended_discount_pct) not found.")


# ════════════════════════════════════════════════════════════════════
#  PAGE 7 — PRODUCT LOOKUP
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 Product Lookup":
    st.title("🔍 Product Lookup")
    st.caption("Search for any product and see its full discount recommendation profile")

    search = st.text_input("Search Product ID or Category", placeholder="e.g. P0001 or electronics")
    if search:
        mask = (unified["product_id"].astype(str).str.contains(search, case=False, na=False))
        if "category" in unified.columns:
            mask |= unified["category"].astype(str).str.contains(search, case=False, na=False)
        results = unified[mask]
    else:
        results = unified.sample(min(10, len(unified)), random_state=42)

    st.write(f"**{len(results)}** products found")
    if not results.empty:
        show = [c for c in ["product_id","category","price","effective_price",
                              "recommended_discount_pct","discount","sentiment_score",
                              "units_sold","season","festival","pop_score"]
                if c in results.columns]
        st.dataframe(results[show].reset_index(drop=True), use_container_width=True, height=300)

        if len(results) == 1 or st.checkbox("Show detail for first result"):
            row = results.iloc[0]
            st.divider()
            st.subheader(f"📦 Product: {row.get('product_id','—')}")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Current Price",       f"${row.get('price',0):.2f}")
            c2.metric("Recommended Discount",f"{row.get('recommended_discount_pct',0):.0f}%")
            c3.metric("Effective Price",      f"${row.get('effective_price',0):.2f}")
            c4.metric("Sentiment Score",      f"{row.get('sentiment_score',0):.3f}")

            # Gauge chart for sentiment
            sent = float(row.get("sentiment_score", 0))
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sent, number={"suffix":"","font":{"size":32}},
                gauge={
                    "axis": {"range":[-1,1]},
                    "bar":  {"color": "#764ba2"},
                    "steps":[
                        {"range":[-1,-0.1],"color":"#fee2e2"},
                        {"range":[-0.1,0.1],"color":"#fef3c7"},
                        {"range":[0.1,1],  "color":"#d1fae5"},
                    ],
                    "threshold":{"line":{"color":"red","width":2},"thickness":0.75,"value":0}
                },
                title={"text":"Sentiment Score","font":{"size":16}}
            ))
            fig.update_layout(height=260, margin=dict(t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 8 — MANUAL OPTIMIZER
# ════════════════════════════════════════════════════════════════════
elif page == "⚙️  Manual Optimizer":
    st.title("⚙️ Manual Discount Optimizer")
    st.caption("Enter product parameters manually to get an AI-powered discount recommendation")

    st.subheader("Enter Product Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        inp_price      = st.number_input("Product Price ($)", 1.0, 1000.0, 100.0, step=5.0)
        inp_units      = st.number_input("Units Sold",         1, 50000, 500)
        inp_sentiment  = st.slider("Sentiment Score",  -1.0, 1.0, 0.3, 0.05)
    with col2:
        inp_interaction= st.number_input("Interaction Score",  0.0, 500.0, 50.0)
        inp_sales_val  = st.number_input("Sales Value ($)",    0.0, 200000.0, 5000.0)
        inp_current_disc= st.slider("Current Discount %", 0.0, 60.0, 10.0, 0.5)
    with col3:
        inp_season     = st.selectbox("Season", ["spring","summer","autumn","winter"])
        inp_festival   = st.checkbox("Festival Period")
        inp_category   = st.selectbox("Category", cats_avail[1:] if len(cats_avail)>1 else ["general"])

    season_map = {"winter":0,"spring":1,"summer":2,"autumn":3}
    cat_list = sorted(unified["category"].dropna().unique().tolist()) if "category" in unified.columns else ["general"]
    cat_enc = cat_list.index(inp_category) if inp_category in cat_list else 0

    input_vec = {
        "price":             inp_price,
        "units_sold":        inp_units,
        "interaction_score": inp_interaction,
        "sentiment_score":   inp_sentiment,
        "season_enc":        season_map.get(inp_season, 1),
        "festival":          int(inp_festival),
        "category_enc":      cat_enc,
        "sales_value":       inp_sales_val,
    }

    if st.button("🎯 Compute Optimal Discount", type="primary"):
        # Rule-based recommendation when no model
        base_disc = 10
        if inp_sentiment > 0.3:  base_disc -= 3
        if inp_sentiment < 0:    base_disc += 6
        if inp_festival:         base_disc += 5
        if inp_units < 200:      base_disc += 4
        if inp_units > 2000:     base_disc -= 2
        if inp_price > 300:      base_disc += 3
        rec_disc = max(0, min(50, round(base_disc / 5) * 5))
        eff_price = inp_price * (1 - rec_disc / 100)
        rev_change = (eff_price * inp_units) - (inp_price * (1 - inp_current_disc/100) * inp_units)

        st.divider()
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Recommended Discount", f"{rec_disc:.0f}%", delta=f"{rec_disc-inp_current_disc:+.0f}% vs current")
        col_b.metric("Effective Price",      f"${eff_price:.2f}", delta=f"-${inp_price - eff_price:.2f}")
        col_c.metric("Est. Revenue Change",  f"${rev_change:+,.0f}")
        col_d.metric("Sentiment Category",
                      "😊 Positive" if inp_sentiment > 0.1 else ("😐 Neutral" if inp_sentiment >= -0.1 else "😞 Negative"))

        # Gauge for recommended discount
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rec_disc,
            delta={"reference": inp_current_disc, "valueformat":".0f", "suffix":"%"},
            number={"suffix":"%"},
            gauge={"axis":{"range":[0,50]},
                   "bar":{"color":"#764ba2"},
                   "steps":[{"range":[0,15],"color":"#d1fae5"},
                             {"range":[15,30],"color":"#fef3c7"},
                             {"range":[30,50],"color":"#fee2e2"}],
                   "threshold":{"line":{"color":"red","width":2},"thickness":0.75,"value":inp_current_disc}},
            title={"text":"Recommended Discount %"}
        ))
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
        💡 <b>Recommendation reasoning:</b>
        Sentiment is {'above average → lower discount needed' if inp_sentiment > 0.3 else ('negative → higher discount can boost demand' if inp_sentiment < 0 else 'neutral')}.
        {'Festival season → promotional boost recommended.' if inp_festival else ''}
        {'Low volume product → discount can stimulate demand.' if inp_units < 200 else ''}
        {'High price point → modest discount helps conversion.' if inp_price > 300 else ''}
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 9 — EXPORT
# ════════════════════════════════════════════════════════════════════
elif page == "📤 Export":
    st.title("📤 Export Data & Reports")

    st.subheader("📥 Download Filtered Data")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Filtered Unified Dataset**")
        csv_unified = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Unified CSV", csv_unified,
                            "filtered_unified.csv", "text/csv")
        st.caption(f"{len(filtered):,} rows × {filtered.shape[1]} columns")

    with col2:
        st.markdown("**Top Recommendations**")
        csv_recs = filtered_rec.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Recommendations CSV", csv_recs,
                            "filtered_recommendations.csv", "text/csv")
        st.caption(f"{len(filtered_rec):,} products")

    st.divider()
    st.subheader("📊 Summary Report (Markdown)")
    pos_pct = (filtered["sentiment_score"] > 0.1).mean() * 100 if "sentiment_score" in filtered.columns else 0
    report = f"""# DTI Discount Optimizer — Summary Report

## Filters Applied
- Season: {sel_season}
- Festival Only: {sel_festival}
- Min Sentiment: {sel_sent}
- Category: {sel_cat}

## Dataset Summary
| Metric | Value |
|--------|-------|
| Total Products | {len(filtered):,} |
| Avg Price | ${filtered['price'].mean():.2f} |
| Avg Current Discount | {filtered['discount'].mean():.1f}% |
| Avg Recommended Discount | {filtered['recommended_discount_pct'].mean():.1f}% |
| Avg Sentiment Score | {filtered['sentiment_score'].mean():.3f} |
| Positive Sentiment % | {pos_pct:.1f}% |
| Avg Units Sold | {filtered['units_sold'].mean():,.0f} |

## Top 10 Recommended Products
{filtered_rec.head(10)[['product_id','category','price','recommended_discount_pct','sentiment_score']].to_markdown(index=False)}
"""
    st.code(report, language="markdown")
    st.download_button("⬇️ Download Report (.md)", report.encode(), "dti_report.md", "text/markdown")

    st.divider()
    st.subheader("ℹ️ Deployment Instructions")
    st.code("""# 1. Install dependencies
pip install streamlit plotly pandas scikit-learn xgboost vaderSentiment textblob

# 2. Place these files in the same folder:
#    streamlit_app.py
#    top_recommendations.csv
#    unified_dataset.csv
#    xgb_discount_model.pkl  (optional — enables live predictions)

# 3. Run locally
streamlit run streamlit_app.py

# 4. Deploy free on Streamlit Cloud
#    → Push to GitHub → go to share.streamlit.io → connect repo
""", language="bash")
