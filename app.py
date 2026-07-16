import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Price Intelligence",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background — bright, colorful, gently animated mesh */
    .stApp {
        background:
            radial-gradient(ellipse 900px 600px at 8% -10%, rgba(59,130,246,0.16), transparent 60%),
            radial-gradient(ellipse 800px 700px at 100% 0%, rgba(139,92,246,0.14), transparent 55%),
            radial-gradient(ellipse 900px 800px at 50% 110%, rgba(52,211,153,0.14), transparent 60%),
            radial-gradient(ellipse 700px 600px at 90% 90%, rgba(236,72,153,0.10), transparent 55%),
            linear-gradient(180deg, #f7f9fd 0%, #eef2fb 45%, #f3f0fb 100%);
        background-attachment: fixed, fixed, fixed, fixed, fixed;
        background-size: 200% 200%, 200% 200%, 200% 200%, 200% 200%, 100% 100%;
        animation: bgDrift 30s ease-in-out infinite;
        color: #1f2937;
        position: relative;
    }
    @keyframes bgDrift {
        0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 100% 100%, 0 0; }
        50%  { background-position: 15% 10%, 85% 15%, 55% 90%, 90% 85%, 0 0; }
        100% { background-position: 0% 0%, 100% 0%, 50% 100%, 100% 100%, 0 0; }
    }
    /* Subtle dotted grid overlay for texture */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(100,116,139,0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(100,116,139,0.07) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: radial-gradient(ellipse 80% 80% at 50% 20%, #000 40%, transparent 100%);
        z-index: 0;
    }
    .stApp > * { position: relative; z-index: 1; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f7ff 100%) !important;
        border-right: 1px solid #e5e9f2;
        box-shadow: 2px 0 16px rgba(30,41,59,0.04);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #2563eb !important;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border: 1px solid #e5e9f2;
        border-radius: 14px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 18px rgba(59,130,246,0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
        box-shadow: 0 12px 30px -6px rgba(59,130,246,0.25);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        border-radius: 14px 14px 0 0;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        line-height: 1;
    }
    .kpi-delta {
        font-size: 12px;
        font-weight: 500;
        margin-top: 6px;
        color: #10b981;
    }

    /* Section headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        font-weight: 600;
        color: #334155;
        letter-spacing: 0.3px;
        padding-bottom: 6px;
        border-bottom: 2px solid #dbe3f3;
        margin-bottom: 16px;
    }

    /* Metric override */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e9f2;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 10px rgba(30,41,59,0.05);
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 11px !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #10b981 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-bottom: 1px solid #e5e9f2;
        border-radius: 10px 10px 0 0;
        gap: 4px;
        padding: 4px 4px 0;
        box-shadow: 0 2px 10px rgba(30,41,59,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 500;
        font-size: 13px;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #eef2ff, #f5f0ff) !important;
        color: #2563eb !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        padding: 10px 24px;
        width: 100%;
        box-shadow: 0 4px 14px rgba(99,102,241,0.30);
        transition: opacity 0.2s, transform 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        color: white;
        border: none;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid #dbe3f3 !important;
        border-radius: 8px !important;
        color: #1f2937 !important;
    }

    /* Number inputs */
    .stNumberInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #dbe3f3 !important;
        border-radius: 8px !important;
        color: #1f2937 !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid #e5e9f2;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(30,41,59,0.05);
    }

    /* Divider */
    hr { border-color: #e5e9f2; }

    /* Success / info banners */
    .stSuccess { background: #e8f9f1 !important; border-left: 4px solid #10b981 !important; color: #065f46 !important; }
    .stInfo    { background: #eaf2ff !important; border-left: 4px solid #3b82f6 !important; color: #1e3a8a !important; }

    /* Hide default header */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly dark template ────────────────────────────────────────────────────
PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#f6f8fc",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter", color="#6b7280", size=12),
        colorway=["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4","#ec4899"],
        margin=dict(l=40, r=20, t=40, b=40),
    )
)

_AXIS_STYLE = dict(gridcolor="#e5e9f2", linecolor="#e5e9f2", zerolinecolor="#e5e9f2")

def apply_axis_style(fig):
    """Apply consistent dark axis styling without conflicting with per-chart axis kwargs."""
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig

_LEGEND_BASE = dict(bgcolor="#ffffff", bordercolor="#e5e9f2", borderwidth=1)

def mk_legend(**extra):
    """Return legend dict merging dark base styles with any per-chart overrides."""
    return {**_LEGEND_BASE, **extra}

ACCENT = ["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444","#06b6d4"]

# ─── Load Dataset (backend — no upload needed) ───────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("retail_price.csv")
    df.columns = df.columns.str.strip()
    df["month_year"] = pd.to_datetime(df["month_year"], format="%d-%m-%Y", errors="coerce")
    df["price_to_freight_ratio"] = df["unit_price"] / (df["freight_price"] + 1)
    df["avg_revenue_per_customer"] = df["total_price"] / (df["customers"] + 1)
    df["comp_avg_price"] = df[["comp_1","comp_2","comp_3"]].mean(axis=1)
    df["price_vs_comp"] = df["unit_price"] - df["comp_avg_price"]
    return df

df = load_data()

FEATURES = ["qty","total_price","freight_price","product_weight_g",
            "product_score","customers","weekday","holiday","month","year"]
TARGET = "unit_price"
CATEGORIES = sorted(df["product_category_name"].unique())

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 20px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700;
                    background: linear-gradient(90deg,#3b82f6,#8b5cf6); -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;">💹 PriceIQ</div>
        <div style="font-size:11px; color:#475569; margin-top:4px; letter-spacing:1px;">RETAIL INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Category filter ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:10px; font-weight:700; letter-spacing:1.4px;
                text-transform:uppercase; color:#475569; margin-bottom:6px;">
        Product Categories
    </div>
    """, unsafe_allow_html=True)

    CAT_LABELS = {
        "bed_bath_table":        "Bed & Bath",
        "garden_tools":          "Garden Tools",
        "consoles_games":        "Consoles & Games",
        "health_beauty":         "Health & Beauty",
        "cool_stuff":            "Cool Stuff",
        "perfumery":             "Perfumery",
        "computers_accessories": "Computers & Accessories",
        "watches_gifts":         "Watches & Gifts",
        "furniture_decor":       "Furniture & Decor",
    }
    cat_display = [CAT_LABELS.get(c, c.replace("_"," ").title()) for c in CATEGORIES]
    cat_display_all = ["All Categories"] + cat_display

    selected_cat_label = st.selectbox(
        "",
        cat_display_all,
        index=0,
        label_visibility="collapsed",
        key="cat_select"
    )
    if selected_cat_label == "All Categories":
        selected_cats = CATEGORIES
    else:
        selected_cats = [CATEGORIES[cat_display.index(selected_cat_label)]]

    # ── Year filter ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:10px; font-weight:700; letter-spacing:1.4px;
                text-transform:uppercase; color:#475569; margin:14px 0 6px;">
        Year
    </div>
    """, unsafe_allow_html=True)

    YEARS = sorted(df["year"].unique())
    year_display_all = ["All Years"] + [str(y) for y in YEARS]

    selected_year_label = st.selectbox(
        "",
        year_display_all,
        index=0,
        label_visibility="collapsed",
        key="year_select"
    )
    if selected_year_label == "All Years":
        year_filter = YEARS
    else:
        year_filter = [int(selected_year_label)]

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px; font-weight:700; letter-spacing:1.4px;
                text-transform:uppercase; color:#475569; margin-bottom:6px;">
        ML Algorithm
    </div>
    """, unsafe_allow_html=True)
    model_choice = st.selectbox(
        "",
        ["Random Forest","Gradient Boosting","Linear Regression","Decision Tree","SVM"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    active_cats = len(selected_cats)
    st.markdown(f"""
    <div style="background:#ffffff; border:1px solid #e5e9f2; border-radius:8px;
                padding:10px 14px; font-size:11px; color:#64748b; line-height:1.8;">
        <span style="color:#2563eb; font-weight:600;">{len(df):,}</span> total records<br>
        <span style="color:#2563eb; font-weight:600;">{active_cats}</span> of {len(CATEGORIES)} categories<br>
        <span style="color:#2563eb; font-weight:600;">{len(year_filter)}</span> of {len(YEARS)} years active
    </div>
    """, unsafe_allow_html=True)

# ─── Filter data ─────────────────────────────────────────────────────────────
filtered = df[
    df["product_category_name"].isin(selected_cats) &
    df["year"].isin(year_filter)
].copy()

# ─── Page Title ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px;">
    <h1 style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700;
               color:#111827; margin:0; letter-spacing:-0.5px;">
        Retail Price Optimization
    </h1>
    <p style="color:#475569; font-size:13px; margin-top:4px;">
        Competitive intelligence · Revenue analysis · ML price prediction
    </p>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Unit Price</div>
        <div class="kpi-value">₹{filtered['unit_price'].mean():.0f}</div>
        <div class="kpi-delta">↑ across {len(filtered):,} records</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value">₹{filtered['total_price'].sum()/1e6:.2f}M</div>
        <div class="kpi-delta">↑ {len(selected_cats)} categories</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Product Score</div>
        <div class="kpi-value">{filtered['product_score'].mean():.2f}</div>
        <div class="kpi-delta">⭐ out of 5.0</div>
    </div>""", unsafe_allow_html=True)

with k4:
    price_adv = (filtered["price_vs_comp"] < 0).mean() * 100
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Price Advantage</div>
        <div class="kpi-value">{price_adv:.0f}%</div>
        <div class="kpi-delta">↓ below competitor avg</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Freight Cost</div>
        <div class="kpi-value">₹{filtered['freight_price'].mean():.0f}</div>
        <div class="kpi-delta">per unit shipped</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Overview",
    "🏷️ Price Analysis",
    "🤖 ML Prediction",
    "🔍 Data Explorer",
    "🎛️ What-If Simulator",
    "🔬 Product Deep Dive"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Revenue Trend Over Time</div>', unsafe_allow_html=True)
        ts = filtered.groupby("month_year").agg(
            total_rev=("total_price","sum"),
            avg_price=("unit_price","mean"),
            orders=("qty","sum")
        ).reset_index().sort_values("month_year")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["month_year"], y=ts["total_rev"],
            fill="tozeroy", mode="lines",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.12)",
            name="Total Revenue"
        ))
        fig.add_trace(go.Scatter(
            x=ts["month_year"], y=ts["avg_price"],
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=2, dash="dot"),
            marker=dict(size=5),
            name="Avg Unit Price",
            yaxis="y2"
        ))
        fig.update_layout(
            **PLOT_TEMPLATE["layout"],
            yaxis2=dict(title="Avg Price (₹)", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
            legend=mk_legend(orientation="h", y=1.08),
            height=300,
            title="",
        )
        fig.update_yaxes(title_text="Revenue (₹)", gridcolor="#e5e9f2", selector=dict(side="left"))
        apply_axis_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = filtered.groupby("product_category_name")["total_price"].sum().sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(
            x=cat_rev.values, y=cat_rev.index,
            orientation="h",
            marker=dict(
                color=cat_rev.values,
                colorscale=[[0,"#dbe3f3"],[0.5,"#3b82f6"],[1,"#8b5cf6"]],
                showscale=False
            ),
            text=[f"₹{v/1e3:.0f}K" for v in cat_rev.values],
            textposition="outside",
            textfont=dict(color="#6b7280", size=11)
        ))
        fig2.update_layout(**PLOT_TEMPLATE["layout"], height=300, xaxis_title="", yaxis_title="")
        apply_axis_style(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d, col_e = st.columns(3)

    with col_c:
        st.markdown('<div class="section-header">Sales Volume Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(
            filtered, x="qty", nbins=30,
            color_discrete_sequence=["#3b82f6"],
            labels={"qty":"Quantity Sold"},
        )
        fig3.update_traces(marker_line_width=0.5, marker_line_color="#f6f8fc")
        fig3.update_layout(**PLOT_TEMPLATE["layout"], height=240, title="", showlegend=False)
        apply_axis_style(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Avg Score by Category</div>', unsafe_allow_html=True)
        scores = filtered.groupby("product_category_name")["product_score"].mean().sort_values()
        fig4 = go.Figure(go.Bar(
            x=[s.replace("_"," ").title() for s in scores.index],
            y=scores.values,
            marker_color=ACCENT[:len(scores)],
            text=[f"{v:.2f}" for v in scores.values],
            textposition="outside",
            textfont=dict(color="#6b7280", size=11)
        ))
        fig4.update_layout(**PLOT_TEMPLATE["layout"], height=240, title="")
        fig4.update_yaxes(range=[3.0, 4.8])
        apply_axis_style(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    with col_e:
        st.markdown('<div class="section-header">Holiday vs Weekday Sales</div>', unsafe_allow_html=True)
        hmap = filtered.groupby(["weekday","holiday"])["total_price"].mean().unstack(fill_value=0)
        fig5 = px.imshow(
            hmap,
            color_continuous_scale=[[0,"#ffffff"],[0.5,"#dbeafe"],[1,"#3b82f6"]],
            labels=dict(x="Holiday", y="Weekday", color="Avg Rev"),
            aspect="auto"
        )
        fig5.update_layout(**PLOT_TEMPLATE["layout"], height=240, title="",
                           coloraxis_colorbar=dict(thickness=12, tickfont=dict(size=10)))
        apply_axis_style(fig5)
        st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — PRICE ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown('<div class="section-header">Unit Price Distribution by Category</div>', unsafe_allow_html=True)
        fig_box = go.Figure()
        for i, cat in enumerate(filtered["product_category_name"].unique()):
            vals = filtered[filtered["product_category_name"]==cat]["unit_price"]
            fig_box.add_trace(go.Box(
                y=vals, name=cat.replace("_"," ").title(),
                marker_color=ACCENT[i % len(ACCENT)],
                boxmean=True,
                line_width=1.5
            ))
        fig_box.update_layout(**PLOT_TEMPLATE["layout"], height=350,
                               title="", showlegend=False,
                               yaxis_title="Unit Price (₹)")
        apply_axis_style(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_p2:
        st.markdown('<div class="section-header">Price vs Competitor Average</div>', unsafe_allow_html=True)
        fig_comp = px.scatter(
            filtered.sample(min(500, len(filtered))),
            x="comp_avg_price", y="unit_price",
            color="product_category_name",
            size="customers",
            size_max=18,
            color_discrete_sequence=ACCENT,
            labels={"comp_avg_price":"Competitor Avg Price (₹)","unit_price":"Our Price (₹)"},
            opacity=0.75
        )
        max_val = max(filtered["unit_price"].max(), filtered["comp_avg_price"].max())
        fig_comp.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                           line=dict(color="#ef4444", width=1.5, dash="dash"))
        fig_comp.update_layout(**PLOT_TEMPLATE["layout"], height=350,
                                legend=mk_legend(orientation="h", y=-0.22, font=dict(size=11)),
                                title="")
        apply_axis_style(fig_comp)
        st.plotly_chart(fig_comp, use_container_width=True)

    col_p3, col_p4 = st.columns(2)

    with col_p3:
        st.markdown('<div class="section-header">Freight Price vs Unit Price</div>', unsafe_allow_html=True)
        fig_fr = px.scatter(
            filtered,
            x="freight_price", y="unit_price",
            color="product_category_name",
            color_discrete_sequence=ACCENT,
            labels={"freight_price":"Freight Price (₹)","unit_price":"Unit Price (₹)"},
            opacity=0.6
        )
        # Manual numpy trendline (no statsmodels needed)
        _x = filtered["freight_price"].dropna().values
        _y = filtered.loc[filtered["freight_price"].notna(), "unit_price"].values
        _m, _b = np.polyfit(_x, _y, 1)
        _xs = np.linspace(_x.min(), _x.max(), 100)
        fig_fr.add_trace(go.Scatter(
            x=_xs, y=_m * _xs + _b,
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="Trend",
            showlegend=True
        ))
        fig_fr.update_layout(**PLOT_TEMPLATE["layout"], height=280,
                              legend=mk_legend(orientation="h", y=-0.3, font=dict(size=10)),
                              title="")
        apply_axis_style(fig_fr)
        st.plotly_chart(fig_fr, use_container_width=True)

    with col_p4:
        st.markdown('<div class="section-header">Avg Price Trend by Month</div>', unsafe_allow_html=True)
        monthly = filtered.groupby(["month","year"])["unit_price"].mean().reset_index()
        monthly["period"] = monthly["month"].astype(str) + "/" + monthly["year"].astype(str)
        monthly = monthly.sort_values(["year","month"])
        fig_m = go.Figure()
        for yr in sorted(monthly["year"].unique()):
            sub = monthly[monthly["year"]==yr]
            fig_m.add_trace(go.Scatter(
                x=sub["month"], y=sub["unit_price"],
                name=str(yr), mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=6)
            ))
        fig_m.update_layout(**PLOT_TEMPLATE["layout"], height=280,
                             yaxis_title="Avg Unit Price (₹)", title="")
        fig_m.update_xaxes(tickmode="array", tickvals=list(range(1,13)),
                           ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                     "Jul","Aug","Sep","Oct","Nov","Dec"])
        apply_axis_style(fig_m)
        st.plotly_chart(fig_m, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["unit_price","total_price","freight_price","qty","customers",
                "product_score","comp_avg_price","price_vs_comp","product_weight_g"]
    corr = filtered[num_cols].corr().round(2)
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=[[0,"#dbeafe"],[0.5,"#f6f8fc"],[1,"#6d28d9"]],
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig_heat.update_layout(**PLOT_TEMPLATE["layout"], height=380,
                            coloraxis_colorbar=dict(thickness=14),
                            title="")
    fig_heat.update_traces(textfont_size=11)
    apply_axis_style(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTION
# ════════════════════════════════════════════════════════════
with tab3:

    # ── Train ONE global model on full dataset ────────────────────────────────
    @st.cache_resource
    def train_global_model(model_name):
        _df = load_data()
        # encode category as integer so model can differentiate products
        _df["cat_code"] = _df["product_category_name"].astype("category").cat.codes
        feat = FEATURES + ["cat_code"]
        X_all = _df[feat].dropna()
        y_all = _df.loc[X_all.index, TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        _model_map = {
            "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Linear Regression":  LinearRegression(),
            "Decision Tree":      DecisionTreeRegressor(max_depth=10, random_state=42),
            "SVM":                SVR(kernel="rbf"),
        }
        mdl = _model_map[model_name]
        mdl.fit(X_tr_s, y_tr)
        y_pred_te = mdl.predict(X_te_s)
        _r2   = r2_score(y_te, y_pred_te)
        _mae  = mean_absolute_error(y_te, y_pred_te)
        _rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
        # category code lookup
        cat_codes = dict(enumerate(_df["product_category_name"].astype("category").cat.categories))
        cat_to_code = {v: k for k, v in cat_codes.items()}
        return mdl, sc, _r2, _mae, _rmse, y_te, y_pred_te, cat_to_code, feat

    model, scaler, r2, mae, rmse, y_test, y_pred, CAT_TO_CODE, FEAT_WITH_CAT = \
        train_global_model(model_choice)

    # ── Model metrics row ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score",     f"{r2:.4f}",       help="Coefficient of determination")
    m2.metric("MAE",          f"₹{mae:.2f}",     help="Mean absolute error")
    m3.metric("RMSE",         f"₹{rmse:.2f}",    help="Root mean squared error")
    m4.metric("Test Samples", f"{len(y_test):,}")

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.markdown('<div class="section-header">Actual vs Predicted Prices</div>', unsafe_allow_html=True)
        lim = [min(float(y_test.min()), float(y_pred.min()))-5,
               max(float(y_test.max()), float(y_pred.max()))+5]
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=y_test.values, y=y_pred,
            mode="markers",
            marker=dict(color="#3b82f6", size=5, opacity=0.55),
            name="Predictions"
        ))
        fig_ap.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="#ef4444", dash="dash", width=1.5),
            name="Perfect Fit"
        ))
        fig_ap.update_layout(**PLOT_TEMPLATE["layout"], height=320,
                              xaxis_title="Actual Price (₹)", yaxis_title="Predicted Price (₹)", title="")
        apply_axis_style(fig_ap)
        st.plotly_chart(fig_ap, use_container_width=True)

    with col_ml2:
        st.markdown('<div class="section-header">Prediction Residuals</div>', unsafe_allow_html=True)
        residuals = y_test.values - y_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode="markers",
            marker=dict(color="#8b5cf6", size=5, opacity=0.55),
            name="Residuals"
        ))
        fig_res.add_hline(y=0, line=dict(color="#ef4444", width=1.5, dash="dash"))
        fig_res.update_layout(**PLOT_TEMPLATE["layout"], height=320,
                               xaxis_title="Predicted Price (₹)", yaxis_title="Residual (₹)", title="")
        apply_axis_style(fig_res)
        st.plotly_chart(fig_res, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        imp = pd.Series(model.feature_importances_, index=FEAT_WITH_CAT).sort_values(ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation="h",
            marker=dict(color=imp.values,
                        colorscale=[[0,"#dbe3f3"],[0.5,"#3b82f6"],[1,"#8b5cf6"]],
                        showscale=False),
            text=[f"{v:.3f}" for v in imp.values],
            textposition="outside",
            textfont=dict(color="#6b7280", size=11)
        ))
        fig_imp.update_layout(**PLOT_TEMPLATE["layout"], height=340, title="",
                               xaxis_title="Importance Score")
        apply_axis_style(fig_imp)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Predict panel ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Predict Unit Price — per Product</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#475569; margin-bottom:16px;">
        Select <b style="color:#2563eb">All Categories</b> to get a predicted unit price for every
        product in the dataset under your chosen conditions, or pick a specific category for a
        focused prediction. Each row shows a <b style="color:#2563eb">separate predicted price</b>
        for that individual product.
    </div>
    """, unsafe_allow_html=True)

    # ── Shared inputs ─────────────────────────────────────────────────────────
    pi1, pi2, pi3, pi4, pi5 = st.columns(5)
    with pi1:
        qty_in         = st.number_input("Quantity",          value=10.0,  min_value=0.0, key="p_qty")
        total_price_in = st.number_input("Total Price (₹)",   value=800.0, min_value=0.0, key="p_total")
    with pi2:
        freight_in     = st.number_input("Freight Price (₹)", value=18.0,  min_value=0.0, key="p_freight")
        weight_in      = st.number_input("Product Weight (g)",value=400.0, min_value=0.0, key="p_weight")
    with pi3:
        score_in       = st.number_input("Product Score",     value=4.1,   min_value=0.0, max_value=5.0, step=0.1, key="p_score")
        customers_in   = st.number_input("Customers",         value=60.0,  min_value=0.0, key="p_cust")
    with pi4:
        weekday_in     = st.number_input("Weekday (1-7)",     value=3.0,   min_value=1.0, max_value=7.0, key="p_wday")
        holiday_in     = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="p_hol")
    with pi5:
        month_in       = st.number_input("Month (1-12)",      value=6.0,   min_value=1.0, max_value=12.0, key="p_month")
        year_in        = st.number_input("Year",              value=2024.0, key="p_year")

    if st.button("⚡  Predict Unit Price for All Products", use_container_width=True):

        # Decide which categories to predict for
        if selected_cat_label == "All Categories":
            predict_cats = CATEGORIES
        else:
            predict_cats = selected_cats

        results = []
        for cat in predict_cats:
            cat_code = CAT_TO_CODE.get(cat, 0)
            # Get all unique products in this category
            products_in_cat = df[df["product_category_name"] == cat]["product_id"].unique()
            for pid in products_in_cat:
                row = np.array([[qty_in, total_price_in, freight_in, weight_in,
                                 score_in, customers_in, weekday_in, holiday_in,
                                 month_in, float(year_in), float(cat_code)]])
                pred_price = model.predict(scaler.transform(row))[0]
                # Get actual avg price for this product from data
                actual_avg = df[df["product_id"] == pid]["unit_price"].mean()
                diff = pred_price - actual_avg
                results.append({
                    "Product ID":        pid,
                    "Category":          CAT_LABELS.get(cat, cat.replace("_"," ").title()),
                    "Predicted Price (₹)": round(pred_price, 2),
                    "Actual Avg Price (₹)": round(actual_avg, 2),
                    "Difference (₹)":    round(diff, 2),
                    "Status":            "↑ Higher" if diff > 0.5 else ("↓ Lower" if diff < -0.5 else "≈ Same"),
                })

        results_df = pd.DataFrame(results).sort_values("Category")

        # ── Summary KPIs ──────────────────────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;">
                <div class="kpi-label">Total Products</div>
                <div class="kpi-value">{len(results_df)}</div>
                <div class="kpi-delta">across {len(predict_cats)} categories</div>
            </div>""", unsafe_allow_html=True)
        with rk2:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;">
                <div class="kpi-label">Avg Predicted Price</div>
                <div class="kpi-value">₹{results_df['Predicted Price (₹)'].mean():.2f}</div>
                <div class="kpi-delta">per unit</div>
            </div>""", unsafe_allow_html=True)
        with rk3:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;">
                <div class="kpi-label">Min Predicted</div>
                <div class="kpi-value">₹{results_df['Predicted Price (₹)'].min():.2f}</div>
                <div class="kpi-delta">{results_df.loc[results_df['Predicted Price (₹)'].idxmin(),'Product ID']}</div>
            </div>""", unsafe_allow_html=True)
        with rk4:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;">
                <div class="kpi-label">Max Predicted</div>
                <div class="kpi-value">₹{results_df['Predicted Price (₹)'].max():.2f}</div>
                <div class="kpi-delta">{results_df.loc[results_df['Predicted Price (₹)'].idxmax(),'Product ID']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Bar chart: predicted price per product colored by category ────────
        st.markdown('<div class="section-header">Predicted Unit Price — Each Product</div>', unsafe_allow_html=True)

        fig_bar = go.Figure()
        for i, cat in enumerate(results_df["Category"].unique()):
            sub = results_df[results_df["Category"] == cat]
            fig_bar.add_trace(go.Bar(
                x=sub["Product ID"],
                y=sub["Predicted Price (₹)"],
                name=cat,
                marker_color=ACCENT[i % len(ACCENT)],
                text=[f"₹{v:.0f}" for v in sub["Predicted Price (₹)"]],
                textposition="outside",
                textfont=dict(size=10, color="#6b7280"),
            ))
        fig_bar.update_layout(
            **PLOT_TEMPLATE["layout"],
            height=380,
            title="",
            barmode="group",
            xaxis_title="Product ID",
            yaxis_title="Predicted Unit Price (₹)",
            legend=mk_legend(orientation="h", y=1.08, font=dict(size=11)),
        )
        apply_axis_style(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Actual vs Predicted comparison chart ─────────────────────────────
        st.markdown('<div class="section-header">Predicted vs Actual Avg Price — Per Product</div>', unsafe_allow_html=True)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            x=results_df["Product ID"],
            y=results_df["Actual Avg Price (₹)"],
            name="Actual Avg Price",
            marker_color="#dbeafe",
        ))
        fig_cmp.add_trace(go.Bar(
            x=results_df["Product ID"],
            y=results_df["Predicted Price (₹)"],
            name="Predicted Price",
            marker_color="#3b82f6",
        ))
        fig_cmp.update_layout(
            **PLOT_TEMPLATE["layout"],
            height=340,
            title="",
            barmode="group",
            xaxis_title="Product ID",
            yaxis_title="Price (₹)",
            legend=mk_legend(orientation="h", y=1.08, font=dict(size=11)),
        )
        apply_axis_style(fig_cmp)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Full results table ────────────────────────────────────────────────
        st.markdown('<div class="section-header">Full Prediction Results Table</div>', unsafe_allow_html=True)
        st.caption(f"Each row = one product · Predicted price = per single unit · Model: {model_choice} · R² = {r2:.4f}")
        st.dataframe(
            results_df.reset_index(drop=True),
            use_container_width=True,
            height=320,
        )

        # Download button
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Predictions as CSV",
            data=csv_bytes,
            file_name="predicted_prices.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ════════════════════════════════════════════════════════════
with tab4:
    col_e1, col_e2 = st.columns([3, 1])

    with col_e2:
        st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
        stat_col = st.selectbox("Column", [TARGET] + FEATURES)
        series = filtered[stat_col].dropna()
        st.metric("Mean",   f"{series.mean():.2f}")
        st.metric("Median", f"{series.median():.2f}")
        st.metric("Std Dev",f"{series.std():.2f}")
        st.metric("Min",    f"{series.min():.2f}")
        st.metric("Max",    f"{series.max():.2f}")

    with col_e1:
        st.markdown(f'<div class="section-header">Distribution — {stat_col}</div>', unsafe_allow_html=True)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=filtered[stat_col], nbinsx=40,
            marker_color="#3b82f6",
            marker_line_width=0.3,
            marker_line_color="#f6f8fc",
            name="Frequency"
        ))
        fig_dist.update_layout(**PLOT_TEMPLATE["layout"], height=260,
                                xaxis_title=stat_col, yaxis_title="Count", title="")
        apply_axis_style(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown('<div class="section-header">Filtered Dataset</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered.reset_index(drop=True).head(200),
        use_container_width=True,
        height=320,
    )
    st.caption(f"Showing first 200 of {len(filtered):,} filtered rows")

# ════════════════════════════════════════════════════════════
# TAB 5 — WHAT-IF PRICING SIMULATOR
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div style="padding:4px 0 18px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:15px; font-weight:600; color:#334155;">
            Drag the sliders to instantly see how each variable affects the predicted unit price.
        </div>
        <div style="font-size:12px; color:#475569; margin-top:4px;">
            The model re-predicts live as you adjust inputs — no button click needed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Train model for simulator (Random Forest default) ────────────────────
    @st.cache_resource
    def get_simulator_model(df):
        X_sim = df[FEATURES].dropna()
        y_sim = df.loc[X_sim.index, TARGET]
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_sim)
        m = RandomForestRegressor(n_estimators=100, random_state=42)
        m.fit(X_sc, y_sim)
        return m, sc, X_sim

    sim_model, sim_scaler, X_sim_full = get_simulator_model(df)

    # ── Slider bounds from actual data ───────────────────────────────────────
    def col_range(col):
        return float(df[col].min()), float(df[col].max()), float(df[col].mean())

    qty_min,      qty_max,      qty_def      = col_range("qty")
    total_min,    total_max,    total_def    = col_range("total_price")
    freight_min,  freight_max,  freight_def  = col_range("freight_price")
    weight_min,   weight_max,   weight_def   = col_range("product_weight_g")
    score_min,    score_max,    score_def    = col_range("product_score")
    cust_min,     cust_max,     cust_def     = col_range("customers")
    comp_min = float(df[["comp_1","comp_2","comp_3"]].min().min())
    comp_max = float(df[["comp_1","comp_2","comp_3"]].max().max())
    comp_def = float(df[["comp_1","comp_2","comp_3"]].mean().mean())

    # ── Layout ───────────────────────────────────────────────────────────────
    sim_left, sim_right = st.columns([1, 1])

    with sim_left:
        st.markdown('<div class="section-header">📦 Product & Order Inputs</div>', unsafe_allow_html=True)

        s_qty = st.slider(
            "Quantity Sold", min_value=int(qty_min), max_value=int(qty_max),
            value=int(qty_def), step=1,
            help="Number of units sold in the period"
        )
        s_total = st.slider(
            "Total Price (₹)", min_value=float(round(total_min,2)), max_value=float(round(total_max,2)),
            value=float(round(total_def,2)), step=10.0,
            help="Total order value"
        )
        s_freight = st.slider(
            "Freight Price (₹)", min_value=float(round(freight_min,2)), max_value=float(round(freight_max,2)),
            value=float(round(freight_def,2)), step=0.5,
            help="Shipping cost per unit"
        )
        s_weight = st.slider(
            "Product Weight (g)", min_value=float(round(weight_min,0)), max_value=float(round(weight_max,0)),
            value=float(round(weight_def,0)), step=50.0,
            help="Weight of the product in grams"
        )
        s_score = st.slider(
            "Product Score", min_value=float(round(score_min,1)), max_value=float(round(score_max,1)),
            value=float(round(score_def,1)), step=0.1,
            help="Customer review score (0–5)"
        )
        s_customers = st.slider(
            "Customers", min_value=int(cust_min), max_value=int(cust_max),
            value=int(cust_def), step=5,
            help="Number of customers in the period"
        )

    with sim_right:
        st.markdown('<div class="section-header">🏪 Market & Competitor Inputs</div>', unsafe_allow_html=True)

        s_comp1 = st.slider(
            "Competitor 1 Price (₹)", min_value=float(round(comp_min,2)), max_value=float(round(comp_max,2)),
            value=float(round(comp_def,2)), step=1.0,
            help="Competitor 1 unit price"
        )
        s_comp2 = st.slider(
            "Competitor 2 Price (₹)", min_value=float(round(comp_min,2)), max_value=float(round(comp_max,2)),
            value=float(round(comp_def * 0.95, 2)), step=1.0,
            help="Competitor 2 unit price"
        )
        s_comp3 = st.slider(
            "Competitor 3 Price (₹)", min_value=float(round(comp_min,2)), max_value=float(round(comp_max,2)),
            value=float(round(comp_def * 1.05, 2)), step=1.0,
            help="Competitor 3 unit price"
        )

        st.markdown('<div class="section-header" style="margin-top:16px;">📅 Time Context</div>', unsafe_allow_html=True)

        s_weekday = st.slider("Weekday (1=Mon … 7=Sun)", min_value=1, max_value=7, value=3, step=1)
        s_month   = st.slider("Month", min_value=1, max_value=12, value=6, step=1)
        s_year    = st.selectbox("Year", options=sorted(df["year"].unique().tolist()), index=0)
        s_holiday = st.radio("Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x else "No",
                              horizontal=True, index=0)

    # ── Live prediction ───────────────────────────────────────────────────────
    sim_inp = np.array([[
        s_qty, s_total, s_freight, s_weight,
        s_score, s_customers, s_weekday, s_holiday,
        s_month, float(s_year)
    ]])
    sim_inp_scaled = sim_scaler.transform(sim_inp)
    sim_pred = sim_model.predict(sim_inp_scaled)[0]

    # Competitor context
    avg_comp = (s_comp1 + s_comp2 + s_comp3) / 3.0
    delta_vs_comp = sim_pred - avg_comp
    delta_color   = "#10b981" if delta_vs_comp < 0 else "#ef4444"
    delta_label   = "below" if delta_vs_comp < 0 else "above"
    delta_icon    = "↓" if delta_vs_comp < 0 else "↑"

    st.markdown("---")
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-label">Predicted Unit Price</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:38px; font-weight:700;
                        background:linear-gradient(90deg,#3b82f6,#8b5cf6);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                ₹{sim_pred:.2f}
            </div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-label">Avg Competitor Price</div>
            <div class="kpi-value">₹{avg_comp:.2f}</div>
            <div class="kpi-delta" style="color:#6b7280;">across 3 competitors</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-label">Price vs Competitors</div>
            <div class="kpi-value" style="color:{delta_color};">
                {delta_icon} ₹{abs(delta_vs_comp):.2f}
            </div>
            <div class="kpi-delta" style="color:{delta_color};">{delta_label} competitor avg</div>
        </div>""", unsafe_allow_html=True)

    with r4:
        margin_pct = ((sim_pred - s_freight) / sim_pred * 100) if sim_pred > 0 else 0
        m_color = "#10b981" if margin_pct > 50 else "#f59e0b" if margin_pct > 20 else "#ef4444"
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-label">Est. Margin (after freight)</div>
            <div class="kpi-value" style="color:{m_color};">{margin_pct:.1f}%</div>
            <div class="kpi-delta" style="color:{m_color};">freight = ₹{s_freight:.0f}</div>
        </div>""", unsafe_allow_html=True)

    # ── Sensitivity chart — show how freight & qty affect predicted price ─────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sensitivity Analysis — Freight vs Quantity</div>', unsafe_allow_html=True)

    freight_range = np.linspace(freight_min, freight_max, 30)
    qty_range     = np.linspace(max(1, qty_min), qty_max, 30)

    base_row = [s_qty, s_total, s_freight, s_weight,
                s_score, s_customers, s_weekday, s_holiday, s_month, float(s_year)]

    freight_preds, qty_preds = [], []
    for fv in freight_range:
        row = base_row.copy(); row[2] = fv
        freight_preds.append(sim_model.predict(sim_scaler.transform([row]))[0])
    for qv in qty_range:
        row = base_row.copy(); row[0] = qv
        qty_preds.append(sim_model.predict(sim_scaler.transform([row]))[0])

    fig_sens = make_subplots(rows=1, cols=2,
                              subplot_titles=["Freight Price → Predicted Price",
                                              "Quantity Sold → Predicted Price"])
    fig_sens.add_trace(go.Scatter(
        x=freight_range, y=freight_preds,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=4),
        name="vs Freight"
    ), row=1, col=1)
    fig_sens.add_vline(x=s_freight, line=dict(color="#ef4444", dash="dash", width=1.5), row=1, col=1)

    fig_sens.add_trace(go.Scatter(
        x=qty_range, y=qty_preds,
        mode="lines+markers",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=4),
        name="vs Qty"
    ), row=1, col=2)
    fig_sens.add_vline(x=s_qty, line=dict(color="#ef4444", dash="dash", width=1.5), row=1, col=2)

    fig_sens.update_layout(
        **PLOT_TEMPLATE["layout"],
        height=300,
        showlegend=False,
        title=""
    )
    fig_sens.update_annotations(font=dict(color="#6b7280", size=12))
    apply_axis_style(fig_sens)
    st.plotly_chart(fig_sens, use_container_width=True)

    st.caption("Red dashed line = your current slider value. Curve shows predicted price as that variable sweeps its full range.")


# ════════════════════════════════════════════════════════════
# TAB 6 — PRODUCT DEEP DIVE
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    <div style="padding:4px 0 18px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:15px; font-weight:600; color:#334155;">
            Select a category, then click a product to drill into its full pricing history and competitive position.
        </div>
    </div>
    """, unsafe_allow_html=True)

    dd_col1, dd_col2 = st.columns([1, 3])

    with dd_col1:
        st.markdown('<div class="section-header">Select Category</div>', unsafe_allow_html=True)
        dd_cat_label = st.selectbox(
            "", [CAT_LABELS.get(c, c.replace("_"," ").title()) for c in CATEGORIES],
            label_visibility="collapsed", key="dd_cat"
        )
        dd_cat = CATEGORIES[[CAT_LABELS.get(c, c.replace("_"," ").title()) for c in CATEGORIES].index(dd_cat_label)]

        cat_products = sorted(df[df["product_category_name"] == dd_cat]["product_id"].unique().tolist())

        st.markdown('<div class="section-header" style="margin-top:14px;">Select Product</div>', unsafe_allow_html=True)
        dd_product = st.selectbox(
            "", cat_products,
            label_visibility="collapsed", key="dd_product"
        )

        # Product summary card
        prod_df = df[df["product_id"] == dd_product].sort_values("month_year")
        if not prod_df.empty:
            avg_p  = prod_df["unit_price"].mean()
            min_p  = prod_df["unit_price"].min()
            max_p  = prod_df["unit_price"].max()
            n_mo   = len(prod_df)
            avg_sc = prod_df["product_score"].mean()
            avg_cu = prod_df["customers"].mean()

            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e5e9f2; border-radius:10px;
                        padding:14px 16px; margin-top:12px; font-size:12px; color:#64748b; line-height:2;">
                <span style="color:#2563eb; font-weight:600;">ID:</span> {dd_product}<br>
                <span style="color:#2563eb; font-weight:600;">Months:</span> {n_mo}<br>
                <span style="color:#2563eb; font-weight:600;">Avg Price:</span> ₹{avg_p:.2f}<br>
                <span style="color:#2563eb; font-weight:600;">Range:</span> ₹{min_p:.2f} – ₹{max_p:.2f}<br>
                <span style="color:#2563eb; font-weight:600;">Avg Score:</span> {avg_sc:.2f} ⭐<br>
                <span style="color:#2563eb; font-weight:600;">Avg Customers:</span> {avg_cu:.0f}
            </div>
            """, unsafe_allow_html=True)

    with dd_col2:
        if prod_df.empty:
            st.info("No data found for this product.")
        else:
            # ── Chart 1: Unit price vs competitor prices over time ────────────
            st.markdown('<div class="section-header">Unit Price vs Competitors Over Time</div>', unsafe_allow_html=True)
            fig_dd1 = go.Figure()
            fig_dd1.add_trace(go.Scatter(
                x=prod_df["month_year"], y=prod_df["unit_price"],
                mode="lines+markers", name="Our Price",
                line=dict(color="#3b82f6", width=2.5),
                marker=dict(size=6)
            ))
            fig_dd1.add_trace(go.Scatter(
                x=prod_df["month_year"], y=prod_df["comp_1"],
                mode="lines+markers", name="Comp 1",
                line=dict(color="#8b5cf6", width=1.5, dash="dot"),
                marker=dict(size=4)
            ))
            fig_dd1.add_trace(go.Scatter(
                x=prod_df["month_year"], y=prod_df["comp_2"],
                mode="lines+markers", name="Comp 2",
                line=dict(color="#10b981", width=1.5, dash="dot"),
                marker=dict(size=4)
            ))
            fig_dd1.add_trace(go.Scatter(
                x=prod_df["month_year"], y=prod_df["comp_3"],
                mode="lines+markers", name="Comp 3",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
                marker=dict(size=4)
            ))
            fig_dd1.add_trace(go.Scatter(
                x=prod_df["month_year"], y=prod_df["lag_price"],
                mode="lines", name="Lag Price",
                line=dict(color="#64748b", width=1, dash="dash"),
            ))
            fig_dd1.update_layout(
                **PLOT_TEMPLATE["layout"],
                height=280,
                title="",
                legend=mk_legend(orientation="h", y=1.12, font=dict(size=11)),
                yaxis_title="Price (₹)"
            )
            apply_axis_style(fig_dd1)
            st.plotly_chart(fig_dd1, use_container_width=True)

            # ── Chart 2: Quantity + Customers over time ───────────────────────
            c2a, c2b = st.columns(2)

            with c2a:
                st.markdown('<div class="section-header">Quantity Sold Over Time</div>', unsafe_allow_html=True)
                fig_dd2 = go.Figure()
                fig_dd2.add_trace(go.Bar(
                    x=prod_df["month_year"], y=prod_df["qty"],
                    marker=dict(
                        color=prod_df["qty"],
                        colorscale=[[0,"#dbe3f3"],[0.5,"#3b82f6"],[1,"#8b5cf6"]],
                        showscale=False
                    ),
                    name="Qty"
                ))
                fig_dd2.update_layout(**PLOT_TEMPLATE["layout"], height=220,
                                       title="", yaxis_title="Units Sold")
                apply_axis_style(fig_dd2)
                st.plotly_chart(fig_dd2, use_container_width=True)

            with c2b:
                st.markdown('<div class="section-header">Customers Over Time</div>', unsafe_allow_html=True)
                fig_dd3 = go.Figure()
                fig_dd3.add_trace(go.Scatter(
                    x=prod_df["month_year"], y=prod_df["customers"],
                    fill="tozeroy", mode="lines",
                    line=dict(color="#10b981", width=2),
                    fillcolor="rgba(52,211,153,0.1)",
                    name="Customers"
                ))
                fig_dd3.update_layout(**PLOT_TEMPLATE["layout"], height=220,
                                       title="", yaxis_title="Customers")
                apply_axis_style(fig_dd3)
                st.plotly_chart(fig_dd3, use_container_width=True)

            # ── Chart 3: Price advantage over time ────────────────────────────
            st.markdown('<div class="section-header">Price Advantage vs Competitor Avg</div>', unsafe_allow_html=True)
            prod_df = prod_df.copy()
            prod_df["comp_avg"] = prod_df[["comp_1","comp_2","comp_3"]].mean(axis=1)
            prod_df["advantage"] = prod_df["comp_avg"] - prod_df["unit_price"]

            colors = ["#10b981" if v >= 0 else "#ef4444" for v in prod_df["advantage"]]
            fig_dd4 = go.Figure()
            fig_dd4.add_trace(go.Bar(
                x=prod_df["month_year"],
                y=prod_df["advantage"],
                marker_color=colors,
                name="Price Advantage"
            ))
            fig_dd4.add_hline(y=0, line=dict(color="#64748b", width=1))
            fig_dd4.update_layout(
                **PLOT_TEMPLATE["layout"],
                height=220,
                title="",
                yaxis_title="Advantage (₹)",
            )
            fig_dd4.update_layout(
                annotations=[dict(
                    text="Green = cheaper than competitors · Red = more expensive",
                    x=0, y=-0.22, xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=10, color="#475569"),
                    align="left"
                )]
            )
            apply_axis_style(fig_dd4)
            st.plotly_chart(fig_dd4, use_container_width=True)

            # ── Raw monthly table ─────────────────────────────────────────────
            with st.expander("📋 Raw Monthly Data for this Product"):
                show_cols = ["month_year","unit_price","comp_1","comp_2","comp_3",
                             "qty","customers","freight_price","product_score","lag_price"]
                st.dataframe(
                    prod_df[show_cols].sort_values("month_year").reset_index(drop=True),
                    use_container_width=True,
                    height=260
                )