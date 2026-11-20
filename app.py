import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SEO Analytics BI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (executive dark theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Background */
    .stApp { background: #0d1117; color: #e6edf3; }

    /* Hide hamburger / footer */
    #MainMenu, footer { visibility: hidden; }

    /* Section headers */
    h1 { font-family: 'Space Grotesk', sans-serif !important;
         font-size: 1.9rem !important; color: #58a6ff !important; }
    h3 { font-family: 'Space Grotesk', sans-serif !important;
         color: #c9d1d9 !important; font-size: 1.1rem !important; margin-bottom: 0 !important; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 10px;
    }
    .kpi-label { font-size: 0.75rem; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; }
    .kpi-value { font-size: 2rem; font-weight: 600; color: #58a6ff; line-height: 1.2; }
    .kpi-delta { font-size: 0.78rem; color: #3fb950; margin-top: 2px; }

    /* Insight box */
    .insight-box {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 6px 0 14px 0;
        font-size: 0.82rem;
        color: #8b949e;
        line-height: 1.55;
    }
    .insight-box b { color: #c9d1d9; }

    /* Section divider */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #58a6ff;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
        margin: 28px 0 14px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; border-bottom: 1px solid #21262d; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border-radius: 8px 8px 0 0;
        color: #8b949e; padding: 8px 18px; font-size: 0.85rem; border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] { background: #21262d; color: #58a6ff !important; border-bottom: 2px solid #58a6ff; }

    /* Plotly chart container */
    .js-plotly-plot { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE (consistent across all charts)
# ─────────────────────────────────────────────
CHART_TEMPLATE = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9", family="DM Sans"),
    title=dict(font=dict(color="#e6edf3", size=14)),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    coloraxis_colorbar=dict(bgcolor="#161b22", tickcolor="#8b949e")
)

def apply_theme(fig, height=360, margin=None, legend_override=None, showlegend=None, **kwargs):
    """Apply the dark theme to any Plotly figure without keyword conflicts."""
    if margin is None:
        margin = dict(t=20, b=20, l=10, r=10)
    legend = legend_override if legend_override else CHART_TEMPLATE["legend"]
    update_args = dict(
        paper_bgcolor=CHART_TEMPLATE["paper_bgcolor"],
        plot_bgcolor=CHART_TEMPLATE["plot_bgcolor"],
        font=CHART_TEMPLATE["font"],
        title=CHART_TEMPLATE["title"],
        legend=legend,
        coloraxis_colorbar=CHART_TEMPLATE["coloraxis_colorbar"],
        height=height,
        margin=margin,
        **kwargs
    )
    if showlegend is not None:
        update_args["showlegend"] = showlegend
    fig.update_layout(**update_args)
    fig.update_xaxes(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e")
    fig.update_yaxes(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e")
    return fig
PALETTE = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]

# ─────────────────────────────────────────────
# LOAD + CLEAN DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv").dropna()

    # ── DERIVED FEATURES ──────────────────────────────────────────────────
    # 1. Engagement Score  → quality of visit (high time on page, low bounce)
    df["Engagement Score"] = (
        df["Time on Page"] * (1 - df["Bounce Rate"])
    ).round(4)

    # 2. User Value Score  → business worth of the user
    df["User Value Score"] = (
        df["Page Views"] * (1 + df["Previous Visits"]) * (1 - df["Bounce Rate"])
    ).round(4)

    # 3. Session Quality Ratio  → how much of session time was productive
    df["Session Quality"] = (
        df["Time on Page"] / (df["Session Duration"] + 1e-6)
    ).round(4)

    # 4. Converted  → binary flag (Conversion Rate < 1 means partial / no full conversion)
    df["Converted"] = (df["Conversion Rate"] >= 1.0).astype(int)

    # 5. Return visitor flag
    df["Is Returning"] = (df["Previous Visits"] > 0).astype(int)

    return df

df = load_data()

# ─────────────────────────────────────────────
# K-MEANS CLUSTERING  (named clusters)
# ─────────────────────────────────────────────
FEATURES = ["Page Views", "Session Duration", "Bounce Rate",
            "Time on Page", "Previous Visits", "Engagement Score"]

scaler = StandardScaler()
scaled = scaler.fit_transform(df[FEATURES])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster_raw"] = kmeans.fit_predict(scaled)

# Rank clusters by mean Engagement Score → meaningful labels
eng_by_cluster = df.groupby("Cluster_raw")["Engagement Score"].mean().sort_values()
label_map = {eng_by_cluster.index[0]: "Low Engagement",
             eng_by_cluster.index[1]: "Mid Engagement",
             eng_by_cluster.index[2]: "High Engagement"}
df["Segment"] = df["Cluster_raw"].map(label_map)

SEG_COLORS = {"Low Engagement": "#f78166",
              "Mid Engagement": "#ffa657",
              "High Engagement": "#3fb950"}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; padding:10px 0 0 0;'>"
    "📊 SEO Analytics — Executive Dashboard</h1>"
    "<p style='text-align:center; color:#8b949e; font-size:0.85rem; margin-top:2px;'>"
    "Business Intelligence Lab · 2,000 sessions · KMeans segmentation</p>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# ── SECTION 1 · KPI SUMMARY ROW ──────────────
# ─────────────────────────────────────────────
st.markdown("<div class='section-title'>01 — Executive KPIs</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)

def kpi(col, label, value, delta=""):
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-delta'>{delta}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

kpi(c1, "Total Sessions",        f"{len(df):,}")
kpi(c2, "Total Page Views",      f"{int(df['Page Views'].sum()):,}")
kpi(c3, "Avg Bounce Rate",       f"{df['Bounce Rate'].mean():.1%}")
kpi(c4, "Avg Session Duration",  f"{df['Session Duration'].mean():.1f}s")
kpi(c5, "Avg Engagement Score",  f"{df['Engagement Score'].mean():.2f}")
kpi(c6, "High-Value Sessions",   f"{(df['Segment']=='High Engagement').sum():,}",
        f"▲ {(df['Segment']=='High Engagement').mean():.0%} of total")

# ─────────────────────────────────────────────
# TABS  (clean one-page layout)
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Traffic & Trends",
    "🧠  User Segmentation",
    "📊  Engagement Deep Dive",
    "🔗  Feature Analysis"
])

# ══════════════════════════════════════════════
# TAB 1 — TRAFFIC & TRENDS
# ══════════════════════════════════════════════
with tab1:

    col_a, col_b = st.columns(2)

    # ── Chart 1: Traffic Source Volume + Avg Engagement Score (dual axis) ──
    with col_a:
        st.markdown("### Traffic Source — Volume vs Engagement")
        ts = df.groupby("Traffic Source").agg(
            Sessions=("Page Views", "count"),
            Avg_Engagement=("Engagement Score", "mean"),
            Avg_Bounce=("Bounce Rate", "mean")
        ).reset_index().sort_values("Sessions", ascending=False)

        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(
            go.Bar(x=ts["Traffic Source"], y=ts["Sessions"],
                   name="Sessions", marker_color="#58a6ff", opacity=0.85),
            secondary_y=False
        )
        fig_ts.add_trace(
            go.Scatter(x=ts["Traffic Source"], y=ts["Avg_Engagement"],
                       name="Avg Engagement", mode="lines+markers",
                       marker=dict(size=10, color="#3fb950"),
                       line=dict(color="#3fb950", width=2)),
            secondary_y=True
        )
        apply_theme(fig_ts, height=340, margin=dict(t=20, b=20, l=10, r=10),
                    legend_override=dict(bgcolor="#161b22", bordercolor="#30363d",
                                        borderwidth=1, orientation="h", y=-0.2))
        fig_ts.update_yaxes(title_text="Sessions",         secondary_y=False, gridcolor="#21262d")
        fig_ts.update_yaxes(title_text="Avg Engagement",   secondary_y=True,  gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>"
            "<b>Insight:</b> Organic drives the most sessions but compare its Engagement Score "
            "against Paid/Referral. If Referral shows higher engagement, it signals "
            "<b>better audience targeting</b> from that channel — worth increasing budget there."
            "</div>", unsafe_allow_html=True
        )

    # ── Chart 2: Bounce Rate by Traffic Source (horizontal bar) ──
    with col_b:
        st.markdown("### Bounce Rate by Traffic Source")
        ts_sorted = ts.sort_values("Avg_Bounce", ascending=True)
        fig_br = go.Figure(go.Bar(
            x=ts_sorted["Avg_Bounce"], y=ts_sorted["Traffic Source"],
            orientation="h",
            marker=dict(
                color=ts_sorted["Avg_Bounce"],
                colorscale=[[0,"#3fb950"],[0.5,"#ffa657"],[1,"#f78166"]],
                showscale=True,
                colorbar=dict(title="Bounce", thickness=10)
            )
        ))
        apply_theme(fig_br, height=340, margin=dict(t=20, b=20, l=10, r=10),
                    xaxis_title="Average Bounce Rate")
        st.plotly_chart(fig_br, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>"
            "<b>Insight:</b> Channels with the <b>highest bounce rate</b> are wasting ad spend — "
            "users arrive but don't engage. Landing page optimisation or audience re-targeting "
            "is recommended for those channels."
            "</div>", unsafe_allow_html=True
        )


# ══════════════════════════════════════════════
# TAB 2 — USER SEGMENTATION
# ══════════════════════════════════════════════
with tab2:

    col_c, col_d = st.columns(2)

    # ── Chart 4: Scatter — Session Duration vs Engagement Score, coloured by Segment ──
    with col_c:
        st.markdown("### User Segments — Duration × Engagement")
        fig_seg = px.scatter(
            df, x="Session Duration", y="Engagement Score",
            color="Segment", size="Page Views",
            color_discrete_map=SEG_COLORS,
            hover_data=["Bounce Rate", "Traffic Source"],
            opacity=0.7
        )
        apply_theme(fig_seg, height=380, margin=dict(t=10, b=20, l=10, r=10),
                    legend_override=dict(bgcolor="#161b22", bordercolor="#30363d",
                                        borderwidth=1, orientation="h", y=-0.2))
        st.plotly_chart(fig_seg, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>"
            "<b>Insight:</b> <b>High Engagement</b> users cluster at higher engagement "
            "scores regardless of session duration — they are the power users. "
            "<b>Low Engagement</b> users near zero engagement score are candidates for "
            "re-engagement email campaigns."
            "</div>", unsafe_allow_html=True
        )

    # ── Chart 5: Radar — Cluster Profiles ──
    with col_d:
        st.markdown("### Segment Radar — Feature Profile")
        dims = ["Page Views", "Session Duration", "Bounce Rate",
                "Time on Page", "Previous Visits", "Engagement Score"]
        seg_means = df.groupby("Segment")[dims].mean()

        # Normalise 0-1 for radar
        seg_norm = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-9)

        fig_radar = go.Figure()
        for seg, color in SEG_COLORS.items():
            vals = seg_norm.loc[seg].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=dims + [dims[0]],
                fill="toself", fillcolor=color,
                opacity=0.25,
                line=dict(color=color, width=2),
                name=seg
            ))
        fig_radar.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", family="DM Sans"),
            polar=dict(
                bgcolor="#0d1117",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#21262d", tickcolor="#8b949e"),
                angularaxis=dict(gridcolor="#21262d", tickcolor="#8b949e")
            ),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d",
                        borderwidth=1, orientation="h", y=-0.15),
            height=380, margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>"
            "<b>Insight:</b> The radar reveals <b>which dimensions differentiate segments</b>. "
            "If High Engagement leads only on Time on Page and Engagement Score but "
            "not Previous Visits, these are new-but-quality users — prime candidates "
            "for loyalty programmes."
            "</div>", unsafe_allow_html=True
        )

    # ── Chart 6: Segment size + Avg User Value Score ──
    st.markdown("### Segment Summary Table")
    summary = df.groupby("Segment").agg(
        Sessions=("Page Views", "count"),
        Avg_Engagement=("Engagement Score", "mean"),
        Avg_User_Value=("User Value Score", "mean"),
        Avg_Bounce=("Bounce Rate", "mean"),
        Avg_PageViews=("Page Views", "mean"),
        Pct_Returning=("Is Returning", "mean")
    ).round(3).reset_index()
    summary.columns = ["Segment", "Sessions", "Avg Engagement",
                       "Avg User Value", "Avg Bounce Rate",
                       "Avg Page Views", "% Returning"]
    summary["% Returning"] = summary["% Returning"].map("{:.0%}".format)
    def color_scale(col):
        """Apply a blue gradient manually — no matplotlib needed."""
        mn, mx = col.min(), col.max()
        rng = mx - mn if mx != mn else 1
        styles = []
        for val in col:
            intensity = int(40 + 180 * (val - mn) / rng)   # 40–220 brightness
            styles.append(
                f"background-color: rgba(88, 166, 255, {(val - mn)/rng * 0.6:.2f});"
                f"color: #e6edf3;"
            )
        return styles

    styled = summary.style.apply(color_scale, subset=["Avg Engagement", "Avg User Value"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='insight-box'>"
        "<b>Viva tip:</b> User Value Score = Page Views × (1 + Previous Visits) × (1 − Bounce Rate). "
        "This composite score combines <b>volume, loyalty, and quality</b> in one number — "
        "far more meaningful than raw page views alone."
        "</div>", unsafe_allow_html=True
    )


# ══════════════════════════════════════════════
# TAB 3 — ENGAGEMENT DEEP DIVE
# ══════════════════════════════════════════════
with tab3:

    col_e, col_f = st.columns(2)

    # ── Chart 7: Box plot — Engagement Score by Traffic Source ──
    with col_e:
        st.markdown("### Engagement Score Distribution by Source")
        fig_box = px.box(
            df, x="Traffic Source", y="Engagement Score",
            color="Traffic Source",
            color_discrete_sequence=PALETTE,
            points="outliers"
        )
        apply_theme(fig_box, height=360, margin=dict(t=10, b=20, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>"
            "<b>Insight:</b> Wide IQR for a source means <b>inconsistent content quality</b> "
            "for that channel. A tight, high box indicates a reliable, well-targeted audience. "
            "Aim to replicate what makes the top-quartile sessions work."
            "</div>", unsafe_allow_html=True
        )

    # ── Chart 8: Violin — Session Quality by Segment ──
    with col_f:
        st.markdown("### Session Quality Ratio by Segment")
        fig_vio = px.violin(
            df, x="Segment", y="Session Quality",
            color="Segment", color_discrete_map=SEG_COLORS,
            box=True, points=False
        )
        apply_theme(fig_vio, height=360, margin=dict(t=10, b=20, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>"
            "<b>Insight:</b> Session Quality = Time on Page ÷ Session Duration. "
            "A value near 1 means the user spent <b>almost all session time on content</b> "
            "(not loading/idling). High Engagement segment should show highest session quality."
            "</div>", unsafe_allow_html=True
        )

    # ── Chart 9: NEW — Funnel chart ──
    st.markdown("### Conversion Funnel — From Visits to Value")

    total          = len(df)
    returning      = int(df["Is Returning"].sum())
    high_pv        = int((df["Page Views"] >= 5).sum())
    low_bounce     = int((df["Bounce Rate"] < 0.3).sum())
    high_eng       = int((df["Segment"] == "High Engagement").sum())

    funnel_labels  = ["All Sessions", "Returning Users", "5+ Page Views",
                      "Bounce Rate < 30%", "High Engagement"]
    funnel_values  = [total, returning, high_pv, low_bounce, high_eng]
    funnel_colors  = ["#58a6ff", "#79c0ff", "#3fb950", "#ffa657", "#f78166"]

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_labels, x=funnel_values,
        textinfo="value+percent initial",
        marker=dict(color=funnel_colors),
        connector=dict(line=dict(color="#30363d", width=1))
    ))
    apply_theme(fig_funnel, height=340, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown(
        "<div class='insight-box'>"
        "<b>Insight:</b> The funnel shows how many sessions pass each quality gate. "
        "Large drops between stages highlight <b>where value is being lost</b>. "
        "If 'Bounce Rate &lt; 30%' drops sharply, invest in landing-page content quality."
        "</div>", unsafe_allow_html=True
    )


# ══════════════════════════════════════════════
# TAB 4 — FEATURE ANALYSIS
# ══════════════════════════════════════════════
with tab4:

    col_g, col_h = st.columns(2)

    # ── Chart 10: Correlation heatmap (improved — only meaningful numeric features) ──
    with col_g:
        st.markdown("### Feature Correlation Heatmap")
        corr_cols = ["Page Views", "Session Duration", "Bounce Rate",
                     "Time on Page", "Previous Visits",
                     "Engagement Score", "User Value Score", "Session Quality"]
        corr = df[corr_cols].corr().round(2)

        fig_hm = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            color_continuous_midpoint=0
        )
        apply_theme(fig_hm, height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown(
            "<div class='insight-box'>"
            "<b>Insight:</b> Look for <b>red cells</b> (strong positive) between features "
            "like Engagement Score and User Value Score — confirming derived features are "
            "mathematically meaningful. <b>Blue cells</b> (negative) like Bounce Rate × "
            "Engagement confirm our engagement formula penalises bounces correctly."
            "</div>", unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; color:#30363d; font-size:0.75rem; margin-top:30px;'>"
    "BI Lab Dashboard · KMeans n=3 · Derived: Engagement Score · User Value Score · Session Quality"
    "</p>",
    unsafe_allow_html=True
)