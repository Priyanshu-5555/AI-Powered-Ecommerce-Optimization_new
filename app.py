import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------- Page Setup --------------------------
st.set_page_config(
    page_title="AI-Powered E-Commerce Optimization",
    layout="wide",
    page_icon="🧠"
)

# ---------------------- Theming (Dark CSS) ------------------
st.markdown("""
<style>
/* Dark theme */
.reportview-container .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem; background:#07101a; color:#e6eef3;}
body, .stApp {background-color: #07101a; color: #e6eef3;}
.big-metric { font-size: 1.8rem; font-weight: 700; margin-top: -8px; color:#f0f7fb; }
.subtle { color:#9fb0bf; }
.section-title { font-size: 1.4rem; font-weight: 700; margin: 0.2rem 0 0.6rem 0; color:#eaf4ff; }
.pill { display:inline-block; padding:6px 12px; border-radius:999px; background:#103243; color:#9fe0ff; margin-right:6px; border:1px solid #123a49; font-size:0.9rem;}
hr {border: none; height: 1px; background: #12323b; margin: 1rem 0 1.2rem 0;}
.card {border:1px solid #12323b; border-radius:14px; padding:14px 16px; background: rgba(16,50,60,0.45); color:#e6eef3;}
.kpi {border:1px solid #12323b; border-radius:16px; padding:14px 16px; background: rgba(10,30,36,0.5); color:#e6eef3;}
.footer-note {color:#7f98a6; font-size:0.85rem;}
.stMetric {background:#0b2a34; border-radius:14px; padding:8px 10px; border:1px solid #12323b; color:#e6eef3;}
/* Streamlit component backgrounds */
.stButton>button { background-color: #0f3b52; color: #e6eef3; border: 1px solid #144851; }
.stDownloadButton>button { background-color: #0f3b52; color: #e6eef3; border: 1px solid #144851; }
.dataframe, .stDataFrame { background: transparent; color: #e6eef3; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Helper: find report ------------------
POSSIBLE_REPORTS = [
    "AI_Product_Club_Enhanced_Team_Report.pdf",
    "AI_Product_Club_Combined_Report_Team.pdf",
    "AI_Product_Club_Combined_Report.pdf",
    "USE OF AI TOOLS.pdf",
    "USE OF AI TOOLS (1).pdf",
    "Product club Case study compitetion.pdf"
]

def find_first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

REPORT_PATH = find_first_existing(POSSIBLE_REPORTS)

# ---------------------- Preset Scenarios ---------------------
SCENARIOS = {
    "Conservative (Safe)": dict(conv=11.5, add=48, retain=66, nps=74, time_s=6.7),
    "Balanced (Recommended)": dict(conv=13.0, add=51, retain=70, nps=78, time_s=6.0),
    "Aggressive (Ambitious)": dict(conv=14.5, add=55, retain=75, nps=82, time_s=5.4),
}

# Technique deltas are *added* to baseline when selected
TECH_DELTAS = {
    "NLP (Search Optimization)":        dict(conv=+2.0, add=+1.0, retain=+3.0, nps=+4.0, time_s=-1.5),
    "ML Prediction Model":              dict(conv=+1.2, add=+2.5, retain=+4.0, nps=+3.0, time_s=-1.0),
    "Recommendation Engine":            dict(conv=+3.0, add=+4.0, retain=+5.0, nps=+4.0, time_s=-0.6),
    "Sentiment Analysis":               dict(conv=+1.0, add=+1.5, retain=+1.0, nps=+3.0, time_s=-0.3),
    "Dynamic Pricing/Promotions":       dict(conv=+1.8, add=+2.8, retain=+1.5, nps=+1.5, time_s=-0.2),
    "OOS Forecasting + ETA Restock":    dict(conv=+1.6, add=+2.2, retain=+2.5, nps=+2.0, time_s=-0.4),
}

PERSONAS = [
    ("Fast Browser",       "18–40", "Quick, relevant results", "Irrelevant search outcomes"),
    ("Convenience Seeker", "22–40", "Reliable, fast access",    "Frequent out-of-stock items"),
    ("Weekly Planner",     "35–55", "Bulk/family shopping",     "Incomplete orders, app switching"),
    ("Budget Buyer",       "25–45", "Transparent prices (MRP)", "Cannot compare offers due to missing MRP"),
]

# ---------------------- Sidebar Controls ---------------------
st.sidebar.header("⚙️ Controls")

preset = st.sidebar.selectbox("Scenario Preset", list(SCENARIOS.keys()), index=1)
base_conv = st.sidebar.number_input("Baseline Search→Cart (%)", min_value=1.0, max_value=40.0, value=10.0, step=0.1)
base_add  = st.sidebar.number_input("Baseline Add-to-Cart (%)", min_value=1.0, max_value=100.0, value=45.0, step=0.5)
base_ret  = st.sidebar.number_input("Baseline Retention (%)",   min_value=1.0, max_value=100.0, value=60.0, step=0.5)
base_nps  = st.sidebar.number_input("Baseline NPS",             min_value=-100, max_value=100, value=65, step=1)
base_time = st.sidebar.number_input("Baseline Search Time (s)", min_value=1.0, max_value=60.0, value=8.0, step=0.1)

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.subheader("📦 Apply AI Techniques")
chosen_tech = st.sidebar.multiselect("Select one or more techniques to simulate:",
                                     list(TECH_DELTAS.keys()),
                                     default=["NLP (Search Optimization)", "Recommendation Engine"])

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
uploaded_pdf = st.sidebar.file_uploader("Optional: Upload a PDF to attach for Download section", type=["pdf"])
if uploaded_pdf is not None:
    REPORT_PATH = uploaded_pdf.name
    with open(REPORT_PATH, "wb") as f:
        f.write(uploaded_pdf.read())

# ---------------------- Core Simulation Logic ----------------
def apply_preset(baseline, preset_name):
    # returns target metrics (post-AI) for the preset, blended a bit with baseline to remain realistic
    target = SCENARIOS[preset_name]
    # Blend: 60% preset target + 40% baseline lift
    out = dict(
        conv = 0.6*target["conv"] + 0.4*max(baseline["conv"], 0.1),
        add  = 0.6*target["add"]  + 0.4*max(baseline["add"],  0.1),
        retain = 0.6*target["retain"] + 0.4*max(baseline["retain"], 0.1),
        nps  = 0.6*target["nps"]  + 0.4*baseline["nps"],
        time_s = 0.6*target["time_s"] + 0.4*baseline["time_s"],
    )
    return out

def apply_techniques(current, techs):
    cur = current.copy()
    for t in techs:
        d = TECH_DELTAS[t]
        cur["conv"]   += d["conv"]
        cur["add"]    += d["add"]
        cur["retain"] += d["retain"]
        cur["nps"]    += d["nps"]
        cur["time_s"] += d["time_s"]
    # clamp to reasonable bounds
    cur["conv"]   = float(np.clip(cur["conv"],   1, 50))
    cur["add"]    = float(np.clip(cur["add"],    1, 100))
    cur["retain"] = float(np.clip(cur["retain"], 1, 100))
    cur["nps"]    = int(np.clip(cur["nps"],     -100, 100))
    cur["time_s"] = float(np.clip(cur["time_s"], 1, 60))
    return cur

baseline = dict(conv=base_conv, add=base_add, retain=base_ret, nps=base_nps, time_s=base_time)
preset_target = apply_preset(baseline, preset)
final_after = apply_techniques(preset_target, chosen_tech)

# ---------------------- Header / Hero ------------------------
st.markdown("<div class='pill'>Zepto Case Study</div> <div class='pill'>AI Tools: NLP • ML • Recommender • Sentiment</div>", unsafe_allow_html=True)
st.title("🧠 AI-Powered E-Commerce Optimization")
st.write("This interactive web app demonstrates how **AI** can recover a **drop in Search→Cart conversions** by applying "
         "**NLP**, **Recommendation Systems**, **Predictive Models**, and **Sentiment Analysis**—exactly as described in your report.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------------- KPI Row ------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Search→Cart (%)", f"{final_after['conv']:.1f}%", f"{final_after['conv']-baseline['conv']:+.1f}%")
with c2:
    st.metric("Add-to-Cart (%)", f"{final_after['add']:.1f}%", f"{final_after['add']-baseline['add']:+.1f}%")
with c3:
    st.metric("Retention (%)", f"{final_after['retain']:.1f}%", f"{final_after['retain']-baseline['retain']:+.1f}%")
with c4:
    st.metric("NPS", f"{final_after['nps']}", f"{final_after['nps']-baseline['nps']:+d} pts")
with c5:
    st.metric("Avg. Search Time", f"{final_after['time_s']:.1f}s", f"{baseline['time_s']-final_after['time_s']:+.1f}s faster")

# ---------------------- Tabs -------------------------------
tab_overview, tab_problem, tab_ai, tab_dashboard, tab_personas, tab_ethics, tab_files = st.tabs([
    "Overview", "Problem", "AI Analysis", "Dashboard", "Personas", "Ethics", "Downloads"
])

# ---------------------- Overview Tab ------------------------
with tab_overview:
    st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)
    st.write("""
This app mirrors your combined PDF content in an interactive way:

- **Problem:** A ~10% decline in **Search→Cart** conversion in recent weeks.
- **Objective:** Recover and exceed prior performance by using **AI**.
- **Approach:** 
  - **NLP** to improve search intent matching  
  - **Recommendation** for substitutes & bundles  
  - **ML Prediction** for demand & churn signals  
  - **Sentiment Analysis** on reviews/feedback
- **Outcome Targets:** Lift conversion to **13–15%+**, reduce search time, increase **NPS** and **Retention**.
    """)
    st.info("Tip: Use the **sidebar** to choose a scenario and stack multiple AI techniques. The KPIs and charts will update live.")

# ---------------------- Problem Tab -------------------------
with tab_problem:
    st.markdown("<div class='section-title'>Business Problem</div>", unsafe_allow_html=True)
    st.write("""
- **Observed:** Drop in Search→Cart conversion (e.g., 15% → 10% in the PDFs).
- **Impact:** Lower transactions, revenue leakage, user frustration and churn.
- **Likely Causes:** 
  1) Irrelevant search results / poor tagging  
  2) Out-of-stock items without substitutes or ETA  
  3) Missing **MRP** transparency & unclear images  
  4) No prominent feedback channel  
  5) Seasonal/regional demand mismatch
    """)
    causes = ["Irrelevant Search", "Out-of-Stock", "Missing MRP", "No Feedback", "Seasonality"]
    severity = [40, 25, 15, 10, 10]
    fig_cause = px.pie(names=causes, values=severity, title="Root-Cause Share (Illustrative)", color_discrete_sequence=px.colors.sequential.Agsunset)
    st.plotly_chart(fig_cause, use_container_width=True)

# ---------------------- AI Analysis Tab ---------------------
with tab_ai:
    st.markdown("<div class='section-title'>AI Techniques & Expected Effects</div>", unsafe_allow_html=True)
    df_ai = pd.DataFrame([
        ["NLP (BERT/GPT/Embeddings)", "Match user intent with relevant results", "+conv, -time"],
        ["Recommendation Engine", "Suggest substitutes & bundles", "+add, +retain, +nps"],
        ["ML Prediction", "Forecast demand & churn; rank relevance", "+retain, +conv"],
        ["Sentiment Analysis", "Mine feedback & reviews for pain points", "+nps, small +conv"],
        ["Dynamic Pricing", "Targeted promos for price-sensitive SKUs", "+conv, +add"],
        ["OOS Forecasting", "ETA & back-in-stock alerts", "+conv, +retain"],
    ], columns=["Technique", "Purpose", "Primary Impact"])
    st.dataframe(df_ai, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    ph = st.empty()
    insight_text = f"Applying **{', '.join(chosen_tech) or 'no AI yet'}** on top of the **{preset}** scenario projects Search→Cart ≈ **{final_after['conv']:.1f}%**, NPS ≈ **{final_after['nps']}**, with search time ≈ **{final_after['time_s']:.1f}s**."
    for i in range(0, len(insight_text)+1):
        ph.markdown("🧩 **AI Insight:** " + insight_text[:i])
        time.sleep(0.005)

# ---------------------- Dashboard Tab -----------------------
with tab_dashboard:
    st.markdown("<div class='section-title'>Interactive Dashboard</div>", unsafe_allow_html=True)

    metrics = ["Search→Cart (%)", "Add-to-Cart (%)", "Retention (%)", "NPS", "Avg Search Time (s)"]
    before_vals = [baseline['conv'], baseline['add'], baseline['retain'], baseline['nps'], baseline['time_s']]
    after_vals  = [final_after['conv'], final_after['add'], final_after['retain'], final_after['nps'], final_after['time_s']]
    df_comp = pd.DataFrame({"Metric": metrics, "Before": before_vals, "After": after_vals})

    # Bar comparison
    fig_bar = px.bar(df_comp, x="Metric", y=["Before", "After"], barmode="group",
                     color_discrete_sequence=["#2E86C1", "#58D68D"], text_auto=True, template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Line trend (synthetic 8-week view)
    weeks = np.arange(1, 9)
    base_trend = np.linspace(baseline['conv']-1, baseline['conv']+0.5, len(weeks))
    after_trend = base_trend + (final_after['conv'] - baseline['conv']) * np.linspace(0.2, 1.0, len(weeks))
    df_trend = pd.DataFrame({"Week": weeks, "Baseline Conv%": base_trend, "Post-AI Conv%": after_trend})
    fig_line = px.line(df_trend, x="Week", y=["Baseline Conv%", "Post-AI Conv%"],
                       markers=True, color_discrete_sequence=["#8391a7", "#2E86C1"], template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

    # Radar / Polar chart
    polar_df = pd.DataFrame({
        "Dimension": ["Conv", "Add", "Retention", "NPS", "Speed(Inv)"],
        "Before": [baseline['conv'], baseline['add'], baseline['retain'], baseline['nps'], 100 - baseline['time_s']*5],
        "After":  [final_after['conv'], final_after['add'], final_after['retain'], final_after['nps'], 100 - final_after['time_s']*5],
    })
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatterpolar(
        r=polar_df["Before"], theta=polar_df["Dimension"], fill='toself', name='Before', line_color="#8391a7"
    ))
    fig_polar.add_trace(go.Scatterpolar(
        r=polar_df["After"], theta=polar_df["Dimension"], fill='toself', name='After', line_color="#2E86C1"
    ))
    fig_polar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_polar, use_container_width=True)

# ---------------------- Personas Tab ------------------------
with tab_personas:
    st.markdown("<div class='section-title'>User Personas</div>", unsafe_allow_html=True)
    grid = st.columns(2)
    for i, (name, age, need, pain) in enumerate(PERSONAS):
        with grid[i % 2]:
            st.markdown(f"""
<div class='card'>
<b>{name}</b> &nbsp; <span class='subtle'>({age})</span><br/>
<span class='pill'>Need</span> {need}<br/>
<span class='pill'>Pain</span> {pain}
</div>
""", unsafe_allow_html=True)

# ---------------------- Ethics Tab --------------------------
with tab_ethics:
    st.markdown("<div class='section-title'>Ethical AI Considerations</div>", unsafe_allow_html=True)
    st.write("""
**Privacy & Data Protection** • **Bias & Fairness** • **Transparency** • **Human–AI Collaboration** • **Equity & Access**

- Collect & process data with **informed consent** and strong security.
- Audit models for **bias**; retrain with inclusive datasets.
- Provide **explanations** where decisions affect users (search ranking, pricing).
- Keep **humans in the loop** for oversight.
- Ensure **equal access**; avoid widening digital divides.
    """)
    ethics_dims = ["Privacy", "Fairness", "Transparency", "Human-AI", "Equity"]
    maturity_before = [60, 55, 50, 65, 58]
    maturity_after  = [80, 75, 72, 82, 78]
    df_eth = pd.DataFrame({"Dimension": ethics_dims, "Before": maturity_before, "After": maturity_after})
    fig_eth = px.bar(df_eth, x="Dimension", y=["Before", "After"], barmode="group",
                     color_discrete_sequence=["#8391a7", "#58D68D"], text_auto=True, template="plotly_dark")
    st.plotly_chart(fig_eth, use_container_width=True)

# ---------------------- Downloads Tab -----------------------
with tab_files:
    st.markdown("<div class='section-title'>Downloads</div>", unsafe_allow_html=True)

    if REPORT_PATH and Path(REPORT_PATH).exists():
        st.success(f"Attached report found: `{REPORT_PATH}`")
        with open(REPORT_PATH, "rb") as f:
            st.download_button("📥 Download Full Report (PDF)", f, file_name="AI_Project_Report.pdf", mime="application/pdf")
    else:
        st.warning("No report PDF found in the app folder. Place your file (e.g., `AI_Product_Club_Enhanced_Team_Report.pdf` or `USE OF AI TOOLS.pdf`) next to app.py, or upload it via the sidebar.")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.write("You can also upload a different PDF in the **sidebar** to replace the current one on the fly.")

# ---------------------- Footer ------------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Developed for the Product Club Case Study • Interactive, presentation-ready web app • Streamlit + Plotly (dark theme)")
