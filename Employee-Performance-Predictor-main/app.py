"""
app.py  ─  Employee Performance Predictor  |  Streamlit Dashboard
=================================================================
Multi-page Streamlit app with:
  🏠 Home / Overview
  📊 Analytics & EDA
  🤖 Predict (single + batch)
  🎯 HR Recommendations
  📈 Model Insights
  ⚙️  Settings / Train Model
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# ── path fix ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import generate_employee_dataset
from predictor import (
    load_artifacts, predict_employee, batch_predict,
    get_recommendations, get_feature_importance,
    BAND_COLORS, BAND_ICONS
)

# ── Page config  (MUST be first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Auto-initialize: generate data + train model on first run ─────────────
_DATA_PATH_INIT  = os.path.join(BASE_DIR, "data", "employee_features.csv")
_MODEL_PATH_INIT = os.path.join(BASE_DIR, "models", "employee_perf_model.pkl")

def _auto_setup():
    """Run once on cold start: generate dataset + train model silently."""
    import io, contextlib, warnings as _w
    _w.filterwarnings("ignore")
    os.makedirs(os.path.join(BASE_DIR, "data"),   exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # 1. Generate dataset if missing
    if not os.path.exists(_DATA_PATH_INIT):
        generate_employee_dataset(1000, save_path=_DATA_PATH_INIT)

    # 2. Train model if missing
    if not os.path.exists(_MODEL_PATH_INIT):
        from train_model import load_and_validate, train as _train
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df_init = load_and_validate(_DATA_PATH_INIT)
            _train(df_init)

def _ensure_model_ready():
    """Called on any page that needs the model. Trains inline if missing."""
    if not os.path.exists(_MODEL_PATH_INIT):
        with st.spinner("🤖 Training model for the first time — please wait (~60 sec)..."):
            _auto_setup()
        st.success("✅ Model trained! Loading your results...")
        st.rerun()

if "initialized" not in st.session_state:
    with st.spinner("⚙️ First-time setup — training model (~60 sec, runs once only)..."):
        _auto_setup()
    st.session_state["initialized"] = True

# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 14px !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .value {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .label { font-size: .85rem; color: #94a3b8; margin-top:4px; }

/* Band badges */
.band-high   { background:#166534; color:#4ade80; padding:4px 14px; border-radius:999px; font-weight:700; font-size:.9rem; }
.band-medium { background:#78350f; color:#fbbf24; padding:4px 14px; border-radius:999px; font-weight:700; font-size:.9rem; }
.band-low    { background:#7f1d1d; color:#f87171; padding:4px 14px; border-radius:999px; font-weight:700; font-size:.9rem; }

/* Recommendation cards */
.rec-card {
    background: #1e293b;
    border-left: 4px solid #818cf8;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size:.9rem;
}
.rec-high { border-left-color: #ef4444; }
.rec-medium { border-left-color: #f59e0b; }

/* Section header */
.section-header {
    font-size:1.6rem; font-weight:800; color:#e2e8f0;
    border-bottom: 2px solid rgba(99,102,241,.4);
    padding-bottom:8px; margin-bottom:20px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 8px;
    color: #94a3b8; font-weight: 600; padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white; border: none; border-radius: 10px;
    font-weight: 700; padding: 10px 24px;
    transition: opacity .2s;
}
.stButton button:hover { opacity: .88; }

/* Override streamlit background */
.main .block-container { padding: 2rem 2rem 3rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(BASE_DIR, "data", "employee_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "employee_perf_model.pkl")

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        df = generate_employee_dataset(1000, save_path=DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
    return df

def model_exists():
    return os.path.exists(MODEL_PATH)

def band_badge(band):
    cls = {"High": "band-high", "Medium": "band-medium", "Low": "band-low"}.get(band, "")
    icon = BAND_ICONS.get(band, "")
    return f'<span class="{cls}">{icon} {band}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 PerfPredict AI")
    st.markdown("*HR Analytics Dashboard*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Analytics & EDA", "🤖 Live Prediction",
         "📋 Batch Scoring", "🎯 HR Recommendations", "📈 Model Insights", "⚙️ Train / Setup"],
        label_visibility="collapsed"
    )
    st.divider()
    model, encoder = load_artifacts()
    if model:
        st.success("✅ Model Ready")
        st.caption("RandomForest · 3-class · F1-macro optimized")
    else:
        with st.spinner("Training model..."):
            _auto_setup()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="section-header">🎯 Employee Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown("""
    > **AI-powered HR analytics** that predicts employee performance bands (High / Medium / Low)
    > and surfaces actionable drivers for coaching, L&D, and retention decisions.
    """)

    df = load_data()

    # KPI row
    total = len(df)
    high  = (df["perf_band_next"] == "High").sum()
    med   = (df["perf_band_next"] == "Medium").sum()
    low   = (df["perf_band_next"] == "Low").sum()
    avg_score = df["performance_score"].mean()
    avg_train = df["training_hours"].mean()

    cols = st.columns(6)
    metrics = [
        (total,        "Total Employees",   "👥"),
        (high,         "High Performers",   "🌟"),
        (med,          "Mid Performers",    "📈"),
        (low,          "At-Risk (Low)",     "⚠️"),
        (f"{avg_score:.1f}", "Avg Score",  "📊"),
        (f"{avg_train:.0f}h","Avg Training","🎓"),
    ]
    for col, (val, label, icon) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem">{icon}</div>
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,1])

    with c1:
        # Donut chart
        band_counts = df["perf_band_next"].value_counts()
        fig = go.Figure(go.Pie(
            labels=band_counts.index,
            values=band_counts.values,
            hole=0.55,
            marker_colors=["#22c55e","#f59e0b","#ef4444"]
        ))
        fig.update_layout(
            title="Performance Distribution",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Avg score by dept
        dept_score = df.groupby("department")["performance_score"].mean().sort_values()
        fig2 = go.Figure(go.Bar(
            x=dept_score.values, y=dept_score.index,
            orientation="h",
            marker=dict(
                color=dept_score.values,
                colorscale="Viridis"
            )
        ))
        fig2.update_layout(
            title="Avg Performance by Department",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320, xaxis_title="Avg Score", yaxis_title=""
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        # Training hours vs performance score scatter (sampled)
        sample = df.sample(min(300, len(df)), random_state=1)
        fig3 = px.scatter(
            sample, x="training_hours", y="performance_score",
            color="perf_band_next",
            color_discrete_map=BAND_COLORS,
            title="Training Hours vs Score",
            template="plotly_dark",
            height=320,
            opacity=0.7,
            labels={"training_hours": "Training Hours", "performance_score": "Score"}
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    # Project workflow
    st.markdown("### 🔄 System Workflow")
    workflow_html = """
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:12px 0;">
        <div style="background:#1e293b; border:1px solid #6366f1; border-radius:12px; padding:10px 18px; font-size:.85rem; font-weight:600; color:#818cf8;">📂 HR Data</div>
        <span style="color:#6366f1; font-size:1.2rem">→</span>
        <div style="background:#1e293b; border:1px solid #6366f1; border-radius:12px; padding:10px 18px; font-size:.85rem; font-weight:600; color:#818cf8;">🧹 Preprocessing</div>
        <span style="color:#6366f1; font-size:1.2rem">→</span>
        <div style="background:#1e293b; border:1px solid #6366f1; border-radius:12px; padding:10px 18px; font-size:.85rem; font-weight:600; color:#818cf8;">🔍 EDA</div>
        <span style="color:#6366f1; font-size:1.2rem">→</span>
        <div style="background:#1e293b; border:1px solid #6366f1; border-radius:12px; padding:10px 18px; font-size:.85rem; font-weight:600; color:#818cf8;">🤖 ML Model</div>
        <span style="color:#6366f1; font-size:1.2rem">→</span>
        <div style="background:#1e293b; border:1px solid #6366f1; border-radius:12px; padding:10px 18px; font-size:.85rem; font-weight:600; color:#818cf8;">🎯 Prediction</div>
        <span style="color:#6366f1; font-size:1.2rem">→</span>
        <div style="background:#1e293b; border:1px solid #6366f1; border-radius:12px; padding:10px 18px; font-size:.85rem; font-weight:600; color:#818cf8;">💡 HR Insights</div>
        <span style="color:#6366f1; font-size:1.2rem">→</span>
        <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6); border-radius:12px; padding:10px 18px; font-size:.85rem; font-weight:700; color:white;">✅ Decisions</div>
    </div>
    """
    st.markdown(workflow_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS & EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Analytics & EDA":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    df = load_data()

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📉 Distributions", "🔗 Correlations", "🏢 Segment Analysis"])

    with tab1:
        st.subheader("Dataset Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows",    len(df))
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing", int(df.isnull().sum().sum()))
        st.dataframe(df.head(50), use_container_width=True, height=400)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

    with tab2:
        feat = st.selectbox("Select feature", [c for c in df.select_dtypes("number").columns if c not in ["employee_id"]])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=feat, color="perf_band_next",
                               color_discrete_map=BAND_COLORS,
                               barmode="overlay", template="plotly_dark",
                               title=f"{feat} Distribution by Band",
                               opacity=0.75)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.box(df, x="perf_band_next", y=feat,
                          color="perf_band_next",
                          color_discrete_map=BAND_COLORS,
                          template="plotly_dark",
                          title=f"{feat} Box Plot by Band",
                          category_orders={"perf_band_next": ["Low","Medium","High"]})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        num_df = df.select_dtypes("number").drop(columns=["salary"], errors="ignore")
        corr   = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r",
                        template="plotly_dark",
                        title="Feature Correlation Matrix",
                        height=600)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Top correlations with performance score
        st.subheader("Top Correlations with Performance Score")
        perf_corr = num_df.corr()["performance_score"].drop("performance_score").sort_values()
        fig2 = go.Figure(go.Bar(
            x=perf_corr.values,
            y=perf_corr.index,
            orientation="h",
            marker=dict(
                color=perf_corr.values,
                colorscale="RdYlGn",
                showscale=True
            )
        ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=500,
            xaxis_title="Correlation Coefficient"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            # Performance by dept
            dept_dist = df.groupby(["department","perf_band_next"]).size().reset_index(name="count")
            fig = px.bar(dept_dist, x="department", y="count", color="perf_band_next",
                         color_discrete_map=BAND_COLORS, barmode="stack",
                         template="plotly_dark", title="Performance by Department")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Performance by job level
            lvl_dist = df.groupby(["job_level","perf_band_next"]).size().reset_index(name="count")
            order = ["Junior","Mid","Senior","Lead","Manager"]
            fig2 = px.bar(lvl_dist, x="job_level", y="count", color="perf_band_next",
                          color_discrete_map=BAND_COLORS, barmode="group",
                          template="plotly_dark", title="Performance by Job Level",
                          category_orders={"job_level": order})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            # Avg training by band
            tr = df.groupby("perf_band_next")["training_hours"].mean().reset_index()
            fig3 = px.bar(tr, x="perf_band_next", y="training_hours",
                          color="perf_band_next", color_discrete_map=BAND_COLORS,
                          template="plotly_dark", title="Avg Training Hours by Band")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            # Gender distribution
            gender_perf = df.groupby(["gender","perf_band_next"]).size().reset_index(name="count")
            fig4 = px.bar(gender_perf, x="gender", y="count", color="perf_band_next",
                          color_discrete_map=BAND_COLORS, barmode="stack",
                          template="plotly_dark", title="Gender & Performance Distribution")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Live Prediction":
    st.markdown('<div class="section-header">🤖 Live Employee Performance Prediction</div>', unsafe_allow_html=True)

    _ensure_model_ready()


    df = load_data()

    mode = st.radio("Input Mode", ["🖊️ Manual Entry", "🔍 Lookup from Dataset"], horizontal=True)

    if mode == "🔍 Lookup from Dataset":
        emp_id = st.selectbox("Select Employee ID", df["employee_id"].tolist())
        row_df = df[df["employee_id"] == emp_id].copy()
        st.dataframe(row_df.T, use_container_width=True)
    else:
        st.markdown("#### Enter Employee Details")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age              = st.number_input("Age",              18, 65, 32)
            experience_years = st.number_input("Experience (yrs)", 0,  40,  8)
            manager_tenure   = st.number_input("Manager Tenure",   1,  15,  3)
            salary           = st.number_input("Salary (₹)",       0, 5000000, 1200000, step=50000)
        with c2:
            on_time          = st.slider("On-Time Delivery Rate",  0.0, 1.0, 0.80, 0.01)
            delivery_delay   = st.slider("Avg Task Delay (days)",  0.0, 30.0, 2.0, 0.5)
            billable         = st.slider("Billable Hours Ratio",   0.0, 1.0, 0.75, 0.01)
            projects         = st.number_input("Projects Count",   1, 15, 4)
        with c3:
            training_hrs     = st.number_input("Training Hours",   0, 150, 35)
            certifications   = st.number_input("Certifications",   0, 10, 2)
            hackathons       = st.number_input("Hackathons",       0, 10, 1)
            kudos            = st.number_input("Kudos Count",      0, 50, 6)
        with c4:
            bug_count        = st.number_input("Bug Count",        0, 40, 4)
            code_review      = st.slider("Code Review Score",      0.0, 5.0, 3.5, 0.1)
            peer_score       = st.slider("Peer Feedback Score",    0.0, 5.0, 3.8, 0.1)
            manager_score    = st.slider("Manager Score",          0.0, 5.0, 3.9, 0.1)

        c5, c6, c7 = st.columns(3)
        with c5:
            sick_days        = st.number_input("Sick Days",        0, 40, 3)
            unplanned_abs    = st.number_input("Unplanned Absences",0, 20, 1)
            avg_login        = st.slider("Avg Login Hours",        2.0, 14.0, 7.5, 0.5)
        with c6:
            qa_defect        = st.slider("QA Defect Density",      0.0, 15.0, 1.5, 0.1)
            promotions       = st.selectbox("Promotions in 2y",    [0, 1, 2])
        with c7:
            gender           = st.selectbox("Gender",              ["Male","Female","Non-binary"])
            education        = st.selectbox("Education",           ["Bachelor's","Master's","PhD","Diploma"])
            department       = st.selectbox("Department",          ["Engineering","Sales","HR","Finance","Marketing","Operations","Product","Data Science"])
            job_level        = st.selectbox("Job Level",           ["Junior","Mid","Senior","Lead","Manager"])
            salary_band      = st.selectbox("Salary Band",         ["Low","Mid","High"])

        row_df = pd.DataFrame([{
            "age": age, "gender": gender, "education": education,
            "experience_years": experience_years, "department": department,
            "job_level": job_level, "manager_tenure": manager_tenure,
            "salary": salary, "on_time_delivery_rate": on_time,
            "avg_task_delay_days": delivery_delay, "projects_count": projects,
            "billable_hours_ratio": billable, "bug_count": bug_count,
            "code_review_score": code_review, "qa_defect_density": qa_defect,
            "training_hours": training_hrs, "certifications_count": certifications,
            "hackathons_participated": hackathons, "sick_days": sick_days,
            "unplanned_absences": unplanned_abs, "avg_login_hours": avg_login,
            "peer_feedback_score": peer_score, "manager_score": manager_score,
            "kudos_count": kudos, "promotions_in_2y": promotions,
            "salary_percentile_band": salary_band,
        }])

    if st.button("🔮 Predict Performance", use_container_width=True):
        with st.spinner("Running model..."):
            label, probas, proba_dict = predict_employee(row_df)

        st.markdown("---")
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            color = BAND_COLORS.get(label, "#6366f1")
            icon  = BAND_ICONS.get(label, "")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e293b,#0f172a);
                        border:2px solid {color}; border-radius:20px;
                        padding:32px; text-align:center;">
                <div style="font-size:3.5rem">{icon}</div>
                <div style="font-size:1.1rem; color:#94a3b8; margin:8px 0">Predicted Band</div>
                <div style="font-size:3rem; font-weight:900; color:{color}">{label}</div>
                <div style="font-size:.85rem; color:#64748b; margin-top:8px">
                    Confidence: {max(proba_dict.values())*100:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)

        with cc2:
            fig = go.Figure(go.Bar(
                x=list(proba_dict.keys()),
                y=[v*100 for v in proba_dict.values()],
                marker_color=[BAND_COLORS.get(k,"#6366f1") for k in proba_dict.keys()],
                text=[f"{v*100:.1f}%" for v in proba_dict.values()],
                textposition="outside"
            ))
            fig.update_layout(
                title="Prediction Probabilities (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Probability (%)",
                height=280,
                yaxis=dict(range=[0, 110])
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("#### 💡 Recommended HR Actions")
        recs = get_recommendations(row_df.iloc[0])
        if recs:
            for r in recs:
                sev_cls = "rec-high" if r["severity"] == "High" else "rec-medium"
                st.markdown(f"""
                <div class="rec-card {sev_cls}">
                    {r['icon']} <strong>{r['feature'].replace('_',' ').title()}</strong>
                    — Current: <code>{r['value']}</code> | Threshold: <code>{r['threshold']}</code>
                    <br><span style="color:#94a3b8">{r['action']}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ No critical interventions needed. Employee is performing well!")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BATCH SCORING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Batch Scoring":
    st.markdown('<div class="section-header">📋 Batch Employee Scoring</div>', unsafe_allow_html=True)

    _ensure_model_ready()


    df = load_data()
    upload = st.file_uploader("Upload CSV (optional — uses default dataset if not uploaded)",
                               type=["csv"])
    if upload:
        df = pd.read_csv(upload)
        st.success(f"✅ Loaded {len(df)} employees from upload")

    if st.button("🚀 Score All Employees", use_container_width=True):
        with st.spinner("Scoring employees..."):
            result = batch_predict(df)

        st.success(f"✅ Scored {len(result)} employees!")

        # Summary
        c1, c2, c3 = st.columns(3)
        for band, col in zip(["High","Medium","Low"], [c1,c2,c3]):
            n = (result["pred_band"] == band).sum()
            col.metric(f"{BAND_ICONS[band]} {band}", n, f"{n/len(result)*100:.1f}%")

        st.dataframe(
            result[["employee_id","department","job_level","pred_band",
                     "prob_High","prob_Medium","prob_Low"]]\
                  .sort_values("prob_Low", ascending=False),
            use_container_width=True, height=450
        )

        # Download
        csv = result.to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Results CSV",
            csv, "scored_employees.csv", "text/csv",
            use_container_width=True
        )

        # At-risk employees
        st.markdown("### ⚠️ At-Risk Employees (Low Predicted Band)")
        at_risk = result[result["pred_band"]=="Low"].sort_values("prob_Low", ascending=False)
        if len(at_risk):
            st.dataframe(at_risk[["employee_id","department","job_level",
                                   "prob_Low","training_hours",
                                   "on_time_delivery_rate","manager_score"]].head(20),
                         use_container_width=True)
        else:
            st.success("No at-risk employees found!")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HR RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 HR Recommendations":
    st.markdown('<div class="section-header">🎯 HR Recommendations Engine</div>', unsafe_allow_html=True)

    _ensure_model_ready()


    df = load_data()
    result = batch_predict(df)

    dept_filter  = st.multiselect("Filter by Department", df["department"].unique(), default=list(df["department"].unique()))
    level_filter = st.multiselect("Filter by Job Level",  df["job_level"].unique(),  default=list(df["job_level"].unique()))
    band_filter  = st.multiselect("Show Bands",           ["High","Medium","Low"],   default=["Low","Medium"])

    filtered = result[
        (result["department"].isin(dept_filter)) &
        (result["job_level"].isin(level_filter)) &
        (result["pred_band"].isin(band_filter))
    ]

    st.markdown(f"**{len(filtered)} employees matched filters**")

    # Aggregate intervention frequency
    all_recs = []
    for _, row in filtered.iterrows():
        recs = get_recommendations(row)
        for r in recs:
            all_recs.append({"employee_id": row["employee_id"],
                              "department": row["department"],
                              "pred_band":  row["pred_band"],
                              **r})

    if all_recs:
        rec_df = pd.DataFrame(all_recs)

        # Top intervention areas
        top = rec_df["feature"].value_counts().head(10).reset_index()
        top.columns = ["feature","count"]
        fig = go.Figure(go.Bar(
            x=top["count"], y=top["feature"].str.replace("_"," ").str.title(),
            orientation="h",
            marker=dict(color=top["count"], colorscale="Plasma")
        ))
        fig.update_layout(
            title="Top Intervention Areas (Frequency)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # Individual recommendations table
        st.markdown("### 📋 Detailed Action Items")
        display = rec_df[["employee_id","department","pred_band","feature","value","action","severity"]]\
                  .sort_values(["severity","pred_band"])
        st.dataframe(display, use_container_width=True, height=400)

        csv2 = display.to_csv(index=False)
        st.download_button("⬇️ Download Recommendations CSV", csv2,
                           "hr_recommendations.csv", "text/csv")
    else:
        st.info("No recommendations for selected filters.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Insights":
    st.markdown('<div class="section-header">📈 Model Insights & Fairness</div>', unsafe_allow_html=True)

    _ensure_model_ready()


    model, encoder = load_artifacts()
    df = load_data()

    # Feature importance
    st.subheader("🔑 Feature Importance (MDI)")
    imp = get_feature_importance(20)
    if not imp.empty:
        # Clean names from ColumnTransformer prefix
        clean_names = []
        for name in imp.index:
            if "num__" in name:
                clean_names.append(name.replace("num__", ""))
            elif "cat__" in name:
                parts = name.replace("cat__", "").split("_")
                clean_names.append(" ".join(parts[-2:]).title())
            else:
                clean_names.append(name)

        fig = go.Figure(go.Bar(
            x=imp.values,
            y=clean_names,
            orientation="h",
            marker=dict(color=imp.values, colorscale="Viridis", showscale=True)
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=500,
            xaxis_title="Importance Score", yaxis_autorange="reversed"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Fairness / group analysis
    st.subheader("⚖️ Fairness Audit")
    result = batch_predict(df)
    for group_col in ["gender", "department", "job_level"]:
        if group_col not in result.columns:
            continue
        grp = result.groupby(group_col)["pred_band"].value_counts(normalize=True).unstack().fillna(0)
        fig2 = px.bar(grp, barmode="group",
                      color_discrete_map=BAND_COLORS,
                      template="plotly_dark",
                      title=f"Prediction Distribution by {group_col.replace('_',' ').title()}",
                      labels={"value": "Proportion", "variable": "Band"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # Selection rate table
    st.subheader("📊 High-Band Selection Rates by Gender")
    rates = result.groupby("gender").apply(
        lambda x: (x["pred_band"] == "High").mean()
    ).reset_index(name="High Band Rate")
    st.dataframe(rates.style.format({"High Band Rate": "{:.2%}"}), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETUP / TRAIN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Train / Setup":
    st.markdown('<div class="section-header">⚙️ Setup & Model Training</div>', unsafe_allow_html=True)

    st.markdown("""
    This page handles:
    - **Generating** a fresh synthetic HR dataset
    - **Training** the RandomForest model with GridSearch
    - **Viewing** training logs and results
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Generate Dataset")
        n_emp = st.slider("Number of employees", 200, 2000, 1000, 100)
        if st.button("🔄 Generate / Refresh Dataset"):
            with st.spinner("Generating synthetic HR dataset..."):
                load_data.clear()
                os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
                df = generate_employee_dataset(n_emp, save_path=DATA_PATH)
                st.success(f"✅ Generated {len(df)} employee records!")
                st.dataframe(df.head(5), use_container_width=True)

    with col2:
        st.subheader("2️⃣ Train Model")
        st.markdown("""
        **Model**: RandomForest Classifier  
        **Objective**: Macro-F1 optimized  
        **CV**: Stratified 5-fold  
        **Balancing**: class_weight='balanced'
        """)
        if st.button("🚀 Train Model (may take ~1-2 minutes)"):
            if not os.path.exists(DATA_PATH):
                st.error("Please generate dataset first!")
            else:
                import subprocess
                train_script = os.path.join(BASE_DIR, "src", "train_model.py")
                with st.spinner("Training model with GridSearchCV..."):
                    log_area = st.empty()
                    import io, contextlib
                    from train_model import load_and_validate, train as run_train
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        df_train = load_and_validate(DATA_PATH)
                        run_train(df_train)
                    log_area.code(buf.getvalue())
                st.success("✅ Model trained and saved!")
                st.balloons()

    st.divider()
    st.subheader("📂 Folder Structure")
    st.code("""
Employee-Performance-Predictor/
│
├── app.py                   ← Streamlit dashboard (main entry)
├── requirements.txt         ← Python dependencies
├── README.md                ← Project documentation
│
├── data/
│   └── employee_features.csv    ← Synthetic HR dataset (1000 employees)
│
├── src/
│   ├── data_generator.py    ← Synthetic data generation
│   ├── train_model.py       ← ML pipeline: preprocess → train → save
│   └── predictor.py         ← Prediction + recommendations utilities
│
├── models/
│   ├── employee_perf_model.pkl  ← Trained RandomForest pipeline
│   └── label_encoder.pkl        ← Label encoder for target classes
│
└── outputs/
    └── scored_employees.csv     ← Batch prediction outputs
    """, language="")
