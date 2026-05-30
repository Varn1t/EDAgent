import streamlit as st
import pandas as pd
import os
import sys

# Make sure pipeline.py is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EDAgent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark background */
  .stApp { background-color: #0f172a; color: #e2e8f0; }

  /* Main header */
  .eda-header {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
  }
  .eda-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: .4rem;
  }
  .eda-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 0;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    background: #1e293b;
    border: 2px dashed #334155;
    border-radius: 12px;
    padding: 1rem;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 8px;
    padding: 6px;
    display: flex;
    gap: 12px;
    overflow-x: auto;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 6px;
    font-size: .9rem;
    padding: 8px 16px;
    margin-right: 4px;
    white-space: nowrap;
  }
  .stTabs [aria-selected="true"] {
    background: #334155 !important;
    color: #f1f5f9 !important;
  }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

  /* Status box */
  [data-testid="stStatusWidget"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
  }

  /* Download button */
  .stDownloadButton button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: .6rem 1.4rem;
    width: 100%;
  }

  /* Primary button */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 700;
    padding: .8rem 2rem;
    width: 100%;
    transition: opacity .2s;
  }
  .stButton > button[kind="primary"]:hover { opacity: .85; }

  /* Divider */
  hr { border-color: #1e293b; }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Better spacing for markdown content */
  .stMarkdown p {
    margin-bottom: 1.5rem;
    line-height: 1.8;
  }
  .stMarkdown li {
    margin-bottom: 0.8rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="eda-header">
    <h1>EDAgent</h1>
    <h3 style="color: #cbd5e1; font-weight: 500; font-size: 1.4rem; margin-top: -0.5rem; margin-bottom: 1rem;">Exploratory Data Analysis <span style="color: #60a5fa;">+</span> AI Agent</h3>
    <p>Upload a CSV or Excel dataset. Nine AI agents analyze it. Get a complete EDA report — running fully local.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── File Upload ────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload your CSV or Excel dataset",
    type=["csv", "xlsx", "xls"],
    help="The pipeline runs locally via Ollama — your data never leaves your machine."
)

# ── Dataset preview ────────────────────────────────────────────────────────

if uploaded is not None:
    try:
        if uploaded.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except UnicodeDecodeError:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding='latin1')
    dataset_name = uploaded.name

    st.markdown("### Dataset Preview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.dataframe(df.head(10), width="stretch")

    st.markdown("---")

    # ── Run button ─────────────────────────────────────────────────────────
    if st.button("Run EDA Pipeline", type="primary"):
        st.session_state.pop("result", None)  # clear previous run

        with st.status("Running EDA pipeline — this takes a few minutes...", expanded=True) as status:
            agent_labels = {
                "Running schema agent...":              "Schema agent",
                "Running quality agent...":             "Quality agent",
                "Running stats agent...":               "Statistics agent",
                "Running outlier agent...":             "Outlier agent",
                "Running correlation agent...":         "Correlation agent",
                "Running feature importance agent...":  "Feature importance agent",
                "Running synthesis agent...":           "Synthesis agent",
                "Running model recommendation agent...":"Model recommendation agent",
                "Running feature engineering agent...": "Feature engineering agent",
            }

            def on_step(msg: str):
                label = agent_labels.get(msg, msg)
                st.write(f"✓ {label}")

            result = pipeline.run_pipeline(df, on_step=on_step)
            status.update(label="EDA complete!", state="complete", expanded=False)

        st.session_state["result"] = result
        st.session_state["df"] = df
        st.session_state["dataset_name"] = dataset_name
        st.rerun()

# ── Results ────────────────────────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    df_stored = st.session_state["df"]
    name = st.session_state["dataset_name"]

    st.success("EDA complete! Explore your results below.")
    st.markdown("---")

    tabs = st.tabs([
        "Schema",
        "Quality",
        "Statistics",
        "Outliers",
        "Correlation",
        "Feature Importance",
        "EDA Narrative",
        "Model Recommendation",
        "Feature Engineering",
    ])

    with tabs[0]:
        st.markdown("#### Schema Analysis")
        st.markdown(result["schema"])

    with tabs[1]:
        st.markdown("#### Data Quality")
        st.markdown(result["quality"])

    with tabs[2]:
        st.markdown("#### Statistics")
        st.markdown(result["stats"])

    with tabs[3]:
        st.markdown("#### Outlier Detection")
        st.markdown(result["outliers"])

    with tabs[4]:
        st.markdown("#### Correlation Analysis")
        st.markdown(result["correlation"])
        heatmap_path = "output/correlation_heatmap.png"
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Pearson Correlation Heatmap", width="stretch")

    with tabs[5]:
        st.markdown("#### Feature Importance")
        st.markdown(result["importance"])

    with tabs[6]:
        st.markdown("#### EDA Narrative")
        st.markdown(result["narrative"])

    with tabs[7]:
        st.markdown("#### Model Recommendation")
        st.markdown(result["model_recommendation"])

    with tabs[8]:
        st.markdown("#### Feature Engineering Suggestions")
        st.markdown(result["feature_engineering"])

    # ── Download report ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Export Report")

    html_report = pipeline.build_html_report(result, name, df_stored)
    base_name = os.path.splitext(name)[0]
    st.download_button(
        label="📄 Download HTML Report",
        data=html_report.encode("utf-8"),
        file_name=f"eda_report_{base_name}.html",
        mime="text/html",
    )
