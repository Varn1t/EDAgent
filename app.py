import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDAgent — AI EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject CSS (via components.html so the parser never touches it) ───────────
_css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(_css_path, "r", encoding="utf-8") as _f:
    _css = _f.read()

components.html(
    f'<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">'
    f'<style>{_css}</style>',
    height=0,
    scrolling=False,
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge"><span class="dot"></span>Powered by Ollama — 100% Local</div>
  <h1>EDAgent</h1>
  <p class="tagline">Upload any dataset. <strong>Ten specialized AI agents</strong> analyze
  schema, quality, statistics, outliers, correlations, and more — in one run.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your CSV or Excel file here",
    type=["csv", "xlsx", "xls"],
    help="Your data stays on your machine — Ollama runs fully locally."
)

# ── Preview ───────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        df = (pd.read_excel(uploaded)
              if uploaded.name.endswith(('.xlsx', '.xls'))
              else pd.read_csv(uploaded))
    except UnicodeDecodeError:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding='latin1')
    dataset_name = uploaded.name

    st.markdown('<p class="section-title">Dataset preview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           f"{df.shape[0]:,}")
    c2.metric("Columns",        f"{df.shape[1]}")
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")

    # ── Run ───────────────────────────────────────────────────────────────────
    if st.button("🚀  Run EDA Pipeline", type="primary"):
        st.session_state.pop("result", None)

        AGENTS = [
            ("Running schema agent...",               "🔍 Schema"),
            ("Running quality agent...",              "🧪 Quality"),
            ("Running stats agent...",                "📊 Statistics"),
            ("Running outlier agent...",              "🎯 Outliers"),
            ("Running cleaning agent...",             "🧹 Data Cleaning"),
            ("Running correlation agent...",          "🔗 Correlation"),
            ("Running feature importance agent...",   "⚡ Feature Importance"),
            ("Running synthesis agent...",            "📝 Narrative"),
            ("Running model recommendation agent...", "🤖 Model Rec."),
            ("Running feature engineering agent...",  "⚙️ Feature Eng."),
        ]
        total     = len(AGENTS)
        label_map = {k: v for k, v in AGENTS}
        completed = []

        pb   = st.progress(0, text="Initialising pipeline…")
        grid = st.empty()

        def on_step(msg: str):
            label = label_map.get(msg, msg)
            completed.append(label)
            pb.progress(len(completed) / total,
                        text=f"Agent {len(completed)}/{total} — {label}")
            items = "".join(
                f'<div class="step-item"><span class="step-check">✓</span>{l}</div>'
                for l in completed
            )
            grid.markdown(
                f'<div class="step-list">{items}</div>',
                unsafe_allow_html=True
            )

        result = pipeline.run_pipeline(df, on_step=on_step)
        pb.progress(1.0, text="✅  Pipeline complete!")

        st.session_state["result"]       = result
        st.session_state["df"]           = df
        st.session_state["dataset_name"] = dataset_name
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result   = st.session_state["result"]
    df_orig  = st.session_state["df"]
    name     = st.session_state["dataset_name"]
    df_clean = result.get("df", df_orig)

    st.success("✅  Analysis complete — explore every insight below.")
    st.markdown("---")

    tabs = st.tabs([
        "🔍 Schema",
        "🧪 Quality",
        "📊 Statistics",
        "🎯 Outliers",
        "🧹 Cleaning",
        "🔗 Correlation",
        "⚡ Importance",
        "📝 Narrative",
        "🤖 Model",
        "⚙️ Features",
    ])

    with tabs[0]:
        st.markdown("#### Schema Analysis")
        st.markdown(result["schema"])

    with tabs[1]:
        st.markdown("#### Data Quality Report")
        st.markdown(result["quality"])

    with tabs[2]:
        st.markdown("#### Descriptive Statistics")
        st.markdown(result["stats"])

    with tabs[3]:
        st.markdown("#### Outlier Detection")
        st.markdown(result["outliers"])

    # ── Cleaning tab ──────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### Data Cleaning")

        r  = df_orig.shape[0];   rc = df_clean.shape[0]
        m  = int(df_orig.isnull().sum().sum())
        mc = int(df_clean.isnull().sum().sum())
        d  = int(df_orig.duplicated().sum())
        dc_val = int(df_clean.duplicated().sum())

        def dc_card(label, old, new):
            if old == new:
                inner = f'<span class="dc-same">{new}</span>'
                cls   = ""
            else:
                inner = (f'<span class="dc-old">{old}</span>'
                         f'<span class="dc-arrow">→</span>'
                         f'<span class="dc-new">{new}</span>')
                cls   = "improved"
            return (f'<div class="diff-card {cls}">'
                    f'<div class="dc-label">{label}</div>'
                    f'<div class="dc-values">{inner}</div>'
                    f'</div>')

        st.markdown(
            '<div class="diff-grid">'
            + dc_card("Rows",          r,  rc)
            + dc_card("Missing values", m,  mc)
            + dc_card("Duplicate rows", d,  dc_val)
            + '</div>',
            unsafe_allow_html=True
        )

        st.markdown("##### Before & After")
        shared = [c for c in df_orig.columns if c in df_clean.columns]
        cb, ca = st.columns(2)
        with cb:
            st.markdown('<span class="ba-label before">⬛ Before</span>',
                        unsafe_allow_html=True)
            st.dataframe(df_orig[shared].astype(str), use_container_width=True)
        with ca:
            st.markdown('<span class="ba-label after">✅ After</span>',
                        unsafe_allow_html=True)
            st.dataframe(df_clean[shared].astype(str), use_container_width=True)

        st.markdown("---")
        st.markdown("##### Cleaning Log")
        st.markdown(result["cleaning"])

        st.markdown("---")
        base = os.path.splitext(name)[0]
        st.download_button(
            "📥 Download Cleaned CSV",
            df_clean.to_csv(index=False).encode("utf-8"),
            file_name=f"cleaned_{base}.csv",
            mime="text/csv",
            key="dl-clean"
        )

    with tabs[5]:
        st.markdown("#### Correlation Analysis")
        st.markdown(result["correlation"])
        if os.path.exists("output/correlation_heatmap.png"):
            st.image("output/correlation_heatmap.png",
                     caption="Pearson Correlation Heatmap",
                     use_container_width=True)

    with tabs[6]:
        st.markdown("#### Feature Importance")
        st.markdown(result["importance"])

    with tabs[7]:
        st.markdown("#### EDA Narrative")
        st.markdown(result["narrative"])

    with tabs[8]:
        st.markdown("#### Model Recommendation")
        st.markdown(result["model_recommendation"])

    with tabs[9]:
        st.markdown("#### Feature Engineering Suggestions")
        st.markdown(result["feature_engineering"])

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    base = os.path.splitext(name)[0]
    col_txt, col_btn = st.columns([3, 1])
    with col_txt:
        st.markdown("### Export Full Report")
        st.markdown(
            '<p style="color:#475569;font-size:.88rem;margin:0">'
            'Self-contained HTML with all analysis sections.</p>',
            unsafe_allow_html=True
        )
    with col_btn:
        st.write("")
        st.download_button(
            "📄 Download HTML Report",
            pipeline.build_html_report(result, name, df_orig).encode("utf-8"),
            file_name=f"eda_report_{base}.html",
            mime="text/html",
        )
