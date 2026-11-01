import os

import numpy as np
import pandas as pd
import streamlit as st

import pipeline

st.set_page_config(page_title="EDAgent — AI EDA", page_icon="📊", layout="wide")

# ── CSS (st.html injects into the page itself, not an iframe) ─────────────────
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"), encoding="utf-8") as _f:
    st.html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">'
            f"<style>{_f.read()}</style>")

LABELS = {
    "schema": "🔍 Schema", "quality": "🧪 Quality", "stats": "📊 Statistics", "outliers": "🎯 Outliers",
    "cleaning": "🧹 Data Cleaning", "correlation": "🔗 Correlation", "importance": "⚡ Feature Importance",
    "baseline": "🏁 Baseline Models", "synthesis": "📝 Narrative", "model_rec": "🤖 Model Rec.",
    "feature_eng": "⚙️ Feature Eng.",
}


# ── Sidebar: model ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model")
    local = pipeline.ollama_models()  # not cached: an empty list must not stick after Ollama starts
    if not local:
        st.warning("Ollama isn't running or has no models. Start it (or run `ollama pull qwen3.5:4b`), "
                   "then refresh the page.")
    options = (local
               + (pipeline.CLAUDE_CODE_MODELS if pipeline.has_claude_code() else [])
               + (pipeline.CLAUDE_MODELS if pipeline.has_claude_key() else []))
    if not options:
        st.error("No model available. Start Ollama, or install Claude Code.")
        st.stop()
    default = pipeline.default_model()
    model = st.selectbox("LLM", options, index=options.index(default) if default in options else 0)
    if model.startswith("claude-code:"):
        st.caption("Runs through Claude Code on your Claude plan: about 10 calls per run, which count "
                   "toward your usage limits. Data summaries leave your machine.")
    elif model.startswith("claude-"):
        st.caption("Runs on the Claude API, so data summaries leave your machine. Billed per run.")
    else:
        st.caption("Runs locally with Ollama. Nothing leaves your machine.")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge"><span class="dot"></span>Local with Ollama, or fast with Claude</div>
  <h1>EDAgent</h1>
  <p class="tagline">Upload any dataset. <strong>Eleven specialized agents</strong> analyze
  schema, quality, statistics, outliers, correlations, baselines and more, in one run.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

uploaded = st.file_uploader("Drop your CSV or Excel file here", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    df = pipeline.load_table(uploaded, uploaded.name)

    st.markdown('<p class="section-title">Dataset preview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(10), use_container_width=True)

    cols = list(df.columns)
    guess = pipeline.guess_target(df)
    target = st.selectbox("Target column (what you want to predict)", ["(none)"] + cols,
                          index=cols.index(guess) + 1 if guess in cols else 0,
                          help="Drives feature importance, the baseline models and the model recommendation.")
    target = None if target == "(none)" else target
    st.markdown("---")

    if st.button("🚀  Run EDA Pipeline", type="primary"):
        for k in ("result", "chat"):
            st.session_state.pop(k, None)
        done = []
        pb = st.progress(0, text="Starting agents…")
        grid = st.empty()

        def on_step(node: str):
            done.append(LABELS.get(node, node))
            pb.progress(len(done) / len(LABELS), text=f"{len(done)}/{len(LABELS)} agents done — {done[-1]}")
            grid.markdown('<div class="step-list">' + "".join(
                f'<div class="step-item"><span class="step-check">✓</span>{l}</div>' for l in done) + "</div>",
                unsafe_allow_html=True)

        try:
            result = pipeline.run_pipeline(df, model, target, on_step=on_step)
        except Exception as e:
            st.error(f"Pipeline failed: {type(e).__name__}: {e}")
            st.stop()
        st.session_state.update(result=result, df=df, dataset_name=uploaded.name)
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    df_orig = st.session_state["df"]
    name = st.session_state["dataset_name"]
    base = os.path.splitext(name)[0]
    df_clean = result.get("df", df_orig)

    st.success("✅  Analysis complete — explore every insight below.")
    st.markdown("---")

    tabs = st.tabs(["🔍 Schema", "🧪 Quality", "📊 Statistics", "🎯 Outliers", "🧹 Cleaning", "📈 Distributions",
                    "🔗 Correlation", "⚡ Importance", "🏁 Baseline", "📝 Narrative", "🤖 Model", "⚙️ Features",
                    "💬 Ask"])

    for tab, title, key in [(tabs[0], "Schema Analysis", "schema"), (tabs[1], "Data Quality Report", "quality"),
                            (tabs[2], "Descriptive Statistics", "stats"), (tabs[3], "Outlier Detection", "outliers"),
                            (tabs[7], "Feature Importance", "importance"), (tabs[8], "Baseline Models", "baseline"),
                            (tabs[9], "EDA Narrative", "narrative"), (tabs[10], "Model Recommendation", "model_recommendation")]:
        with tab:
            st.markdown(f"#### {title}")
            st.markdown(result.get(key, ""))

    # ── Cleaning ──────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### Data Cleaning")

        def dc_card(label, old, new):
            inner = (f'<span class="dc-same">{new}</span>' if old == new else
                     f'<span class="dc-old">{old}</span><span class="dc-arrow">→</span><span class="dc-new">{new}</span>')
            return (f'<div class="diff-card {"" if old == new else "improved"}"><div class="dc-label">{label}</div>'
                    f'<div class="dc-values">{inner}</div></div>')

        st.markdown('<div class="diff-grid">'
                    + dc_card("Rows", df_orig.shape[0], df_clean.shape[0])
                    + dc_card("Missing values", int(df_orig.isnull().sum().sum()), int(df_clean.isnull().sum().sum()))
                    + dc_card("Duplicate rows", int(df_orig.duplicated().sum()), int(df_clean.duplicated().sum()))
                    + "</div>", unsafe_allow_html=True)

        st.markdown("##### Before & After (first 200 rows)")
        shared = [c for c in df_orig.columns if c in df_clean.columns]
        cb, ca = st.columns(2)
        with cb:
            st.markdown('<span class="ba-label before">⬛ Before</span>', unsafe_allow_html=True)
            st.dataframe(df_orig[shared].head(200).astype(str), use_container_width=True)
        with ca:
            st.markdown('<span class="ba-label after">✅ After</span>', unsafe_allow_html=True)
            st.dataframe(df_clean[shared].head(200).astype(str), use_container_width=True)

        st.markdown("---")
        st.markdown("##### Cleaning Log")
        st.markdown(result.get("cleaning", ""))
        script = pipeline.cleaning_script(result.get("cleaning_code", []), name)
        with st.expander("Cleaning script (reproduces every step above)"):
            st.code(script, language="python")
        d1, d2 = st.columns(2)
        d1.download_button("📥 Download Cleaned CSV", df_clean.to_csv(index=False).encode("utf-8"),
                           file_name=f"cleaned_{base}.csv", mime="text/csv")
        d2.download_button("🐍 Download Cleaning Script", script.encode("utf-8"),
                           file_name=f"clean_{base}.py", mime="text/x-python")

    # ── Distributions ─────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("#### Missing values per column (before cleaning, %)")
        miss = df_orig.isnull().mean().mul(100).round(1)
        miss = miss[miss > 0]
        if miss.empty:
            st.caption("No missing values.")
        else:
            st.bar_chart(miss, height=220)

        st.markdown("#### Column distributions (after cleaning)")
        shown = list(df_clean.columns[:30])
        if len(df_clean.columns) > 30:
            st.caption(f"Showing the first 30 of {len(df_clean.columns)} columns.")
        grid = st.columns(3)
        for i, c in enumerate(shown):
            s = df_clean[c].dropna()
            with grid[i % 3]:
                st.caption(f"**{c}**")
                if s.empty:
                    st.caption("All values missing.")
                elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > 20:
                    counts, edges = np.histogram(s, bins=20)
                    st.bar_chart(pd.Series(counts, index=edges[:-1].round(3)), height=180)
                else:
                    st.bar_chart(s.astype(str).value_counts().head(15), height=180)

    # ── Correlation ───────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("#### Correlation Analysis")
        st.markdown(result.get("correlation", ""))
        if result.get("heatmap"):
            st.image(result["heatmap"], caption="Pearson Correlation Heatmap", use_container_width=True)

    # ── Features ──────────────────────────────────────────────────────────────
    with tabs[11]:
        st.markdown("#### Feature Engineering Suggestions")
        st.markdown(result.get("feature_engineering", ""))
        featured = result.get("df_features")
        new_cols = [] if featured is None else [c for c in featured.columns if c not in df_clean.columns]
        if new_cols:
            st.markdown("---")
            st.markdown(f"##### {len(new_cols)} feature(s) applied: " + ", ".join(f"`{c}`" for c in new_cols))
            st.dataframe(featured[new_cols].head(20), use_container_width=True)
            st.download_button("📥 Download Cleaned CSV + New Features", featured.to_csv(index=False).encode("utf-8"),
                               file_name=f"featured_{base}.csv", mime="text/csv")

    # ── Ask ───────────────────────────────────────────────────────────────────
    with tabs[12]:
        st.markdown("#### Ask a question about the cleaned data")
        st.caption("The model writes pandas code, which runs in a separate restricted process.")
        chat = st.session_state.setdefault("chat", [])
        for q, note, code, answer in chat:
            st.markdown(f"**You:** {q}")
            if note:
                st.markdown(note)
            if answer is not None:
                st.write(answer)
            if code:
                with st.expander("Code"):
                    st.code(code, language="python")
            st.markdown("---")
        with st.form("ask", clear_on_submit=True):
            q = st.text_input("Question", placeholder="e.g. Average salary by city?")
            if st.form_submit_button("Ask") and q.strip():
                with st.spinner("Thinking…"):
                    chat.append((q, *pipeline.ask_data(pipeline.make_llm(model), df_clean, q)))
                st.rerun()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_txt, col_btn = st.columns([3, 1])
    with col_txt:
        st.markdown("### Export Full Report")
        st.markdown('<p style="color:#475569;font-size:.88rem;margin:0">'
                    "Self-contained HTML with all analysis sections.</p>", unsafe_allow_html=True)
    with col_btn:
        st.write("")
        st.download_button("📄 Download HTML Report",
                           pipeline.build_html_report(result, name, df_orig).encode("utf-8"),
                           file_name=f"eda_report_{base}.html", mime="text/html")
