<div align="center">

# EDAgent

Your personal Exploratory Data Analyst + AI Agent

**Drop in a CSV or Excel dataset. Get a complete EDA — automatically.**

<img width="1878" height="867" alt="image" src="https://github.com/user-attachments/assets/5919cf06-af14-452a-90ad-ba3caaf27906" />

An agentic, LLM-powered Exploratory Data Analysis pipeline built with LangGraph.  
**Eleven specialized agents** analyze your dataset, clean it, train baseline models and generate a polished HTML report. Run it **100% locally with Ollama**, or **much faster on Claude**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-blueviolet?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red?style=flat-square&logo=streamlit)

</div>

---

## What it does

You give it a CSV or Excel dataset and pick the column you want to predict. It spins up an **11-agent LangGraph pipeline**; agents that don't depend on each other run in parallel. At the end, you get:

- An **interactive Streamlit dashboard** with tabbed results and a live per-agent progress bar
- A **rich, color-coded terminal output** (if run via CLI)
- A **self-contained `report.html`** — dark-themed, browser-ready, heatmap embedded inline
- A **`correlation_heatmap.png`** saved to `output/`
- A **cleaned dataset** ready to download as CSV, plus a **Python script that reproduces every cleaning step**
- **Cross-validated baseline models** (scikit-learn) so the model recommendation rests on real scores
- **Distribution charts** for every column and a missing-values chart
- The suggested **engineered features applied**, downloadable as CSV
- An **Ask tab**: question your data in plain English; the model writes pandas code that runs in a restricted separate process

<img width="1867" height="242" alt="image" src="https://github.com/user-attachments/assets/5f972bea-8bee-4579-a4fb-ccb036983c65" />


---

## Pipeline

```
schema ┐                  ┌ correlation ┐
quality├→ cleaning →──────┼ importance  ├→ synthesis → model_rec → feature_eng
stats  │                  └ baseline    ┘
outliers┘
```

Each node runs a Python analysis tool first, then passes the raw result to the LLM to reason over and summarize in plain English.

**Architecture Note:** All Python computations (correlation matrices, outlier IQR bounds, descriptive stats) run on the **full dataset** to ensure statistical accuracy. The data passed to the LLM is intentionally capped (e.g., top 20 strongest correlations, top 15 outlier features) to prevent context window overload and hallucination.

| # | Agent | What it does |
|---|---|---|
| 1 | **Schema** | Shape, column types, null counts per column |
| 2 | **Quality** | Duplicates, missing value %, columns with nulls |
| 3 | **Statistics** | Descriptive stats, skewness, categorical value counts |
| 4 | **Outliers** | IQR-based detection — count, bounds, example values |
| 5 | **Data Cleaning** | Deterministic pre-cleaner + LLM second-pass (see below) |
| 6 | **Correlation** | Pearson matrix, multicollinearity flags, heatmap |
| 7 | **Feature Importance** | Correlation with the chosen target, falling back to variance / entropy |
| 8 | **Baseline Models** | 5-fold CV of a dummy model, a linear model and gradient boosting on the target (no LLM) |
| 9 | **Synthesis** | Full EDA narrative — overview, issues, patterns, recommendations |
| 10 | **Model Recommendation** | Recommends models using the measured baseline scores, flags uncertainty, suggests metrics |
| 11 | **Feature Engineering** | Suggests new features, tests the code, and applies the ones that work |

---

## Data Cleaning Agent (Agent 5)

The cleaning agent uses a **two-pass strategy** to reliably fix common data issues regardless of LLM quality:

### Pass 1 — Deterministic pre-cleaner (always runs, always correct)

Executes in pure Python in a fixed order designed to maximise duplicate detection accuracy:

1. **Null-like string normalisation** — replaces `"unknown"`, `"n/a"`, `"null"`, `"-"`, `"?"`, etc. with `NaN`
2. **Currency / symbol stripping** — converts text columns where at least 90% of values are numeric after stripping `$`, `£`, `€`, `%`, etc. Zero-padded codes (zip codes, `"00123"`) are left as text, and the log says how many values could not be parsed
3. **Duplicate row removal** — runs *after* steps 1–2 so that `"$50,000"` and `"50000"` (the same value in different formats) are correctly identified as duplicates
4. **Missing value imputation** — numeric columns → median; categorical columns → mode. The target column is never imputed

### Pass 2 — LLM agent (handles edge cases)

After the deterministic pass, the LLM reviews what was done and suggests additional cleaning (e.g. outlier capping, title-casing names, phone number normalisation). The generated code is checked (no imports, file access or dunder attributes) and run in a separate Python process with a timeout, with a 2-attempt self-correction loop on errors. This is a guard rail, not a security boundary: for data from people you don't trust, run EDAgent in a container.

The Streamlit dashboard shows a **Before & After** side-by-side table and colour-coded diff cards (rows / missing values / duplicates).

---

## Quickstart

### 1. Choose a model

**Option A: local and free, with Ollama (default).** Install [Ollama](https://ollama.com) and pull a model.
`qwen3.5:4b` is the default when installed: it fits fully in 8 GB of VRAM even with `OLLAMA_NUM_PARALLEL=4`
(about 2 minutes per run on an RTX 4060). `qwen3.5:9b` writes better but spills to CPU on 8 GB, so is ~3x slower:
```bash
ollama pull qwen3.5:4b
```

**Option B: your Claude Pro/Max plan, through Claude Code.** If the `claude` CLI is installed and logged in,
`claude-code:sonnet` and `claude-code:haiku` appear in the model menu. Nothing else to set up. A run makes
about 11 calls and uses roughly 30k tokens (measured on a 2,000-row × 1,500-column CSV), which count toward
your plan's usage limits.

**Option C: the Claude API.** Create an API key at [console.anthropic.com](https://console.anthropic.com). API usage is billed separately from a Claude Pro subscription. Put the key in a `.env` file in the project folder. `.env` is git-ignored, so the key never reaches GitHub:
```
ANTHROPIC_API_KEY=sk-ant-...
```
With Claude, data summaries (not the raw file) are sent to the API. Pick the model in the app's sidebar.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
Opens the EDAgent web dashboard in your browser. Drag and drop your CSV or Excel dataset into the upload area and click **Run EDA Pipeline**.

#### Running in CLI (Alternative)
```bash
# On your own dataset, optionally naming the target column
python pipeline.py your_dataset.csv
python pipeline.py your_dataset.xlsx SalePrice

# Pick the model with an env var
EDAGENT_MODEL=claude-sonnet-5-5 python pipeline.py your_dataset.csv

# With built-in test data
python pipeline.py
```

---

## Example terminal output

```
┌──────────────────────────────────────────────────────────────────┐
│ EDAgent Pipeline                                                 │
│ Dataset: Teen_Mental_Health_Dataset.csv  Rows: 1200  Cols: 13   │
└──────────────────────────────────────────────────────────────────┘

  Running schema agent...
  Running quality agent...
  Running stats agent...
  Running outlier agent...
  Running cleaning agent...
  Running correlation agent...
  Running feature importance agent...
  Running synthesis agent...
  Running model recommendation agent...
  Running feature engineering agent...

┌─────────────────────────────────────────────────────────────────┐
│ Done!                                                           │
│ HTML Report → output/report.html                                │
│ Heatmap     → output/correlation_heatmap.png                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Tool | Role |
|---|---|
| [Streamlit](https://streamlit.io/) | Interactive web dashboard |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Agent orchestration & state management |
| [Ollama](https://ollama.com) | Local LLM inference (any pulled model) |
| [Claude API](https://docs.claude.com) | Optional hosted LLM, much faster |
| [scikit-learn](https://scikit-learn.org/) | Baseline models |
| [pandas](https://pandas.pydata.org/) | Data analysis & cleaning |
| [seaborn / matplotlib](https://seaborn.pydata.org/) | Correlation heatmap |
| [scipy](https://scipy.org/) | Entropy calculation for feature importance |
| [rich](https://rich.readthedocs.io/) | Terminal UI |

---

## Project Structure

```
EDAgent/
├── app.py               # Streamlit web dashboard
├── pipeline.py          # Full LangGraph pipeline (all 11 agents) + CLI
├── sandbox.py           # Runs LLM-written pandas code in a restricted child process
├── test_edagent.py      # Self-checks, no LLM needed: python test_edagent.py
├── style.css            # Dashboard stylesheet (injected via components.html)
├── requirements.txt
├── README.md
└── output/              # Auto-generated, git-ignored
    ├── report.html
    ├── correlation_heatmap.png
    └── clean_data.py
```

---

## Local or Claude?

| | Ollama | Claude Code (Pro plan) | Claude API |
|---|---|---|---|
| Cost | Free | Counts toward plan limits (~30k tokens/run) | Billed per run |
| Privacy | Nothing leaves your machine | Summaries of your data go to Anthropic | Same |
| Quality | Decent with `qwen3.5:4b`, better with `qwen3.5:9b` | Best | Best |

Wide files are fine: each prompt lists at most the 40 most notable columns, so a 1,500-column CSV
fits a local model's context window.

To make Ollama faster: use a smaller model (e.g. `qwen2.5:1.5b`), and set `OLLAMA_NUM_PARALLEL=4` before starting Ollama so the parallel agents don't queue.

---

Created by Varnit :)
