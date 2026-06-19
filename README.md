<div align="center">

# EDAgent

Your personal Exploratory Data Analyst + AI Agent

**Drop in a CSV or Excel dataset. Get a complete EDA — automatically.**

<img width="1878" height="867" alt="image" src="https://github.com/user-attachments/assets/5919cf06-af14-452a-90ad-ba3caaf27906" />

An agentic, LLM-powered Exploratory Data Analysis pipeline built with LangGraph and Ollama.  
**Ten specialized AI agents** analyze your dataset, clean it, and generate a polished HTML report — all running **100% locally**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-blueviolet?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red?style=flat-square&logo=streamlit)

</div>

---

## What it does

You give it a CSV or Excel dataset. It spins up a **10-stage LangGraph pipeline** where each node is an AI agent that analyzes a different aspect of your data, writes a summary, and passes its findings to the next stage. At the end, you get:

- An **interactive Streamlit dashboard** with tabbed results and a live per-agent progress bar
- A **rich, color-coded terminal output** (if run via CLI)
- A **self-contained `report.html`** — dark-themed, browser-ready, heatmap embedded inline
- A **`correlation_heatmap.png`** saved to `output/`
- A **cleaned dataset** ready to download as CSV — automatically fixed by the cleaning agent

<img width="1867" height="242" alt="image" src="https://github.com/user-attachments/assets/5f972bea-8bee-4579-a4fb-ccb036983c65" />


---

## Pipeline

```
schema → quality → stats → outliers → cleaning → correlation → importance → synthesis → model_rec → feature_eng
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
| 7 | **Feature Importance** | Target-aware correlation ranking, falling back to variance / entropy |
| 8 | **Synthesis** | Full EDA narrative — overview, issues, patterns, recommendations |
| 9 | **Model Recommendation** | Infers problem type, recommends models, flags uncertainty, suggests metrics |
| 10 | **Feature Engineering** | Suggests concrete new features: log transforms, bins, interactions, encodings |

---

## Data Cleaning Agent (Agent 5)

The cleaning agent uses a **two-pass strategy** to reliably fix common data issues regardless of LLM quality:

### Pass 1 — Deterministic pre-cleaner (always runs, always correct)

Executes in pure Python in a fixed order designed to maximise duplicate detection accuracy:

1. **Null-like string normalisation** — replaces `"unknown"`, `"n/a"`, `"null"`, `"-"`, `"?"`, etc. with `NaN`
2. **Currency / symbol stripping** — detects columns where >50% of non-null values look numeric after stripping `$`, `£`, `€`, `%`, etc., then converts them to `float64`
3. **Duplicate row removal** — runs *after* steps 1–2 so that `"$50,000"` and `"50000"` (the same value in different formats) are correctly identified as duplicates
4. **Missing value imputation** — numeric columns → median; categorical columns → mode

### Pass 2 — LLM agent (handles edge cases)

After the deterministic pass, the LLM reviews what was done and suggests additional cleaning (e.g. outlier capping, title-casing names, phone number normalisation). The generated code is sandboxed and tested on a sample before being applied to the full dataset, with a 2-attempt self-correction loop on errors.

The Streamlit dashboard shows a **Before & After** side-by-side table and colour-coded diff cards (rows / missing values / duplicates).

---

## Quickstart

### 1. Prerequisites

Install [Ollama](https://ollama.com) and pull the model:
```bash
ollama pull llama3.2
```

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
# On your own dataset
python pipeline.py your_dataset.csv
python pipeline.py your_dataset.xlsx

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
| [LangChain + Ollama](https://python.langchain.com/) | Local LLM inference |
| [Llama 3.2](https://ollama.com/library/llama3.2) | The underlying language model (2 GB, fast) |
| [pandas](https://pandas.pydata.org/) | Data analysis & cleaning |
| [seaborn / matplotlib](https://seaborn.pydata.org/) | Correlation heatmap |
| [scipy](https://scipy.org/) | Entropy calculation for feature importance |
| [rich](https://rich.readthedocs.io/) | Terminal UI |

---

## Project Structure

```
EDAgent/
├── app.py               # Streamlit web dashboard
├── pipeline.py          # Full LangGraph pipeline (all 10 agents)
├── style.css            # Dashboard stylesheet (injected via components.html)
├── requirements.txt
├── README.md
└── output/              # Auto-generated, git-ignored
    ├── report.html
    └── correlation_heatmap.png
```

---

## Why local?

No API keys. No data sent to the cloud. Everything runs on your machine via Ollama. Swap `llama3.2` for any other Ollama-compatible model by changing one line in `pipeline.py`:

```python
llm = ChatOllama(model="your-model-here")
```

---

Created by Varnit :)
