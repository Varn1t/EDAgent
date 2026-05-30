<div align="center">

# EDAgent

Your personal Exploratory Data Analyst + AI Agent

**Drop in a CSV or Excel dataset. Get a complete EDA — automatically.**
<img width="1878" height="867" alt="image" src="https://github.com/user-attachments/assets/5919cf06-af14-452a-90ad-ba3caaf27906" />


An agentic, LLM-powered Exploratory Data Analysis pipeline built with LangGraph and Ollama.  
Nine specialized AI agents analyze your dataset, then generate a polished HTML report — all running **100% locally**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic-blueviolet?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange?style=flat-square)

</div>

---

## What it does

You give it a CSV or Excel dataset. It spins up a **9-stage LangGraph pipeline** where each node is an AI agent that analyzes a different aspect of your data, writes a summary, and passes its findings to the next stage. At the end, you get:

- An **interactive Streamlit dashboard** with tabbed results and live progress
- A **rich, color-coded terminal output** (if run via CLI)
- A **self-contained `report.html`** — dark-themed, browser-ready, heatmap embedded inline
- A **`correlation_heatmap.png`** saved to `output/`
<img width="560" height="605" alt="image" src="https://github.com/user-attachments/assets/96cc9812-567a-40b4-9086-2119cd6626de" />

---

## Pipeline

```
schema → quality → stats → outliers → correlation → importance → synthesis → model_rec → feature_eng
```

Each node runs a Python analysis tool first, then passes the raw result to the LLM to reason over and summarize in plain English.

**Architecture Note:** All Python computations (correlation matrices, outlier IQR bounds, descriptive stats) run on the **full dataset** to ensure statistical accuracy. However, the data passed to the LLM is intentionally capped (e.g., only the top 20 strongest correlations or top 15 outlier features) to prevent context window overload and hallucination.

| # | Agent | What it analyzes |
|---|---|---|
| 1 | **Schema** | Shape, column types, null counts per column |
| 2 | **Quality** | Duplicates, missing value %, columns with nulls |
| 3 | **Statistics** | Descriptive stats, skewness, categorical value counts |
| 4 | **Outliers** | IQR-based detection — count, bounds, example values |
| 5 | **Correlation** | Pearson matrix, multicollinearity flags, heatmap |
| 6 | **Feature Importance** | Target-aware correlation ranking (powered by robust multi-tiered target detection), falling back to variance (numeric) and entropy (categorical) ranking |
| 7 | **Synthesis** | Full EDA narrative — overview, issues, patterns, recommendations |
| 8 | **Model Recommendation** | Infers problem type, recommends models, flags uncertainty, suggests metrics |
| 9 | **Feature Engineering** | Suggests concrete new features: log transforms, bins, interactions, encodings |

**Note:** Feature engineering code is executed in a restricted sandbox (`__builtins__` stripped, only `pd`, `np`, and `df` exposed) with a 2-attempt reflection loop that feeds errors back to the LLM for self-correction. The pipeline uses robust multi-tiered target detection to completely hide the target variable from the LLM prompt and sandbox to prevent data leakage, and aggressively checks newly generated features to reject trivial copies (correlation > 0.999) of existing features.

<img width="482" height="283" alt="image" src="https://github.com/user-attachments/assets/b716f1f9-e3d6-4cd3-b621-0e239cf0036f" />
<img width="1070" height="352" alt="image" src="https://github.com/user-attachments/assets/722b1a1c-5b75-4838-bedd-9df220b3f19d" />


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
This will open the EDAgent web dashboard in your browser. Just drag and drop your CSV or Excel dataset into the upload area!

#### Running in CLI (Alternative)
If you prefer the terminal, you can run the pipeline directly:
```bash
# On your own dataset (CSV or Excel)
python pipeline.py your_dataset.csv
# or
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

  [ schema ]       Running schema agent...
  [ quality ]      Running quality agent...
  [ stats ]        Running stats agent...
  [ outliers ]     Running outlier agent...
  [ correlation ]  Running correlation agent...
  [ importance ]   Running feature importance agent...
  [ synthesis ]    Running synthesis agent...
  [ model-rec ]    Running model recommendation agent...
  [ feature-eng ]  Running feature engineering agent...

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
| [Llama 3.2](https://ollama.com/library/llama3.2) | The underlying language model |
| [pandas](https://pandas.pydata.org/) | Data analysis |
| [seaborn / matplotlib](https://seaborn.pydata.org/) | Correlation heatmap |
| [scipy](https://scipy.org/) | Entropy calculation for feature importance |
| [rich](https://rich.readthedocs.io/) | Terminal UI |

---

## Project Structure

```
EDAgent/
├── app.py               # Streamlit web dashboard
├── pipeline.py          # Full LangGraph pipeline (all 9 agents)
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
