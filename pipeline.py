from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Callable, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import datetime
import os
import sys
import re

# ── Windows encoding fix ───────────────────────────────────────────────────
if sys.platform == "win32":
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Progress notification (pluggable for CLI or Streamlit) ─────────────────
_on_step: Optional[Callable] = None

def set_notify(fn: Optional[Callable]):
    """Set the progress callback. Called by app.py to hook into Streamlit."""
    global _on_step
    _on_step = fn

def _notify(msg: str):
    if _on_step:
        _on_step(msg)
    else:
        print(msg)

# ── State ──────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    df: Any
    schema: str
    quality: str
    stats: str
    outliers: str
    correlation: str
    importance: str
    narrative: str
    model_recommendation: str
    feature_engineering: str

# ── LLM ────────────────────────────────────────────────────────────────────

llm = ChatOllama(model="llama3.1")

# ── Python Analysis Tools ──────────────────────────────────────────────────

def run_schema(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": df.dtypes.astype(str).to_dict(),
        "nulls": df.isnull().sum().to_dict()
    }

def run_quality(df: pd.DataFrame) -> dict:
    return {
        "duplicates": int(df.duplicated().sum()),
        "null_cols": df.columns[df.isnull().any()].tolist(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    }

def run_stats(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    return {
        "describe": numeric.describe().round(2).to_dict(),
        "skewness": numeric.skew().round(2).to_dict(),
        "value_counts": {
            col: df[col].value_counts().head(5).to_dict()
            for col in categorical.columns
        }
    }

def run_outliers(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    report = {}
    for col in numeric.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_rows = df[(df[col] < lower) | (df[col] > upper)]
        report[col] = {
            "count": len(outlier_rows),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "outlier_values": outlier_rows[col].tolist()[:10]
        }
    return report

def run_correlation(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation"}
    corr = numeric.corr().round(2)
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png", dpi=150)
    plt.close()
    return {
        "correlation_matrix": corr.to_dict(),
        "heatmap_saved": "output/correlation_heatmap.png"
    }

def run_importance(df: pd.DataFrame) -> dict:
    from scipy.stats import entropy
    importance = {}
    for col in df.select_dtypes(include="number").columns:
        importance[col] = {
            "type": "numeric",
            "variance": round(float(df[col].var()), 4),
            "missing_pct": round(float(df[col].isnull().mean() * 100), 2)
        }
    for col in df.select_dtypes(exclude="number").columns:
        counts = df[col].value_counts(normalize=True)
        importance[col] = {
            "type": "categorical",
            "entropy": round(float(entropy(counts)), 4),
            "unique_values": int(df[col].nunique()),
            "missing_pct": round(float(df[col].isnull().mean() * 100), 2)
        }
    return dict(sorted(
        importance.items(),
        key=lambda x: x[1].get("variance", x[1].get("entropy", 0)),
        reverse=True
    ))

# ── LLM summarizer ─────────────────────────────────────────────────────────

def summarize(result: dict, prompt: str) -> str:
    msg = f"{prompt}\n\nHere are the findings:\n{result}"
    final = llm.invoke([HumanMessage(content=msg)])
    return final.content if final.content else "[No response]"

# ── Nodes ──────────────────────────────────────────────────────────────────

def schema_node(state: GraphState) -> dict:
    _notify("Running schema agent...")
    result = run_schema(state["df"])
    return {"schema": summarize(result, "Summarize these schema findings clearly.")}

def quality_node(state: GraphState) -> dict:
    _notify("Running quality agent...")
    result = run_quality(state["df"])
    return {"quality": summarize(result, "Summarize these data quality issues clearly.")}

def stats_node(state: GraphState) -> dict:
    _notify("Running stats agent...")
    result = run_stats(state["df"])
    return {"stats": summarize(result, "Summarize the key statistical findings clearly.")}

def outlier_node(state: GraphState) -> dict:
    _notify("Running outlier agent...")
    result = run_outliers(state["df"])
    return {"outliers": summarize(result, "Summarize which columns have outliers and how severe.")}

def correlation_node(state: GraphState) -> dict:
    _notify("Running correlation agent...")
    result = run_correlation(state["df"])
    return {"correlation": summarize(result, "Summarize the strongest correlations and multicollinearity concerns.")}

def importance_node(state: GraphState) -> dict:
    _notify("Running feature importance agent...")
    result = run_importance(state["df"])
    return {"importance": summarize(result, "Summarize which features are most important and why.")}

def synthesis_node(state: GraphState) -> dict:
    _notify("Running synthesis agent...")
    prompt = f"""
You are a senior data analyst. Write a concise but complete EDA report covering:
1. Dataset overview
2. Key quality issues and how to fix them
3. Important statistical patterns
4. Outlier summary
5. Feature relationships
6. Top recommendations before modeling

Findings:
SCHEMA: {state['schema']}
QUALITY: {state['quality']}
STATS: {state['stats']}
OUTLIERS: {state['outliers']}
CORRELATION: {state['correlation']}
IMPORTANCE: {state['importance']}
"""
    final = llm.invoke([HumanMessage(content=prompt)])
    return {"narrative": final.content}

def model_rec_node(state: GraphState) -> dict:
    _notify("Running model recommendation agent...")
    prompt = f"""
You are a senior ML engineer. Based on the EDA findings below, recommend which ML model(s) to use.

Your response must include:
1. **Problem Type**: Classification, Regression, or Clustering? Which column is the likely target and why?
2. **Primary Model Recommendation**: Best model with justification based on data characteristics.
3. **Uncertainty Check**: If torn between 2 models, say so explicitly and recommend Random Forest as a neutral benchmark. Specify which metric to compare on and why.
4. **Self-Analysis**: What could go wrong with your recommendation? What assumptions are you making?
5. **Quick-Start Checklist**: 3-5 concrete prep steps before training.

SCHEMA: {state['schema']}
QUALITY: {state['quality']}
STATS: {state['stats']}
CORRELATION: {state['correlation']}
IMPORTANCE: {state['importance']}
NARRATIVE: {state['narrative']}
"""
    final = llm.invoke([HumanMessage(content=prompt)])
    return {"model_recommendation": final.content if final.content else "[No response]"}

def feature_eng_node(state: GraphState) -> dict:
    _notify("Running feature engineering agent...")
    df: pd.DataFrame = state["df"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    skewness = df[numeric_cols].skew().round(2).to_dict() if numeric_cols else {}
    value_counts_sample = {col: df[col].value_counts().head(3).to_dict() for col in categorical_cols}

    prompt = f"""
You are a feature engineering expert. Suggest 5-8 concrete new features to create before modeling.

For each, use this format:
**Feature Name**: <name>
**How to create it**: <pandas/numpy code or clear description>
**Why it helps**: <one sentence rationale>

Consider: log/sqrt transforms for skewed columns, binning, interaction features, encoding improvements, date extraction, null indicator flags.

Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}
Skewness: {skewness}
Value counts sample: {value_counts_sample}
Schema: {state['schema']}
Stats: {state['stats']}
Correlation: {state['correlation']}
Model context: {state['model_recommendation']}
"""
    final = llm.invoke([HumanMessage(content=prompt)])
    return {"feature_engineering": final.content if final.content else "[No response]"}

# ── Graph ──────────────────────────────────────────────────────────────────

_graph = StateGraph(GraphState)
_graph.add_node("schema",      schema_node)
_graph.add_node("quality",     quality_node)
_graph.add_node("stats",       stats_node)
_graph.add_node("outliers",    outlier_node)
_graph.add_node("correlation", correlation_node)
_graph.add_node("importance",  importance_node)
_graph.add_node("synthesis",   synthesis_node)
_graph.add_node("model_rec",   model_rec_node)
_graph.add_node("feature_eng", feature_eng_node)

_graph.set_entry_point("schema")
_graph.add_edge("schema",      "quality")
_graph.add_edge("quality",     "stats")
_graph.add_edge("stats",       "outliers")
_graph.add_edge("outliers",    "correlation")
_graph.add_edge("correlation", "importance")
_graph.add_edge("importance",  "synthesis")
_graph.add_edge("synthesis",   "model_rec")
_graph.add_edge("model_rec",   "feature_eng")
_graph.add_edge("feature_eng", END)

_app = _graph.compile()

# ── Public API ─────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, on_step: Optional[Callable] = None) -> dict:
    """Run the full EDA pipeline on a DataFrame. Returns the result dict."""
    set_notify(on_step)
    return _app.invoke({
        "df":                  df,
        "schema":              "",
        "quality":             "",
        "stats":               "",
        "outliers":            "",
        "correlation":         "",
        "importance":          "",
        "narrative":           "",
        "model_recommendation":"",
        "feature_engineering": ""
    })

def build_html_report(result: dict, dataset_name: str, df: pd.DataFrame) -> str:
    """Generate a self-contained HTML report string."""
    heatmap_tag = ""
    heatmap_path = "output/correlation_heatmap.png"
    if os.path.exists(heatmap_path):
        with open(heatmap_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        heatmap_tag = f'<img src="data:image/png;base64,{img_b64}" class="heatmap" alt="Correlation Heatmap"/>'

    def section(title: str, color: str, content: str, extra: str = "") -> str:
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        content = content.replace("\n", "<br>")
        return f"""
        <div class="section">
            <div class="section-header" style="border-left:4px solid {color}">
                <h2>{title}</h2>
            </div>
            <div class="section-body">{content}</div>
            {extra}
        </div>"""

    body = (
        section("Schema",                      "#60a5fa", result["schema"]) +
        section("Data Quality",                "#f59e0b", result["quality"]) +
        section("Statistics",                  "#818cf8", result["stats"]) +
        section("Outliers",                    "#f87171", result["outliers"]) +
        section("Correlation",                 "#a78bfa", result["correlation"],
                extra=f'<div class="heatmap-wrap">{heatmap_tag}</div>' if heatmap_tag else "") +
        section("Feature Importance",          "#34d399", result["importance"]) +
        section("EDA Narrative",               "#94a3b8", result["narrative"]) +
        section("Model Recommendation",        "#22d3ee", result["model_recommendation"]) +
        section("Feature Engineering",         "#fbbf24", result["feature_engineering"])
    )
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>EDAgent Report — {dataset_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.7;padding:2rem}}
.container{{max-width:960px;margin:0 auto}}
header{{text-align:center;padding:2.5rem 0 1.5rem;border-bottom:1px solid #1e293b;margin-bottom:2rem}}
header h1{{font-size:2rem;color:#f8fafc;font-weight:700}}
header p{{color:#94a3b8;margin-top:.4rem;font-size:.9rem}}
.meta-bar{{display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;margin:1rem 0}}
.badge{{background:#1e293b;border:1px solid #334155;padding:.3rem .8rem;border-radius:999px;font-size:.8rem;color:#94a3b8}}
.badge strong{{color:#e2e8f0}}
.section{{background:#1e293b;border-radius:12px;margin-bottom:1.5rem;overflow:hidden;border:1px solid #334155}}
.section-header{{padding:1rem 1.25rem;background:#162032}}
.section-header h2{{font-size:1rem;font-weight:600;color:#f1f5f9}}
.section-body{{padding:1.25rem;font-size:.9rem;color:#cbd5e1;white-space:pre-wrap}}
.heatmap-wrap{{padding:1rem 1.25rem;text-align:center}}
.heatmap{{max-width:100%;border-radius:8px;border:1px solid #334155}}
strong{{color:#f1f5f9}}
footer{{text-align:center;color:#475569;font-size:.8rem;padding:2rem 0 1rem;border-top:1px solid #1e293b;margin-top:1rem}}
</style></head><body><div class="container">
<header>
  <h1>EDAgent Report</h1>
  <p>Dataset: <strong>{dataset_name}</strong></p>
  <div class="meta-bar">
    <span class="badge">Rows: <strong>{df.shape[0]}</strong></span>
    <span class="badge">Columns: <strong>{df.shape[1]}</strong></span>
    <span class="badge">Generated: <strong>{generated}</strong></span>
  </div>
</header>
{body}
<footer>Generated by EDAgent &middot; LangGraph + Ollama (llama3.1)</footer>
</div></body></html>"""

# ── CLI entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich import box

    console = Console()

    dataset_name = "test_data"
    if len(sys.argv) > 1:
        dataset_name = os.path.basename(sys.argv[1])
        df = pd.read_csv(sys.argv[1])
    else:
        df = pd.DataFrame({
            "age":    [25, None, 30, 25, 45, 200, 22],
            "salary": [30000, 45000, None, 30000, 80000, 999999, 25000],
            "city":   ["Delhi", "Mumbai", "Delhi", "Delhi", "Bangalore", "Delhi", "Mumbai"]
        })

    console.print()
    console.print(Panel.fit(
        f"[bold white]EDAgent Pipeline[/bold white]\n"
        f"[dim]Dataset:[/dim] [green]{dataset_name}[/green]  "
        f"[dim]Rows:[/dim] [cyan]{df.shape[0]}[/cyan]  "
        f"[dim]Cols:[/dim] [cyan]{df.shape[1]}[/cyan]",
        border_style="bright_blue", box=box.ROUNDED
    ))
    console.print()

    result = run_pipeline(df, on_step=lambda msg: console.print(f"  [dim]{msg}[/dim]"))

    sections = [
        ("Schema",               "schema",               "cyan"),
        ("Data Quality",         "quality",              "yellow"),
        ("Statistics",           "stats",                "blue"),
        ("Outliers",             "outliers",             "red"),
        ("Correlation",          "correlation",          "magenta"),
        ("Feature Importance",   "importance",           "green"),
        ("EDA Narrative",        "narrative",            "white"),
        ("Model Recommendation", "model_recommendation", "bright_cyan"),
        ("Feature Engineering",  "feature_engineering",  "bright_yellow"),
    ]

    console.print()
    for title, key, color in sections:
        console.print(Rule(f"[bold {color}]{title}[/bold {color}]", style=color))
        console.print(Markdown(result[key]))
        console.print()

    os.makedirs("output", exist_ok=True)
    html_path = "output/report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html_report(result, dataset_name, df))

    console.print(Rule(style="bright_blue"))
    console.print(Panel.fit(
        f"[bold green]Done![/bold green]\n"
        f"[dim]HTML Report →[/dim] [underline bright_white]{os.path.abspath(html_path)}[/underline bright_white]\n"
        f"[dim]Heatmap     →[/dim] [underline bright_white]{os.path.abspath('output/correlation_heatmap.png')}[/underline bright_white]",
        border_style="green", box=box.ROUNDED
    ))