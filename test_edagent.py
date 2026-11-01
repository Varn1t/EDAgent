"""Self-checks for EDAgent's deterministic parts. Run: python test_edagent.py (no LLM needed)."""
import pandas as pd

import pipeline
import sandbox


def test_id_columns():
    df = pd.DataFrame({"user_id": [1, 2, 3], "userId": [1, 1, 2], "width": [1.0, 1.0, 2.0],
                       "humidity": [5, 5, 6], "valid": [1, 0, 1], "price": [1.5, 2.5, 3.5]})
    assert pipeline.is_id_col(df, "user_id") and pipeline.is_id_col(df, "userId")
    for c in ("width", "humidity", "valid", "price"):  # price: unique but float, not an ID
        assert not pipeline.is_id_col(df, c), c


def test_sandbox():
    df = pd.DataFrame({"a": [1, 2, 3]})
    new, result, errors = sandbox.run(df, ["df['b'] = df['a'] * 2\nresult = len(df)"])
    assert not errors and list(new["b"]) == [2, 4, 6] and result == 3
    for bad in ["import os", "x = ().__class__", "pd.read_csv('x')", "df.to_csv('x')",
                "'{0.__class__}'.format(df)", "open('x')"]:
        _, _, errors = sandbox.run(df, [bad])
        assert errors, bad
    _, _, errors = sandbox.run(df, ["while True: pass"], timeout=5)
    assert errors and "timed out" in errors[0]


def test_cleaner():
    df = pd.DataFrame({
        "zip": ["02139", "10001", "02139", None],         # zero-padded: must stay text
        "salary": ["$50,000", "50000", "$50,000", "n/a"],  # → numeric, dupes found after
        "y": [1.0, None, 1.0, 0.0],                        # target: never imputed
    })
    out, log, code = pipeline.clean_deterministic(df, target="y")
    assert out["zip"].dtype != float and "02139" in set(out["zip"])
    assert pd.api.types.is_numeric_dtype(out["salary"])
    assert len(out) == 3 and out["y"].isna().sum() == 1
    # The generated script reproduces the cleaned frame exactly.
    ns = {"pd": pd, "np": pipeline.np, "df": df.copy(),
          "NULL_LIKE": pipeline.NULL_LIKE, "SYMBOLS": pipeline.SYMBOLS}
    exec("\n".join(code), ns)
    pd.testing.assert_frame_equal(ns["df"], out)


def test_ask_guardrails():
    df = pd.DataFrame({"city": ["A", "B", "A"], "salary": [1, 2, 3]})
    off = [
        "OFF_TOPIC",
        "I'm Claude, an AI assistant.",  # chat, no code
        '```python\nresult = "I don\'t have a name, I\'m Claude."\n```',  # the screenshot case
    ]
    for reply in off:
        note, code, result = pipeline.ask_data(lambda p: reply, df, "What is your name?")
        assert note == pipeline.OFF_TOPIC and result is None, reply
    note, _, result = pipeline.ask_data(lambda p: "```python\nresult = df['salary'].sum()\n```", df, "total salary?")
    assert not note and result == 6
    assert pipeline.ask_data(lambda p: "", df, "x" * 501)[0].startswith("Please keep")


def test_data_edit_guardrails():
    df = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})
    ok = df.assign(x=df["x"].clip(upper=90))
    assert pipeline.unsafe_edit(df, ok, "y") is None                       # fixing values is fine
    assert "removed" in pipeline.unsafe_edit(df, df.drop(columns="x"), "y")
    assert "rows" in pipeline.unsafe_edit(df, df.head(50), "y")
    assert "target" in pipeline.unsafe_edit(df, df.assign(y=1), "y")
    assert "added" in pipeline.unsafe_edit(df, df.assign(z=1), "y")
    assert pipeline.unsafe_edit(df, df.assign(z=1), "y", new_cols=True) is None
    assert "existing" in pipeline.unsafe_edit(df, ok.assign(z=1), "y", new_cols=True)

    # Cleaning node rejects a step that drops most rows; feature node rejects target leakage.
    st = {"df": df, "target": "y", "schema": "", "quality": "", "outliers": "", "model_recommendation": "",
          "importance": "", "ask": lambda p: "```python\ndf = df.head(10)\n```"}
    out = pipeline.cleaning_node(st)
    assert len(out["df"]) == 100 and "not applied" in out["cleaning"]
    st["ask"] = lambda p: "```python\ndf['leak'] = df['y'] * 2\n```\n```python\ndf['x2'] = df['x'] ** 2\n```"
    out = pipeline.feature_eng_node(st)
    assert "leak" not in out["df_features"].columns and "x2" in out["df_features"].columns


def test_full_graph_with_fake_llm():
    def fake_ask(prompt):
        if "feature engineering expert" in prompt:
            return "**Feature Name**: x2\n```python\ndf['x2'] = df['x'] ** 2\n```"
        return "No additional cleaning required." if "data cleaning expert" in prompt else "ok"

    pipeline.make_llm, real = (lambda m: fake_ask), pipeline.make_llm
    try:
        df = pd.DataFrame({"x": range(60), "noise": [i % 7 for i in range(60)],
                           "label": ["a" if i < 30 else "b" for i in range(60)]})
        seen = []
        r = pipeline.run_pipeline(df, "fake", "label", on_step=seen.append)
    finally:
        pipeline.make_llm = real
    assert set(seen) == set(pipeline.NODES), seen
    assert "classification" in r["baseline"] and "Gradient boosting" in r["baseline"]
    assert r["heatmap"][:4] == b"\x89PNG"
    assert "x2" in r["df_features"].columns
    assert "<table>" in pipeline.build_html_report(r, "t.csv", df)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
