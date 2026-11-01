"""Run LLM-written pandas code in a separate Python process, with a timeout.

Each code block is checked first: no imports, no dunder or private
attributes, no file/IO functions. Blocks that pass run in a child process
where only `pd`, `np`, `df` and a few harmless builtins exist.
"""
import ast
import os
import pickle
import re
import subprocess
import sys
import tempfile

# ponytail: denylist + child process, not a real security boundary. If you analyse
# data from people you don't trust, run EDAgent in a container with no network.
_BLOCKED = re.compile(
    r"^(read_\w+|to_(csv|excel|pickle|parquet|json|html|sql|feather|hdf|stata|clipboard|latex|markdown|xml|orc|gbq)"
    r"|eval|query|exec|compile|open|input|breakpoint|getattr|setattr|delattr|globals|locals|vars|type"
    r"|load|loads|save|savez\w*|savetxt|loadtxt|genfromtxt|fromregex|fromfile|tofile|memmap"
    r"|io|os|sys|subprocess|builtins|core|compat|util|lib|ctypeslib|testing|HDFStore|ExcelWriter)$"
)

_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in (
    "len range min max sum abs round sorted reversed list dict set tuple str int float bool "
    "zip enumerate isinstance any all map filter print Exception ValueError KeyError TypeError"
).split()}


def check(code: str) -> None:
    """Raise ValueError/SyntaxError if `code` uses anything off-limits."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        name = (node.id if isinstance(node, ast.Name)
                else node.attr if isinstance(node, ast.Attribute) else None)
        if name and (name.startswith("_") or _BLOCKED.match(name)):
            raise ValueError(f"'{name}' is not allowed")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ValueError("strings containing '__' are not allowed")


def run(df, blocks: list, timeout: int = 60):
    """Run code blocks against `df`. Returns (new_df, result_variable, errors)."""
    errors, ok = [], []
    for b in blocks:
        try:
            check(b)
            ok.append(b)
        except (SyntaxError, ValueError) as e:
            errors.append(f"Code: {b.strip()}\nError: {type(e).__name__}: {e}")
    if not ok:
        return df, None, errors

    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.pkl"), os.path.join(d, "out.pkl")
        with open(src, "wb") as f:
            pickle.dump((df, ok), f)
        try:
            subprocess.run([sys.executable, "-I", os.path.abspath(__file__), src, dst],
                           timeout=timeout, check=True, capture_output=True)
        except subprocess.TimeoutExpired:
            return df, None, errors + [f"Code timed out after {timeout}s."]
        except subprocess.CalledProcessError as e:
            return df, None, errors + [e.stderr.decode(errors="replace")[-800:]]
        with open(dst, "rb") as f:
            new_df, result, run_errors = pickle.load(f)
    return new_df, result, errors + run_errors


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    with open(sys.argv[1], "rb") as f:
        df, blocks = pickle.load(f)
    g = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "df": df}
    errs = []
    for b in blocks:
        try:
            exec(b, g)
        except Exception as e:
            errs.append(f"Code: {b.strip()}\nError: {type(e).__name__}: {e}")
    result = g.get("result")
    try:
        pickle.dumps(result)
    except Exception:
        result = repr(result)
    with open(sys.argv[2], "wb") as f:
        pickle.dump((g["df"], result, errs), f)
