from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

def load_final_results():
    path = RESULTS / "FINAL_RESULTS.csv"
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    return dict(zip(df["metric"].astype(str), df["value"]))

def final_metric(name, default=None):
    return load_final_results().get(name, default)

def fmt_pct(value, digits=2):
    if value is None:
        return "-"
    return f"{float(value) * 100:.{digits}f}%"

def fmt_num(value, digits=2):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"
