from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

print("=== LOCATING FORENSIC DATA GENERATOR ===")

for path in ROOT.rglob("*.py"):
    if ".venv" in path.parts:
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    hits = []
    for keyword in [
        "F01_feature_abrupt",
        "F02_feature_gradual",
        "F03_multi_feature",
        "F04_concept",
        "F05_prediction",
        "F06_prediction_controlled",
        "F07_performance_only",
        "F08_recurrent",
        "F09_no_drift",
    ]:
        if keyword in text:
            hits.append(keyword)

    if hits:
        print(f"\nFILE: {path.relative_to(ROOT)}")
        print("SCENARIOS:", ", ".join(hits))

print("\n=== CSV LOCATIONS ===")

for path in ROOT.rglob("F01*.csv"):
    if ".venv" not in path.parts:
        print(path.relative_to(ROOT))

for path in ROOT.rglob("F02*.csv"):
    if ".venv" not in path.parts:
        print(path.relative_to(ROOT))
