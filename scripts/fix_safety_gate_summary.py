from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "results" / "safety_gate_stress_test.csv"
out = ROOT / "results" / "safety_gate_stress_test_summary.csv"

df = pd.read_csv(src)

def extract_decision(value):
    value = str(value)
    if "decision='PROMOTE'" in value:
        return "PROMOTE"
    if "decision='REJECT'" in value:
        return "REJECT"
    return "UNKNOWN"

df["gate_decision"] = df["decision"].apply(extract_decision)

summary = (
    df.groupby("gate_decision")
      .size()
      .reset_index(name="count")
)

summary.to_csv(out, index=False)

print("=== SAFETY-GATE VALIDATION ===")
print(df[["test", "gate_decision"]].to_string(index=False))
print()
print("Promoted:", int((df["gate_decision"] == "PROMOTE").sum()))
print("Rejected :", int((df["gate_decision"] == "REJECT").sum()))
print("Unknown  :", int((df["gate_decision"] == "UNKNOWN").sum()))
print()
print("Safety rejection rate:",
      f"{(df['gate_decision'] == 'REJECT').mean() * 100:.1f}%")
print(f"Saved: {out}")
