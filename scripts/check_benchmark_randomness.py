from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "scripts" / "run_forensic_benchmark.py"

text = path.read_text(encoding="utf-8")

print("=== CHECKING BENCHMARK RANDOMNESS ===")

patterns = [
    r"np\.random\.seed\([^)]+\)",
    r"np\.random\.default_rng\([^)]+\)",
    r"np\.random\.normal\(",
    r"np\.random\.uniform\(",
    r"np\.random\.rand\(",
    r"np\.random\.randn\(",
    r"random\.seed\([^)]+\)",
    r"random\.(?:random|uniform|gauss|choice)\(",
]

for pattern in patterns:
    matches = re.findall(pattern, text)
    if matches:
        print(f"{pattern}: {len(matches)} occurrence(s)")
        for item in matches[:5]:
            print("  ", item)

print()
print("Benchmark seed declarations:")

for line_no, line in enumerate(text.splitlines(), 1):
    if re.search(r"\b(seed|random_state|rng)\b", line, re.I):
        print(f"{line_no}: {line.strip()}")

print()
print("Scenario-generation references:")

for line_no, line in enumerate(text.splitlines(), 1):
    if any(
        word in line.lower()
        for word in [
            "generate",
            "scenario",
            "drift_lab",
            "synthetic",
            "noise",
            "magnitude",
        ]
    ):
        print(f"{line_no}: {line.strip()}")
