from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]

def detect_recurrent_drift(values, window=200, step=100, alpha=0.01, strength=0.50, min_episodes=2):
    values = np.asarray(values, dtype=float)
    candidates = []

    for start in range(window, len(values) - window + 1, step):
        previous = values[start-window:start]
        current = values[start:start+window]
        ks, p = ks_2samp(previous, current)

        if p < alpha and ks >= strength:
            candidates.append((start, float(ks)))

    episodes = []

    for onset, ks in candidates:
        if not episodes or onset - episodes[-1]["last"] > window * 2:
            episodes.append({
                "onset": onset,
                "last": onset,
                "peak_ks": ks
            })
        else:
            episodes[-1]["last"] = onset
            episodes[-1]["peak_ks"] = max(episodes[-1]["peak_ks"], ks)

    recurrent = len(episodes) >= min_episodes

    return {
        "recurrent": recurrent,
        "episode_count": len(episodes),
        "episodes": episodes
    }


path = ROOT / "data" / "interim" / "F08_recurrent.csv"
df = pd.read_csv(path)

result = detect_recurrent_drift(df["sensor_2"].to_numpy())

print("=== RECURRENT DRIFT DETECTOR ===")
print("Recurrent drift:", result["recurrent"])
print("Strong episodes:", result["episode_count"])

for i, episode in enumerate(result["episodes"], 1):
    print(
        f"Episode {i}: "
        f"onset={episode['onset']} | "
        f"last={episode['last']} | "
        f"peak_KS={episode['peak_ks']:.4f}"
    )
