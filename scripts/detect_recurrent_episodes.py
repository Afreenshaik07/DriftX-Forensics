from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]

def detect_episodes(values, window=200, step=100, alpha=0.01):
    values = np.asarray(values, dtype=float)
    candidates = []

    for start in range(window, len(values) - window + 1, step):
        previous = values[start-window:start]
        current = values[start:start+window]

        ks, p = ks_2samp(previous, current)

        if p < alpha:
            candidates.append({
                "onset": start,
                "ks": float(ks)
            })

    if not candidates:
        return []

    episodes = []
    for candidate in candidates:
        if not episodes or candidate["onset"] - episodes[-1]["last_onset"] > window * 2:
            episodes.append({
                "onset": candidate["onset"],
                "last_onset": candidate["onset"],
                "peak_ks": candidate["ks"]
            })
        else:
            episodes[-1]["last_onset"] = candidate["onset"]
            episodes[-1]["peak_ks"] = max(
                episodes[-1]["peak_ks"],
                candidate["ks"]
            )

    return episodes


path = ROOT / "data" / "interim" / "F08_recurrent.csv"
df = pd.read_csv(path)

episodes = detect_episodes(df["sensor_2"].to_numpy())

print("=== F08 RECURRENT TEMPORAL EPISODES ===")
print("Episodes detected:", len(episodes))

for i, episode in enumerate(episodes, 1):
    print(
        f"Episode {i}: "
        f"onset={episode['onset']} | "
        f"last={episode['last_onset']} | "
        f"peak_KS={episode['peak_ks']:.4f}"
    )
