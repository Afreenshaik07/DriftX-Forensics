from pathlib import Path
import pandas as pd

COLUMN_NAMES = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    df = df.iloc[:, :26]
    df.columns = COLUMN_NAMES

    return df

def load_training_data(raw_dir: Path) -> pd.DataFrame:
    file_path = raw_dir / "train_FD001.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {file_path}"
        )

    return load_cmapss_file(file_path)

def create_rul_target(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    max_cycle = result.groupby("unit")["cycle"].transform("max")
    result["RUL"] = max_cycle - result["cycle"]

    return result
