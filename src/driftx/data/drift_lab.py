import numpy as np
import pandas as pd


def inject_feature_drift(
    df: pd.DataFrame,
    features,
    magnitude: float = 0.20
) -> pd.DataFrame:
    result = df.copy()

    for feature in features:
        if feature in result.columns:
            result[feature] = result[feature] * (1.0 + magnitude)

    return result


def inject_gradual_drift(
    df: pd.DataFrame,
    feature: str,
    magnitude: float = 0.30
) -> pd.DataFrame:
    result = df.copy()

    if feature not in result.columns:
        raise ValueError(f"Feature not found: {feature}")

    drift_strength = np.linspace(
        0.0,
        magnitude,
        len(result)
    )

    result[feature] = result[feature] * (1.0 + drift_strength)

    return result


def inject_abrupt_drift(
    df: pd.DataFrame,
    feature: str,
    magnitude: float = 0.30,
    change_point: float = 0.50
) -> pd.DataFrame:
    result = df.copy()

    if feature not in result.columns:
        raise ValueError(f"Feature not found: {feature}")

    start = int(len(result) * change_point)

    result.loc[result.index[start:], feature] = (
        result.loc[result.index[start:], feature]
        * (1.0 + magnitude)
    )

    return result
