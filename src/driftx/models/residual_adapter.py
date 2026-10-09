from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class ResidualIncrementalAdapter:

    def __init__(
        self,
        alpha: float = 0.0001,
        eta0: float = 0.001,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()

        self.model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=alpha,
            learning_rate="adaptive",
            eta0=eta0,
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
        )

        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        residuals: np.ndarray,
    ) -> None:

        if X.empty:
            raise ValueError("Feature data is empty.")

        if len(X) != len(residuals):
            raise ValueError(
                "Feature and residual lengths do not match."
            )

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(
            X_scaled,
            residuals,
        )

        self.is_fitted = True

    def partial_fit(
        self,
        X: pd.DataFrame,
        residuals: np.ndarray,
    ) -> None:

        if X.empty:
            raise ValueError("Feature data is empty.")

        if len(X) != len(residuals):
            raise ValueError(
                "Feature and residual lengths do not match."
            )

        if not self.is_fitted:
            raise RuntimeError(
                "Adapter must be fitted before partial_fit()."
            )

        X_scaled = self.scaler.transform(X)

        self.model.partial_fit(
            X_scaled,
            residuals,
        )

    def predict_correction(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        if not self.is_fitted:
            raise RuntimeError(
                "Adapter has not been fitted."
            )

        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def save(
        self,
        file_path: Path,
    ) -> None:

        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted adapter."
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "scaler": self.scaler,
                "model": self.model,
                "is_fitted": self.is_fitted,
            },
            file_path,
        )

    def load(
        self,
        file_path: Path,
    ) -> None:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Adapter not found: {file_path}"
            )

        payload = joblib.load(file_path)

        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.is_fitted = payload["is_fitted"]
