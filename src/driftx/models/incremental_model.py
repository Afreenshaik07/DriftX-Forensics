from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


class IncrementalRULModel:
    """Incremental RUL regression using SGDRegressor + partial_fit."""

    def __init__(
        self,
        learning_rate: str = "adaptive",
        eta0: float = 0.001,
        alpha: float = 0.0001,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()

        self.model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0,
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            warm_start=True,
        )

        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        self._validate_data(X, y)

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(
            X_scaled,
            y,
        )

        self.is_fitted = True

    def partial_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        self._validate_data(X, y)

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before partial_fit()."
            )

        X_scaled = self.scaler.transform(X)

        self.model.partial_fit(
            X_scaled,
            y,
        )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been fitted."
            )

        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        predictions = self.predict(X)

        return {
            "mae": float(
                mean_absolute_error(
                    y,
                    predictions,
                )
            ),
            "rmse": float(
                mean_squared_error(
                    y,
                    predictions,
                ) ** 0.5
            ),
            "r2": float(
                r2_score(
                    y,
                    predictions,
                )
            ),
        }

    def save(
        self,
        file_path: Path,
    ) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted model."
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
                f"Incremental model not found: {file_path}"
            )

        payload = joblib.load(file_path)

        self.scaler = payload["scaler"]
        self.model = payload["model"]
        self.is_fitted = payload["is_fitted"]

    @staticmethod
    def _validate_data(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        if X.empty:
            raise ValueError(
                "Feature data is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "Feature and target lengths do not match."
            )

        if X.isna().any().any():
            raise ValueError(
                "Feature data contains missing values."
            )

        if pd.isna(y).any():
            raise ValueError(
                "Target data contains missing values."
            )

        if not np.isfinite(
            X.to_numpy()
        ).all():
            raise ValueError(
                "Feature data contains non-finite values."
            )

        if not np.isfinite(
            y.to_numpy()
        ).all():
            raise ValueError(
                "Target data contains non-finite values."
            )
