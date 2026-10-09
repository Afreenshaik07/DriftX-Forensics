from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from src.driftx.models.rul_model import RULModel


class ChallengerModel:
    """
    Challenger model used during DriftX-Forensics adaptation.

    The challenger is trained independently from the current
    champion model. It can only become the new champion after
    passing the safety gate.
    """

    def __init__(self):
        self.model = RULModel()
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """
        Train the challenger model on the supplied adaptation data.
        """

        if X.empty:
            raise ValueError(
                "Challenger training data is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "Feature and target lengths do not match."
            )

        if y.isna().any():
            raise ValueError(
                "Challenger target contains missing values."
            )

        self.model.fit(X, y)
        self.is_trained = True

    def predict(
        self,
        X: pd.DataFrame,
    ):
        """
        Generate predictions using the challenger.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Challenger model has not been trained."
            )

        return self.model.predict(X)

    def save(
        self,
        file_path: Path,
    ) -> None:
        """
        Save the trained challenger model.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Cannot save an untrained challenger model."
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save(file_path)

    def load(
        self,
        file_path: Path,
    ) -> None:
        """
        Load a previously trained challenger model.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Challenger model not found: {file_path}"
            )

        self.model.load(file_path)
        self.is_trained = True

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate challenger performance.

        Returns MAE, RMSE and R2.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Challenger model has not been trained."
            )

        if X.empty:
            raise ValueError(
                "Evaluation data is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "Feature and target lengths do not match."
            )

        return self.model.evaluate(X, y)


def create_challenger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: Optional[Path] = None,
) -> ChallengerModel:
    """
    Train and optionally save a challenger model.
    """

    challenger = ChallengerModel()

    challenger.train(
        X=X_train,
        y=y_train,
    )

    if output_path is not None:
        challenger.save(output_path)

    return challenger