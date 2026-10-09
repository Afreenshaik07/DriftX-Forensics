from pathlib import Path
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

class RULModel:
    def __init__(self, params=None):
        params = params or {}

        self.model = HistGradientBoostingRegressor(
            learning_rate=params.get("learning_rate", 0.08),
            max_iter=params.get("max_iter", 300),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            l2_regularization=params.get("l2_regularization", 0.1),
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
        return self
