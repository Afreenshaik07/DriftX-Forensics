from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from src.driftx.config import CONFIG
from src.driftx.data.preprocessing import load_training_data, create_rul_target
from src.driftx.models.rul_model import RULModel
from src.driftx.evaluation.metrics import evaluate_regression


def main():
    raw_dir = Path(CONFIG["paths"]["raw_data"])
    artifacts_dir = Path(CONFIG["paths"]["artifacts"])

    df = load_training_data(raw_dir)
    df = create_rul_target(df)

    feature_columns = [
        column for column in df.columns
        if column not in ["unit", "RUL"]
    ]

    X = df[feature_columns]
    y = df["RUL"]
    groups = df["unit"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CONFIG["split"]["test_size"],
        random_state=CONFIG["project"]["random_seed"]
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RULModel(CONFIG["model"])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = evaluate_regression(y_test, predictions)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifacts_dir / "baseline_rul_model.joblib")

    print("Baseline training completed.")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
