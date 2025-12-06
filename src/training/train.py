"""Module for training a model."""

import itertools
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]

from ..core.settings import get_settings
from ..io.datasets import CSVReader
from .base import Model
from .evaluate import evaluate_model
from .model import build_xgb_model

settings = get_settings()

FEATURES_TO_EXCLUDE: list[str] = [
    "game_id",
    "play_id",
    "receiver_id",
    "defender_id",
    "target",
    # Based on permutation importance analysis
    # "rec_running_away",
    # "def_back_to_rec",
    # "def_orientation_error",
    # "air_yards",
    # "qb_speed",
    # "closing_speed",
    # "closing_per_yard",
    # "pressure_dist",
    # "qb_to_rec_dist",
]
TARGET_COLUMN: str = "target"


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data from a CSV file.

    Args:
        data_path (Path): Path to the CSV file containing training data.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Features DataFrame and target Series.
    """
    reader = CSVReader()
    df = reader.read(data_path)

    X = df.drop(FEATURES_TO_EXCLUDE, axis=1)
    y = df[TARGET_COLUMN]

    return X, y


def log_feature_importance(
    model: Model, X_test: pd.DataFrame, y_test: pd.Series, plot_path: Path
) -> Path:
    """Compute and save a permutation feature importance plot.

    Args:
        model (Model): Trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        plot_path (Path): Path to save the feature importance plot.

    Returns:
        Path: Path to the saved feature importance plot.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )

    importances = result.importances_mean
    features = X_test.columns
    sorted_idx = importances.argsort()[::-1]  # descending
    sorted_features = features[sorted_idx]
    sorted_importances = importances[sorted_idx]

    plt.style.use("ggplot")
    plt.figure(figsize=(10, max(6, 0.3 * len(sorted_features))))
    plt.barh(sorted_features, sorted_importances, color="#c23728", edgecolor="white")
    plt.xlabel("Permutation Importance Score", fontsize=12, labelpad=15)
    plt.ylabel("Features", fontsize=12, labelpad=15)
    plt.title("Feature Importance Based on Permutation Test", fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return plot_path


def train_model(
    model: Model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Train the model and make predictions on the test set.

    Args:
        model (Model): The machine learning model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.

    Returns:
        tuple[pd.ndarray, pd.ndarray]: Predictions and predicted probabilities.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return y_pred, y_proba


def run_training_pipeline(experiment_name: str) -> None:
    """Run the training pipeline with hyperparameter tuning and MLflow logging.

    Args:
        experiment_name (str): Name of the MLflow experiment.
    """
    mlflow.set_experiment(experiment_name)

    X_train, y_train = load_data(settings.TRAIN_DATA_FILE)
    X_test, y_test = load_data(settings.TEST_DATA_FILE)

    param_grid = settings.XGB_PARAM_GRID.copy()
    param_grid["random_state"] = [settings.XGB_RANDOM_STATE]

    keys, values = zip(*param_grid.items(), strict=False)
    param_sets = [
        dict(zip(keys, vals, strict=False)) for vals in itertools.product(*values)
    ]

    logger.info(f"Training {len(param_sets)} hyperparameter combinations...")

    for params in param_sets:
        with mlflow.start_run():
            mlflow.log_params(params)

            model = build_xgb_model(**params)

            _, y_proba = train_model(model, X_train, y_train, X_test)

            with tempfile.TemporaryDirectory() as tmpdir:
                cc_plot_path = Path(tmpdir) / "calibration_curve.png"
                fi_plot_path = Path(tmpdir) / "feature_importance.png"

                results = evaluate_model(
                    y_true=y_test.to_numpy(),
                    y_proba=y_proba,
                    plot_path=cc_plot_path,
                )
                log_feature_importance(model, X_test, y_test, fi_plot_path)

                mlflow.log_artifact(str(cc_plot_path))
                mlflow.log_artifact(str(fi_plot_path))

            mlflow.log_metrics(
                {
                    "auc": results.auc,
                    "logloss": results.logloss,
                    "brier": results.brier,
                }
            )

            mlflow.sklearn.log_model(model, name="model", input_example=X_test.iloc[:5])

            logger.info(
                f"[Run Completed] AUC: {results.auc:.5f}, | "
                f"LogLoss: {results.logloss:.5f} | Brier: {results.brier:.5f}"
            )


if __name__ == "__main__":
    run_training_pipeline(experiment_name=settings.MLFLOW_EXPERIMENT_NAME)
