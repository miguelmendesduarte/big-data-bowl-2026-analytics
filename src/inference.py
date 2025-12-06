"""Module for predicting P(non-completion) using a trained model and inference data."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from .core.settings import get_settings
from .io.datasets import CSVReader, CSVWriter
from .training.train import FEATURES_TO_EXCLUDE

settings = get_settings()


def load_inference_features(data_path: Path) -> pd.DataFrame:
    """Load inference data and drop excluded features.

    Args:
        data_path (Path): Path to CSV file with inference data.

    Returns:
        pd.DataFrame: Features ready for prediction.
    """
    reader = CSVReader()

    df = reader.read(data_path)

    X = df.drop(FEATURES_TO_EXCLUDE + ["frame_id"], axis=1, errors="ignore")

    return X


def predict_non_completion_probability(
    X: pd.DataFrame,
    model_path: Path,
) -> pd.Series:
    """Predict non-completion probability for the given features.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        model_path (Path): Path to trained model file (.joblib).

    Returns:
        pd.Series: Predicted non-completion probabilities.
    """
    model = mlflow.sklearn.load_model(str(model_path))

    probs: np.ndarray = model.predict_proba(X)[:, 1]

    return pd.Series(probs, index=X.index)


def run_inference(
    data_path: Path,
    model_path: Path,
    output_path: Path,
) -> None:
    """Run inference on a dataset, return probabilities with metadata and save to CSV.

    Args:
        data_path (Path): Path to CSV file with inference data.
        model_path (Path): Path to trained model file (.joblib).
        output_path (Path): Path to save results CSV.
    """
    X = load_inference_features(data_path)

    probs = predict_non_completion_probability(X, model_path)

    reader = CSVReader()
    df = reader.read(data_path)
    results = df[
        ["game_id", "play_id", "frame_id", "receiver_id", "defender_id"]
    ].copy()
    results["pass_result"] = df["target"]
    results["non_completion_probability"] = probs

    writer = CSVWriter()
    writer.write(results, output_path)


if __name__ == "__main__":
    inference_data_path = settings.INFERENCE_DATA_FILE
    trained_model_path = settings.MODEL_PATH
    output_results_path = settings.INFERENCE_RESULTS_FILE

    run_inference(
        data_path=inference_data_path,
        model_path=trained_model_path,
        output_path=output_results_path,
    )
