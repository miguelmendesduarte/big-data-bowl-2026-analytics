"""Module for evaluating a model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""

    auc: float
    logloss: float
    brier: float

    calibration_curve: tuple[np.ndarray, np.ndarray]
    calibration_plot_path: Optional[Path] = None


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    plot_path: Optional[Path] = None,
) -> EvaluationResults:
    """Evaluate classification performance using probabilistic metrics.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        plot_path (Path, optional): File path to save calibration plot.

    Returns:
        EvaluationResults: Structured object with all evaluation outputs.
    """
    auc = roc_auc_score(y_true, y_proba)
    ll = log_loss(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    if plot_path is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, "o-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        plt.title("Calibration Curve")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return EvaluationResults(
        auc=auc,
        logloss=ll,
        brier=brier,
        calibration_curve=(prob_true, prob_pred),
        calibration_plot_path=plot_path if plot_path is not None else None,
    )
