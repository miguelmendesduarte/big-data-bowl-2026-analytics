"""Model-building utilities for the non-completion probability classifier.

The returned model is designed for tabular tracking data
and outputs a probability of non-completion given geometric and kinematic
features (e.g., separation, closing speed, orientation error).
"""

import xgboost as xgb

from .base import Model


def build_xgb_model(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int = 42,
) -> Model:
    """Build an XGBoost classifier.

    Args:
        n_estimators (int): Number of boosting rounds (trees).
        learning_rate (float): Learning rate (shrinkage) for each boosting step.
        max_depth (int): Maximum depth of each tree.
        subsample (float): Fraction of samples to use for each tree.
            Must be in the interval ]0, 1].
        colsample_bytree (float): Fraction of features to use for each tree.
            Must be in the interval ]0, 1].
        reg_lambda (float): L2 regularization term on weights.
        random_state (int, optional): Seed for reproducibility.
            Defaults to 42.

    Raises:
        ValueError: If subsample or colsample_bytree are not in the interval ]0, 1].

    Returns:
        Model: Configured XGBoost classifier.
    """
    if not (0 < subsample <= 1):
        raise ValueError("subsample must be in the interval ]0, 1].")
    if not (0 < colsample_bytree <= 1):
        raise ValueError("colsample_bytree must be in the interval ]0, 1].")

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        eval_metric="auc",
    )
