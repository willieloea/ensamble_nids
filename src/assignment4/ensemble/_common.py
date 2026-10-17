"""Shared validation and hard voting for classification ensembles."""

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import pandas as pd


@dataclass
class VotingEnsemble:
    """Trained classifiers and their vote weights.

    predict_model(model, row) must return one hashable training-class label.
    Ties follow classes, which records first appearance in the training set.
    estimator_errors contains weighted training errors for boosting only.
    """

    estimators: list[Any]
    estimator_weights: list[float]
    classes: list[Any]
    feature_names: list[str]
    predict_model: Callable
    estimator_errors: list[float] = field(default_factory=list)

    def predict(self, row):
        """Predict one observation supplied as a mapping or pandas Series."""
        votes = dict.fromkeys(self.classes, 0.0)
        for model, weight in zip(self.estimators, self.estimator_weights):
            label = self.predict_model(model, row)
            if label not in votes:
                raise ValueError(f"Base learner predicted unknown class: {label!r}")
            votes[label] += weight
        return max(votes, key=votes.get)

    def predict_many(self, df):
        """Return predictions as a Series preserving the input row index."""
        # Dictionaries avoid iterrows() coercing integer features to floats.
        rows = df.loc[:, self.feature_names].to_dict(orient="records")
        return pd.Series([self.predict(row) for row in rows], index=df.index,
                         name="prediction", dtype=object)


def validate_training(df, train_model, predict_model, n_estimators):
    """Check the shared interface and return classes in appearance order."""
    if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) < 2:
        raise ValueError("Provide a nonempty DataFrame with predictors and target last.")
    if not df.columns.is_unique:
        raise ValueError("Training columns must have unique names.")
    if df.iloc[:, -1].isna().any():
        raise ValueError("Target labels must not be missing.")
    if not callable(train_model) or not callable(predict_model):
        raise TypeError("train_model and predict_model must be callable.")
    positive_integer(n_estimators, "n_estimators")
    return df.iloc[:, -1].unique().tolist()


def positive_integer(value, name):
    """Reject booleans, fractional counts, and nonpositive sample counts."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
