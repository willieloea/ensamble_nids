"""Bootstrap aggregation and sampling without replacement for classifiers."""

import random

from ..progress import ProgressBar
from ._common import VotingEnsemble, positive_integer, validate_training


def bagging(df, train_model, predict_model, *, n_estimators=10,
            sample_size=None, bootstrap=True, random_state=None, progress=False):
    """Train independent classifiers on random samples and use majority voting.

    The target is the last DataFrame column. train_model(sample_df) must return
    a fresh fitted model; predict_model(model, row) returns one class label.
    Default bootstrap samples contain len(df) rows drawn with replacement.
    For sub-bagging, set bootstrap=False and sample_size smaller than len(df).
    random_state controls sampling only; seed stochastic base learners too.
    progress=True displays completed models; alternatively provide a callback
    progress(completed_models, total_models). Output is disabled by default.
    """
    classes = validate_training(df, train_model, predict_model, n_estimators)
    sample_size = len(df) if sample_size is None else sample_size
    positive_integer(sample_size, "sample_size")
    if not isinstance(bootstrap, bool):
        raise ValueError("bootstrap must be a boolean.")
    if not bootstrap and sample_size > len(df):
        raise ValueError("Without replacement, sample_size cannot exceed len(df).")

    rng = random.Random(random_state)
    positions = range(len(df))
    models = []
    with ProgressBar(n_estimators, "Bagging models", progress) as bar:
        for _ in range(n_estimators):
            indices = (rng.choices(positions, k=sample_size) if bootstrap
                       else rng.sample(positions, k=sample_size))
            sample = df.iloc[indices].copy().reset_index(drop=True)
            models.append(train_model(sample))
            bar.advance()

    return VotingEnsemble(models, [1.0] * n_estimators, classes,
                          df.columns[:-1].tolist(), predict_model)
