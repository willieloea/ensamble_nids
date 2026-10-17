"""Multiclass AdaBoost (SAMME) using weighted resampling of training rows."""

import math
import random
import warnings

from ..progress import ProgressBar
from ._common import VotingEnsemble, validate_training


def boosting(df, train_model, predict_model, *, n_estimators=50,
             learning_rate=1.0, random_state=None, progress=False):
    """Fit SAMME classifiers, supporting learners without sample-weight APIs.

    train_model(sample_df) returns a fresh model; predict_model(model, row)
    returns one class label. The target must be last. Each learner trains on
    len(df) replacement draws from the current weight distribution. Its error
    is measured on ALL original training rows using those weights, not on a
    validation set or just the resampled rows.

    For K classes, alpha = learning_rate * (log((1-error)/error) + log(K-1)).
    Misclassified rows receive a factor exp(alpha), then weights normalise.
    For two classes this is equivalent to binary AdaBoost's symmetric update
    and half-log vote convention. This is resampling-based AdaBoost, not exact
    weighted fitting of the base learner.

    Stop on a perfect classifier, returning it alone (the limiting dominant
    vote). Reject error >= 1-1/K: raise if first, otherwise warn and stop with
    the accepted ensemble. random_state controls only the resampling.
    progress=True displays evaluated boosting rounds; alternatively provide a
    callback progress(completed_rounds, total_rounds). Early stopping can end
    below the requested round count. Output is disabled by default.
    """
    classes = validate_training(df, train_model, predict_model, n_estimators)
    if len(classes) < 2:
        raise ValueError("Boosting requires at least two target classes.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")

    rng = random.Random(random_state)
    n_rows = len(df)
    rows = df.iloc[:, :-1].to_dict(orient="records")
    targets = df.iloc[:, -1].tolist()
    class_set = set(classes)
    log_weights = [-math.log(n_rows)] * n_rows
    ensemble = VotingEnsemble([], [], classes, df.columns[:-1].tolist(),
                              predict_model)

    with ProgressBar(n_estimators, "Boosting rounds", progress) as bar:
        for _ in range(n_estimators):
            weights = [math.exp(value) for value in log_weights]
            indices = rng.choices(range(n_rows), weights=weights, k=n_rows)
            model = train_model(df.iloc[indices].copy().reset_index(drop=True))
            predictions = [predict_model(model, row) for row in rows]
            if any(label not in class_set for label in predictions):
                raise ValueError("Base learner predicted a class absent from training labels.")
            bar.advance()
            incorrect = [prediction != target
                         for prediction, target in zip(predictions, targets)]
            error = math.fsum(weight for weight, wrong in zip(weights, incorrect)
                              if wrong) / math.fsum(weights)

            if error >= 1 - 1 / len(classes):
                message = "Base learner must have weighted error below 1 - 1 / n_classes."
                if not ensemble.estimators:
                    raise ValueError(message)
                warnings.warn(message + " Stopping with the accepted learners.",
                              RuntimeWarning, stacklevel=2)
                break
            if not any(incorrect):
                ensemble.estimators = [model]
                ensemble.estimator_weights = [1.0]
                ensemble.estimator_errors = [0.0]
                break

            # Compute log(error) directly: tiny incorrect weights may underflow
            # to zero above, but that does not make the classifier perfect.
            wrong_logs = [value for value, wrong in zip(log_weights, incorrect) if wrong]
            largest_wrong = max(wrong_logs)
            log_error = (largest_wrong
                         + math.log(math.fsum(math.exp(value - largest_wrong)
                                              for value in wrong_logs))
                         - math.log(math.fsum(weights)))
            alpha = learning_rate * (math.log1p(-error) - log_error
                                     + math.log(len(classes) - 1))
            if not math.isfinite(alpha):
                raise ValueError("learning_rate produced a nonfinite vote weight.")
            ensemble.estimators.append(model)
            ensemble.estimator_weights.append(alpha)
            ensemble.estimator_errors.append(error)

            # Keep weights in log space to avoid overflow in exp(alpha).
            updated = [value + (alpha if wrong else 0.0)
                       for value, wrong in zip(log_weights, incorrect)]
            largest = max(updated)
            normaliser = math.log(math.fsum(math.exp(value - largest)
                                           for value in updated))
            log_weights = [(value - largest) - normaliser for value in updated]

    return ensemble
