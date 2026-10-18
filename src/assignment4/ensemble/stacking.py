"""Holdout stacking (Topic 7, slides 22–23) and optional cross-fitted stacking."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import random
from typing import Any

import pandas as pd

from ..progress import ProgressBar
from ._common import positive_integer, validate_training


META_TARGET = '__stacking_target__'


def _check_label(label, classes):
    """Reject predictions outside the training label vocabulary."""
    try:
        valid = label in classes
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f'Classifier must predict one known class label, got {label!r}.')


def _meta_columns(n_models, n_classes, encoding):
    if encoding == 'labels':
        return [f'base_{i}' for i in range(n_models)]
    return [f'base_{i}_class_{j}' for i in range(n_models) for j in range(n_classes)]


def _encode(predictions, classes, encoding):
    """Represent base predictions with a fixed schema shared by both stages."""
    if encoding == 'labels':
        return list(predictions)
    return [int(prediction == label) for prediction in predictions for label in classes]


@dataclass
class StackingEnsemble:
    """Fitted base models and a meta-model that consumes only their predictions.

    estimators and predict_models are ordered mappings with matching names.
    Callbacks must remain available when restoring a pickled ensemble.
    """

    estimators: dict[str, Any]
    predict_models: dict[str, Callable]
    meta_estimator: Any
    predict_meta_model: Callable
    classes: list[Any]
    feature_names: list[str]
    meta_feature_names: list[str]
    meta_encoding: str
    training_scheme: str

    def transform(self, row):
        """Return the meta-feature mapping for one mapping or pandas Series."""
        features = {name: row[name] for name in self.feature_names}
        predictions = []
        vocabulary = set(self.classes)
        for name, estimator in self.estimators.items():
            label = self.predict_models[name](estimator, dict(features))
            _check_label(label, vocabulary)
            predictions.append(label)
        return dict(zip(self.meta_feature_names,
                        _encode(predictions, self.classes, self.meta_encoding)))

    def transform_many(self, df):
        """Return meta-features in the input row order, preserving its index."""
        rows = df.loc[:, self.feature_names].to_dict(orient='records')
        return pd.DataFrame([self.transform(row) for row in rows],
                            columns=self.meta_feature_names, index=df.index)

    def predict(self, row):
        """Predict one observation by passing base predictions to the meta-model."""
        label = self.predict_meta_model(self.meta_estimator, self.transform(row))
        _check_label(label, set(self.classes))
        return label

    def predict_many(self, df):
        """Return predictions as a Series preserving the input row index."""
        meta = self.transform_many(df)
        predictions = []
        vocabulary = set(self.classes)
        for row in meta.to_dict(orient='records'):
            label = self.predict_meta_model(self.meta_estimator, row)
            _check_label(label, vocabulary)
            predictions.append(label)
        return pd.Series(predictions, index=df.index, name='prediction', dtype=object)


def _partitions(targets, classes, n_folds, holdout_fraction, rng):
    """Make stratified row-position splits, independent of DataFrame index labels."""
    positions = {label: [] for label in classes}
    for position, label in enumerate(targets):
        positions[label].append(position)
    required = 2 if n_folds is None else n_folds
    if any(len(group) < required for group in positions.values()):
        raise ValueError(f'Every class needs at least {required} observations for this split; '
                         'use more distinct training examples or fewer folds.')
    if n_folds is None:
        train, holdout = [], []
        for group in positions.values():
            rng.shuffle(group)
            count = max(1, min(len(group) - 1, round(len(group) * holdout_fraction)))
            holdout.extend(group[:count])
            train.extend(group[count:])
        rng.shuffle(train)
        rng.shuffle(holdout)
        return [(train, holdout)]

    folds = [[] for _ in range(n_folds)]
    offset = 0
    for group in positions.values():
        rng.shuffle(group)
        for i, position in enumerate(group):
            folds[(offset + i) % n_folds].append(position)
        offset = (offset + len(group)) % n_folds
    splits = []
    for holdout in folds:
        held_out = set(holdout)
        train = [position for position in range(len(targets)) if position not in held_out]
        splits.append((train, holdout))
    return splits


def stacking(df, base_learners, train_meta_model, predict_meta_model, *,
             holdout_fraction=0.2, n_folds=None, meta_encoding='one_hot',
             random_state=None, progress=False):
    """Train a generic classification stack without in-sample meta-features.

    The target is the last column. base_learners maps each model name to
    (train_model, predict_model), using train_model(sample_df) -> fresh model
    and predict_model(model, feature_row) -> class label, as in bagging.
    train_meta_model(meta_df) and predict_meta_model(model, meta_row) use the
    same conventions. The meta target is named __stacking_target__ and is last.

    By default, stratify a separate holdout for meta training, as in Topic 7
    slides 22–23. Base learners are retained from their training partition;
    they are not subsequently refitted on the meta holdout. Each class needs
    at least two rows; class-wise rounding makes the fraction approximate.

    With n_folds >= 2, use out-of-fold predictions for every training row,
    train the meta-model, then refit each base learner on all df for inference.
    Each class needs at least n_folds observations. holdout_fraction is unused.

    Meta-features are one-hot class predictions by default; 'labels' supplies
    one categorical prediction column per base model, matching the slide's
    table. No original predictors are passed to the meta-learner. Probability
    outputs and regression are not supported by this class-label interface.

    Preprocessing/selection learned from data belongs inside each base trainer
    so it is fitted only on that split. Callbacks must handle unseen feature
    categories and return fresh fitted models. random_state controls splits,
    not randomness in the callbacks. progress=True displays completed fits;
    a progress(completed_fits, total_fits) callback is also accepted.
    """
    classes = validate_training(df, train_meta_model, predict_meta_model, 1)
    if not isinstance(base_learners, Mapping) or not base_learners:
        raise ValueError('base_learners must be a nonempty mapping of names to (train, predict).')
    for name, callbacks in base_learners.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Base learner names must be nonempty strings.')
        if (not isinstance(callbacks, (tuple, list)) or len(callbacks) != 2
                or not all(callable(callback) for callback in callbacks)):
            raise TypeError(f'{name}: provide a pair of callable train and predict functions.')
    if meta_encoding not in ('one_hot', 'labels'):
        raise ValueError("meta_encoding must be 'one_hot' or 'labels'.")
    if n_folds is None:
        if not math.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
            raise ValueError('holdout_fraction must be finite and strictly between zero and one.')
    else:
        positive_integer(n_folds, 'n_folds')
        if n_folds < 2:
            raise ValueError('n_folds must be at least two.')

    splits = _partitions(df.iloc[:, -1].tolist(), classes, n_folds,
                         holdout_fraction, random.Random(random_state))
    feature_names = df.columns[:-1].tolist()
    rows = df.iloc[:, :-1].to_dict(orient='records')
    columns = _meta_columns(len(base_learners), len(classes), meta_encoding)
    predictions = [[None] * len(base_learners) for _ in range(len(df))]
    estimators = {}
    predictors = {name: callbacks[1] for name, callbacks in base_learners.items()}
    vocabulary = set(classes)
    total_fits = len(base_learners) * (1 if n_folds is None else n_folds + 1) + 1
    with ProgressBar(total_fits, 'Stacking fits', progress) as bar:
        for train_positions, held_out_positions in splits:
            for i, (name, (train_model, predict_model)) in enumerate(base_learners.items()):
                sample = df.iloc[train_positions].copy().reset_index(drop=True)
                model = train_model(sample)
                for position in held_out_positions:
                    # Fresh mappings isolate callbacks that modify their input.
                    label = predict_model(model, dict(rows[position]))
                    _check_label(label, vocabulary)
                    predictions[position][i] = label
                if n_folds is None:
                    estimators[name] = model
                bar.advance()

        meta_positions = splits[0][1] if n_folds is None else list(range(len(df)))
        meta = pd.DataFrame([_encode(predictions[position], classes, meta_encoding)
                             for position in meta_positions], columns=columns)
        meta[META_TARGET] = df.iloc[meta_positions, -1].tolist()
        meta_estimator = train_meta_model(meta.copy())
        bar.advance()
        if n_folds is not None:
            for name, (train_model, _) in base_learners.items():
                estimators[name] = train_model(df.copy().reset_index(drop=True))
                bar.advance()

    return StackingEnsemble(estimators, predictors, meta_estimator, predict_meta_model,
                            classes, feature_names, columns, meta_encoding,
                            'holdout' if n_folds is None else f'{n_folds}-fold')
