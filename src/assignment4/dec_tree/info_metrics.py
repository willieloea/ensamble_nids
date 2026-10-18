# file: info_metrics.py

import math
from collections import Counter
import pandas as pd

if __package__:
    from . import entropy, gini
else:
    import entropy # type: ignore
    import gini # type: ignore

def information_gains(df):
    # 1. Calcualte the entropy of the original dataset wrt the target feature
    target = df.iloc[:, -1]  # last column
    probabilities = target.value_counts(normalize=True)
    initial_entropy = entropy.entropy(probabilities.tolist())

    # 2. For each descriptive feature, partition the dataset according to the
    #    value of that descriptive feature, and calcualte the entropy of the
    #    feature.
    features = list(df.columns)
    descriptive_features = features[:-1]

    feature_remaining_entropies = dict()

    for descriptive_feature in descriptive_features:
        feature_remaining_entropy = 0
        for _, partition in df.groupby(descriptive_feature):
            partition_probs = partition[features[-1]].value_counts(normalize=True)
            partition_entropy = entropy.entropy(partition_probs.tolist())
            weight = len(partition) / len(df)
            feature_remaining_entropy += weight * partition_entropy

        feature_remaining_entropies[descriptive_feature] = \
            feature_remaining_entropy

    # 3. Subtract the entropy measure of step 2 from the entropy measure of step
    #    1 to give the information gain.
    info_gains = dict()
    for descriptive_feature in descriptive_features:
        info_gains[descriptive_feature] = \
            initial_entropy - feature_remaining_entropies[descriptive_feature]

    return info_gains


def _continuous_values(df, feature):
    """Validate a numerical predictor before sorting or making thresholds."""
    values = df[feature]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError(f"Continuous feature {feature!r} must be numeric")
    if values.isna().any() or any(not math.isfinite(v) for v in values):
        raise ValueError(f"Continuous feature {feature!r} must contain finite values")
    return values


def _threshold_between(lower, upper):
    """Use the existing midpoint rule, including adjacent-float fallback."""
    threshold = lower / 2 + upper / 2
    if not lower < threshold <= upper:
        threshold = upper
    return threshold


def candidate_thresholds(df, feature):
    """Midpoints at target transitions, grouping equal feature values first.

    A mixed-label group can have a useful split on either side, so keep those
    boundaries too. Equal feature values must never be split across branches.
    The target is the last column, just as in information_gains().
    """
    _continuous_values(df, feature)

    target = df.columns[-1]
    groups = [
        (value, partition[target].unique().tolist())
        for value, partition in df.groupby(feature, sort=True, observed=True)
    ]
    thresholds = []
    for (lower, lower_labels), (upper, upper_labels) in zip(groups, groups[1:]):
        if len(lower_labels) == len(upper_labels) == 1 and lower_labels == upper_labels:
            continue
        thresholds.append(_threshold_between(lower, upper))
    return thresholds


def _can_score_incrementally(df, values, info_metric):
    """Keep custom scorers and unusual pandas comparison semantics unchanged."""
    if not any(info_metric is metric for metric in
               (information_gains, gini_information_gains, gain_ratios)):
        return False
    target = df.iloc[:, -1]
    if target.isna().any() or isinstance(target.dtype, pd.CategoricalDtype):
        return False
    if pd.api.types.is_float_dtype(values.dtype) and values.dtype.itemsize != 8:
        # Comparing float32 arrays with float midpoints can round the boundary
        # to an endpoint. Preserve the original Boolean-mask scorer for these.
        return False
    if pd.api.types.is_integer_dtype(values.dtype) and not values.empty:
        # Above this range, pandas' int/float comparisons can differ from the
        # mathematical ordering used by a grouped sweep.
        if values.min() < -(2 ** 53) or values.max() > 2 ** 53:
            return False
    return True


def _impurity_from_counts(counts, size, info_metric):
    """Match value_counts' descending-frequency arithmetic and omit zeros."""
    probabilities = [count / size for count in sorted(counts.values(), reverse=True)
                     if count > 0]
    if info_metric is gini_information_gains:
        return gini.gini_index(probabilities)
    return entropy.entropy(probabilities)


def _incremental_continuous_gains(df, feature, info_metric):
    """Sort once, then move whole equal-value groups between class counts."""
    ordered = df.loc[:, [feature, df.columns[-1]]].sort_values(feature, kind="stable")
    codes = pd.factorize(ordered.iloc[:, -1], sort=False)[0].tolist()
    values = ordered[feature]
    # Extension dtypes expose numpy scalars as groupby keys; ordinary numpy
    # dtypes expose Python scalars. Preserve their midpoint arithmetic.
    sorted_values = (values.array if pd.api.types.is_extension_array_dtype(values.dtype)
                     else values.tolist())
    groups = []
    for value, label in zip(sorted_values, codes):
        if not groups or value != groups[-1][0]:
            groups.append((value, Counter()))
        groups[-1][1][label] += 1
    if len(groups) < 2:
        return {}

    right = Counter()
    for _, counts in groups:
        right.update(counts)
    size = len(df)
    initial_impurity = _impurity_from_counts(right, size, info_metric)
    left = Counter()
    left_size = 0
    scores = {}
    for (lower, counts), (upper, next_counts) in zip(groups, groups[1:]):
        left.update(counts)
        right.subtract(counts)
        left_size += sum(counts.values())
        if len(counts) == len(next_counts) == 1 and counts.keys() == next_counts.keys():
            continue
        right_size = size - left_size
        remaining = (
            (left_size / size) * _impurity_from_counts(left, left_size, info_metric)
            + (right_size / size) * _impurity_from_counts(right, right_size, info_metric)
        )
        gain = initial_impurity - remaining
        if info_metric is gain_ratios:
            split_probabilities = [count / size for count in sorted(
                (left_size, right_size), reverse=True)]
            gain /= entropy.entropy(split_probabilities)
        scores[_threshold_between(lower, upper)] = gain
    return scores


def continuous_information_gains(df, feature, info_metric=information_gains):
    """Return {threshold: score} for a continuous feature at the current node.

    For the built-in entropy, Gini, and gain-ratio scorers, sort the feature
    once and update left/right class counts at equal-value group boundaries.
    Candidate selection and descending-frequency impurity arithmetic match
    the original Boolean-partition scorer. Cost is O(n log n + u k log k) for
    n rows, u distinct feature values, and k observed classes, rather than
    constructing and scanning a DataFrame for each candidate threshold.

    Arbitrary scorer callbacks, missing/categorical targets, low-precision
    floats, and integers outside exact float64 range retain the original
    Boolean-mask calculation to preserve pandas comparison behaviour.
    """
    values = _continuous_values(df, feature)
    if _can_score_incrementally(df, values, info_metric):
        return _incremental_continuous_gains(df, feature, info_metric)
    scores = {}
    for threshold in candidate_thresholds(df, feature):
        split_data = df.loc[:, [feature, df.columns[-1]]].copy()
        split_data[feature] = df[feature] >= threshold
        scores[threshold] = info_metric(split_data)[feature]
    return scores


def gini_information_gains(df):
    # 1. Calcualte the gini of the original dataset wrt the target feature
    target = df.iloc[:, -1]  # last column
    probabilities = target.value_counts(normalize=True)
    initial_gini = gini.gini_index(probabilities.tolist())

    # 2. For each descriptive feature, partition the dataset according to the
    #    value of that descriptive feature, and calcualte the gini of the
    #    feature.
    features = list(df.columns)
    descriptive_features = features[:-1]

    feature_remaining_entropies = dict()

    for descriptive_feature in descriptive_features:
        feature_remaining_gini = 0
        for _, partition in df.groupby(descriptive_feature):
            partition_probs = partition[features[-1]].value_counts(normalize=True)
            partition_gini = gini.gini_index(partition_probs.tolist())
            weight = len(partition) / len(df)
            feature_remaining_gini += weight * partition_gini

        feature_remaining_entropies[descriptive_feature] = \
            feature_remaining_gini

    # 3. Subtract the gini measure of step 2 from the gini measure of step
    #    1 to give the information gain.
    info_gains = dict()
    for descriptive_feature in descriptive_features:
        info_gains[descriptive_feature] = \
            initial_gini - feature_remaining_entropies[descriptive_feature]

    return info_gains

def gain_ratios(df):
    """Calculates Gain Ratio for each descriptive feature."""
    # 1. Get the standard Information Gains (the numerator)
    info_gains = information_gains(df)

    gain_ratios = dict()

    # 2. For each feature, calculate the Split Information (the denominator)
    for feature, ig in info_gains.items():
        # Probabilities of the feature's own values: P(d = l)
        feature_probs = df[feature].value_counts(normalize=True).tolist()

        # Denominator: entropy of the descriptive feature
        split_info = entropy.entropy(feature_probs)

        # 3. Handle edge-case: avoid ZeroDivisionError if a feature has only 1 unique value
        if split_info == 0:
            gain_ratios[feature] = 0.0
        else:
            gain_ratios[feature] = ig / split_info

    return gain_ratios


if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_2.csv")
    info_gains = information_gains(df)
    for info_gain in info_gains:
        print(f"{info_gain} IG: {info_gains[info_gain]}")

    gini_info_gains = gini_information_gains(df)
    for gini_info_gain in gini_info_gains:
        print(f"{gini_info_gain} IG: {gini_info_gains[gini_info_gain]}")
