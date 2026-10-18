# file: reduced_error_pruning.py
"""Bottom-up reduced-error pruning for the tree representation in id3.py."""

from copy import deepcopy

import pandas as pd

if __package__:
    from . import id3, info_metrics
else:
    import id3
    import info_metrics


def error_count(tree, df):
    """Count classification mistakes; the target is the last column."""
    target = df.columns[-1]
    return sum(id3.predict(tree, row) != row[target]
               for _, row in df.iterrows())


def reduced_error_pruning(tree, training_df, validation_df):
    """Return a pruned copy, leaving the supplied tree unchanged.

    Visit children before their parent. Replace each subtree with the
    majority training class at that node if doing so does not increase
    validation errors. Class ties use pandas.mode().iloc[0], as in id3().

    Both frames must have identical columns, with the target last. Inputs
    must contain no missing values. Every validation category encountered
    by the tree must have a branch (see the domains argument in main).

    A node reached by no validation rows is pruned on the zero-error tie.
    A node reached by no training rows is left unchanged: its majority
    cannot be determined from these data. Children return their retained error
    counts, so ancestors do not predict the same validation paths again.
    """
    if training_df.empty:
        raise ValueError("The training set must not be empty.")
    if validation_df.empty:
        raise ValueError("The validation set must not be empty.")
    if not training_df.columns.equals(validation_df.columns):
        raise ValueError("Training and validation columns must match in order.")
    if training_df.isna().any().any() or validation_df.isna().any().any():
        raise ValueError("Handle missing values before growing/pruning the tree.")

    target = training_df.columns[-1]

    def branch_mask(df, split, branch):
        if isinstance(split, tuple):
            feature, threshold = split
            if branch == "<":
                return df[feature] < threshold
            if branch == ">=":
                return df[feature] >= threshold
            raise ValueError(f"Invalid continuous branch: {branch!r}")
        return df[split] == branch

    def prune(node, train, valid):
        if not isinstance(node, dict):
            return node, int((valid[target] != node).sum())
        if train.empty:
            # Leave this entire subtree unchanged, scoring it only once.
            return node, error_count(node, valid)

        majority = train[target].mode().iloc[0]
        split, branches = next(iter(node.items()))

        # Post-order traversal: child replacements are retained when
        # evaluating whether to replace this entire subtree.
        subtree_errors = 0
        routed_rows = 0
        for branch, child in list(branches.items()):
            child_valid = valid.loc[branch_mask(valid, split, branch)]
            branches[branch], child_errors = prune(
                child,
                train.loc[branch_mask(train, split, branch)],
                child_valid,
            )
            subtree_errors += child_errors
            routed_rows += len(child_valid)

        if routed_rows != len(valid):
            # Preserve the predictor's KeyError for an unsupported category;
            # silently omitting those rows would make pruning too optimistic.
            error_count(node, valid)
        leaf_errors = int((valid[target] != majority).sum())

        # Counts and error rates give the same comparison on the same rows.
        # Predictions outside this subtree do not change, so evaluating
        # only rows reaching this node is equivalent to scoring the full tree.
        if leaf_errors <= subtree_errors:
            return majority, leaf_errors
        return node, subtree_errors

    return prune(deepcopy(tree), training_df, validation_df)[0]


def train_validation_split(df, validation_fraction=0.2, random_state=42):
    """Make a reproducible random holdout, using row positions, not index labels."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(df) < 2:
        raise ValueError("At least two rows are required.")
    shuffled = df.sample(frac=1, random_state=random_state)
    n_validation = max(1, min(len(df) - 1, round(len(df) * validation_fraction)))
    return shuffled.iloc[n_validation:].copy(), shuffled.iloc[:n_validation].copy()


if __name__ == "__main__":
    df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_13.csv")
    df = df.iloc[:, 1:]  # Drop the ID column.
    continuous_features = set()

    training_df, validation_df = train_validation_split(df)

    # Supply categorical domains so validation-only categories have the
    # parent-majority fallback branches already supported by your id3().
    # This uses feature values only; validation labels never grow the tree.
    # If domains are known in advance, use those instead.
    domains = {
        col: df[col].unique().tolist()
        for col in df.columns[:-1]
        if col not in continuous_features
    }
    tree = id3.id3(
        training_df,
        info_metrics.information_gains,
        domains=domains,
        continuous_features=continuous_features,
    )
    pruned_tree = reduced_error_pruning(tree, training_df, validation_df)

    print("Original tree:", tree)
    print("Pruned tree:", pruned_tree)
    for name, candidate in (("Original", tree), ("Pruned", pruned_tree)):
        errors = error_count(candidate, validation_df)
        print(f"{name} validation error: {errors}/{len(validation_df)} "
              f"({errors / len(validation_df):.2%})")
