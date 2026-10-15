# file: id3.py

# Iterative Dichotomizer 3
import pandas as pd
if __package__:
    from . import info_metrics
else:
    import info_metrics # type: ignore

def id3(
    df: pd.DataFrame,
    info_metric,
    domains=None,
    parent_majority=None,
    continuous_features=None,
):
    """
    Iteratively partitions the dataset based on which feature is most
    informative and grows branches for each of the descriptive feature levels.

    Explicitly list continuous_features to use binary threshold tests. The
    target must be the last column. Categorical nodes use {feature: branches};
    continuous nodes use {(feature, threshold): {"<": subtree, ">=": subtree}}.
    """
    continuous_features = set(continuous_features or ())
    if domains is None:
        unknown = continuous_features - set(df.columns[:-1])
        if unknown:
            raise ValueError(f"Unknown continuous features: {sorted(unknown)}")
        domains = {
            col: df[col].unique().tolist()
            for col in df.columns[:-1]
            if col not in continuous_features
        }

    if df.empty:
        # a decision tree consisting of a leaf node with the label of the
        # majority target level of the dataset of the immediate parent node
        return parent_majority

    target_feature = list(df.columns)[-1]
    descriptive_features = list(df.columns)[:-1]

    current_majority = df[target_feature].mode()[0]

    if len(df[target_feature].unique()) == 1:
        # a decision tree consisting of a leaf node with label C
        # return df[target_feature].unique()
        return df[target_feature].iloc[0]  # returns 'conifer', not an array


    elif len(descriptive_features) == 0:
        # a decision tree consisting of a leaf node with the label of the
        # majority target level in D
        return df[target_feature].mode()[0]

    else:
        # 5. find the most informative descriptive feature
        categorical_features = [
            feature for feature in descriptive_features
            if feature not in continuous_features
        ]
        data = info_metric(df[categorical_features + [target_feature]])
        thresholds = {}
        for feature in descriptive_features:
            if feature not in continuous_features:
                continue
            scores = info_metrics.continuous_information_gains(df, feature, info_metric)
            if scores:
                threshold = max(scores, key=scores.get)
                thresholds[feature] = threshold
                data[feature] = scores[threshold]

        # Continuous columns with no candidate boundary cannot split this node.
        if not data:
            return current_majority
        best_feature = max(data, key=data.get)

        if best_feature in continuous_features:
            threshold = thresholds[best_feature]
            tree = {(best_feature, threshold): {}}
            above = df[best_feature] >= threshold
            for branch, mask in (("<", ~above), (">=", above)):
                # Keep continuous features: a later node can use another threshold.
                tree[(best_feature, threshold)][branch] = id3(
                    df=df[mask],
                    info_metric=info_metric,
                    domains=domains,
                    parent_majority=current_majority,
                    continuous_features=continuous_features,
                )
            return tree

        # 6. make a new node and label it with the best descriptive feature
        tree = {best_feature: {}}

        # 7. partition D based on the best descriptive feature
        # vals = [val for val, _ in df.groupby(best_feature)]
        for val in domains[best_feature]:
            partition = df[df[best_feature] == val]
            # 8. remove the best descriptive feature from the dataset
            reduced_partition = partition.drop(columns=[best_feature])

            subtree = id3(
                df=reduced_partition,
                info_metric=info_metric,
                domains=domains,
                parent_majority=current_majority,
                continuous_features=continuous_features,
            )
            tree[best_feature][val] = subtree

        return tree


def predict(tree, row):
    """Follow categorical or threshold branches for one row (Series or dict)."""
    while isinstance(tree, dict):
        split, branches = next(iter(tree.items()))
        if isinstance(split, tuple):
            feature, threshold = split
            branch = ">=" if row[feature] >= threshold else "<"
        else:
            branch = row[split]
        tree = branches[branch]
    return tree

if __name__ == '__main__':
    # Read the dataset
    df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_13.csv")
    tree = id3(df.iloc[:, 1:], info_metrics.information_gains) # dropping the ID column
    print("Original tree:", tree)
    # {feature_name: {value_1: subtree_or_leaf, value_2: subtree_or_leaf}}.
