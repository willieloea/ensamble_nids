# file: draw_tree.py

import pandas as pd
import uuid
import graphviz
if __package__:
    from . import id3, info_metrics
else:
    import id3
    import info_metrics

def print_ascii_tree(tree, indent=""):
    """Recursively prints a nested dict tree with branch symbols."""
    if not isinstance(tree, dict):
        # Extract scalar string in case it's still a StringArray/Series
        val = (
            tree.iloc[0]
            if hasattr(tree, "iloc")
            else (tree[0] if hasattr(tree, "__iter__") and not isinstance(tree, str) else tree)
        )
        print(f" ──> [{val}]")
        return

    # A node is {FEATURE: {VALUE: SUBTREE, ...}}
    for feature, branches in tree.items():
        branch_items = list(branches.items())
        for idx, (val, subtree) in enumerate(branch_items):
            is_last = idx == len(branch_items) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if isinstance(feature, tuple):
                name, threshold = feature
                test = f"{name} {val} {threshold:g}"
            else:
                test = f"{feature} == {val}"
            print(f"{indent}{connector}{test}", end="")

            if isinstance(subtree, dict):
                print()
                print_ascii_tree(subtree, indent + extension)
            else:
                leaf = (
                    subtree.iloc[0] if hasattr(subtree, "iloc") else subtree
                )
                print(f" ──> [{leaf}]")

def draw_tree(tree, file_name="decision_tree"):
    dot = graphviz.Digraph(
        comment="Decision Tree",
        node_attr={"shape": "box", "style": "filled,rounded"},
    )

    def add_nodes_edges(sub_tree, parent_id=None, edge_label=None):
        node_id = str(uuid.uuid4())

        if not isinstance(sub_tree, dict):
            # Leaf node (colored green)
            leaf_val = (
                str(sub_tree.iloc[0])
                if hasattr(sub_tree, "iloc")
                else str(sub_tree)
            )
            dot.node(
                node_id,
                leaf_val,
                fillcolor="#c8e6c9",
                shape="ellipse",
                style="filled",
            )
            if parent_id:
                dot.edge(parent_id, node_id, label=str(edge_label))
            return

        # Decision node (colored blue)
        feature_name = list(sub_tree.keys())[0]
        label = feature_name[0] if isinstance(feature_name, tuple) else feature_name
        dot.node(node_id, label, fillcolor="#bbdefb")

        if parent_id:
            dot.edge(parent_id, node_id, label=str(edge_label))

        for branch_val, child in sub_tree[feature_name].items():
            edge_label = (
                f"{branch_val} {feature_name[1]:g}"
                if isinstance(feature_name, tuple) else branch_val
            )
            add_nodes_edges(child, parent_id=node_id, edge_label=edge_label)

    add_nodes_edges(tree)
    # Saves as 'decision_tree.png' and opens it
    dot.render(file_name, format="png", cleanup=True, view=True)


if __name__ == '__main__':
    df = pd.read_csv("~/Laboratory/datasets/foml_datasets/tab4_3.csv")
    tree = id3.id3(df.iloc[:, 1:], info_metrics.information_gains)
    draw_tree(tree, "information_gain")

    tree = id3.id3(df.iloc[:, 1:], info_metrics.gain_ratios)
    draw_tree(tree, "gain_ratio")
