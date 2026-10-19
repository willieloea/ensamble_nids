"""Generate separate subfigure PDF panels conforming to writingRules.pdf (WR25, WR29, WR31)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Normal",
    "Recon.",
    "Backdoor",
    "DoS",
    "Exploits",
    "Analysis",
    "Fuzzers",
    "Worms",
    "Shellcode",
    "Generic",
]


def plot_learning_curves():
    """Figure 1 panels: (a) validation scores vs sample size, (b) training accuracy ceiling."""
    agg = pd.read_csv(
        ROOT / "experiments/overnight-learning-curves/results/analysis/method_comparison.csv"
    )
    conflicts = pd.read_csv(
        ROOT / "experiments/overnight-learning-curves/results/analysis/training_conflicts.csv"
    )

    sizes = [2000, 5000, 10000, 20000, 50000]
    size_labels = ["2k", "5k", "10k", "20k", "50k"]

    raw_rows = [agg[agg["configuration"] == f"Raw tree {s}"].iloc[0] for s in size_labels]
    pruned_rows = [agg[agg["configuration"] == f"Pruned tree {s}"].iloc[0] for s in size_labels]

    raw_f1 = [r["validation_macro_f1_mean"] for r in raw_rows]
    raw_f1_sd = [r["validation_macro_f1_sd"] for r in raw_rows]
    pruned_f1 = [r["validation_macro_f1_mean"] for r in pruned_rows]
    pruned_f1_sd = [r["validation_macro_f1_sd"] for r in pruned_rows]

    raw_acc = [r["validation_accuracy_mean"] for r in raw_rows]
    pruned_acc = [r["validation_accuracy_mean"] for r in pruned_rows]

    conf_summary = conflicts.groupby("n_train").agg(
        ceiling_mean=("accuracy_ceiling", "mean"),
        train_mean=("train_accuracy", "mean"),
        mixed_mean=("mixed_vectors", "mean"),
    ).loc[sizes]

    x = np.arange(len(sizes))

    # Panel (a)
    fig, ax1 = plt.subplots(figsize=(3.4, 2.45))
    ax1.errorbar(
        x, raw_f1, yerr=raw_f1_sd, marker="o", linewidth=1.6, capsize=3,
        color="#1f77b4", label="Unpruned Macro-F1 ($\\bar{f}$)"
    )
    ax1.errorbar(
        x, pruned_f1, yerr=pruned_f1_sd, marker="s", linewidth=1.6, capsize=3,
        color="#ff7f0e", label="Pruned Macro-F1 ($\\bar{f}$)"
    )
    ax1.plot(
        x, raw_acc, marker="^", linestyle="--", linewidth=1.4,
        color="#1f77b4", alpha=0.8, label="Unpruned Accuracy ($a$)"
    )
    ax1.plot(
        x, pruned_acc, marker="v", linestyle="--", linewidth=1.4,
        color="#ff7f0e", alpha=0.8, label="Pruned Accuracy ($a$)"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["2,000", "5,000", "10,000", "20,000", "50,000"])
    ax1.set_xlabel("Training Sample Size ($n$)")
    ax1.set_ylabel("Validation Score")
    ax1.set_ylim(0.40, 0.86)
    ax1.legend(loc="center left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_learning_a.pdf", bbox_inches="tight")
    plt.close(fig)

    # Panel (b)
    fig, ax2 = plt.subplots(figsize=(3.4, 2.45))
    ax2_twin = ax2.twinx()
    ax2_twin.grid(False)
    l1 = ax2.plot(
        x, conf_summary["ceiling_mean"] * 100, marker="o", linewidth=2.0,
        color="#2ca02c", label="Conflict Ceiling (%)"
    )
    l2 = ax2.plot(
        x, conf_summary["train_mean"] * 100, marker="x", linestyle=":",
        linewidth=1.8, markersize=7, color="#111111",
        label="Unpruned Training Accuracy (%)"
    )
    bars = ax2_twin.bar(
        x, conf_summary["mixed_mean"], width=0.38, color="#d62728", alpha=0.28,
        label="Conflicting Vectors"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(["2,000", "5,000", "10,000", "20,000", "50,000"])
    ax2.set_xlabel("Training Sample Size ($n$)")
    ax2.set_ylabel("Training Accuracy (%)")
    ax2_twin.set_ylabel("Conflicting Feature Vectors", color="#b01f20")
    ax2.set_ylim(91.5, 100.2)
    ax2_twin.set_ylim(0, 1800)

    lines = l1 + l2 + [bars]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_learning_b.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_depth_curves():
    """Figure 2 panels: Tree depth sweep at 10,000 rows (a) Natural, (b) Balanced."""
    df = pd.read_csv(ROOT / "experiments/overnight-learning-curves/results/summary.csv")
    tree_10k = df[(df["kind"] == "tree") & (df["n_train"] == 10000) & (df["variant"] == "raw")].copy()
    tree_10k["depth_num"] = tree_10k["max_depth"].fillna(30.0).astype(float)

    depths = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0]
    depth_labels = ["0", "1", "2", "4", "8", "12", "20", "Full"]
    x = np.arange(len(depths))

    # Panel (a): Natural policy
    nat = tree_10k[tree_10k["policy"] == "natural"].groupby("depth_num").agg(
        train_err=("train_error", "mean"),
        val_err=("validation_error", "mean"),
        val_f1=("validation_macro_f1", "mean"),
    ).loc[depths]

    fig, ax1 = plt.subplots(figsize=(3.4, 2.45))
    ax1.plot(x, nat["train_err"], marker="o", linewidth=1.6, color="#1f77b4", label="Train Error (Source)")
    ax1.plot(x, nat["val_err"], marker="s", linewidth=1.6, color="#d62728", label="Validation Error")
    ax1.plot(x, nat["val_f1"], marker="^", linestyle="--", linewidth=1.6, color="#2ca02c", label="Validation Macro-F1 ($\\bar{f}$)")
    ax1.axvline(4, color="#888888", linestyle=":", linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(depth_labels)
    ax1.set_xlabel("Maximum Tree Depth ($d$)")
    ax1.set_ylabel("Error Rate / Macro-F1")
    ax1.set_ylim(0.0, 0.70)
    ax1.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_depth_a.pdf", bbox_inches="tight")
    plt.close(fig)

    # Panel (b): Balanced policy
    bal = tree_10k[tree_10k["policy"] == "balanced"].groupby("depth_num").agg(
        fit_err=("fit_error", "mean"),
        train_err=("train_error", "mean"),
        val_err=("validation_error", "mean"),
        val_f1=("validation_macro_f1", "mean"),
    ).loc[depths]

    fig, ax2 = plt.subplots(figsize=(3.4, 2.45))
    ax2.plot(x, bal["fit_err"], marker="o", linewidth=1.6, color="#1f77b4", label="Fit-Population Error")
    ax2.plot(x, bal["val_err"], marker="s", linewidth=1.6, color="#d62728", label="Validation Error")
    ax2.plot(x, bal["val_f1"], marker="^", linestyle="--", linewidth=1.6, color="#2ca02c", label="Validation Macro-F1 ($\\bar{f}$)")
    ax2.axhline(0.5021, color="#9467bd", linestyle="-.", linewidth=1.2, label="Pruned Full Tree ($\\bar{f}=0.502$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(depth_labels)
    ax2.set_xlabel("Maximum Tree Depth ($d$)")
    ax2.set_ylabel("Error Rate / Macro-F1")
    ax2.set_ylim(0.0, 0.95)
    ax2.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_depth_b.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_10k_comparison():
    """Figure 3 panels: Matched 10,000-row comparison (a) Macro-F1, (b) Accuracy tradeoff."""
    agg = pd.read_csv(
        ROOT / "experiments/overnight-learning-curves/results/analysis/method_comparison.csv"
    ).set_index("configuration")

    methods = [
        ("Unpruned\nTree", "Raw tree 10k", "Balanced raw tree 10k"),
        ("Pruned\nTree", "Pruned tree 10k", "Balanced pruned tree 10k"),
        ("Bagging\n($t=25$)", "Natural bagging 25", "Balanced bagging 25"),
        ("Boosting\n($\\eta=0.5$)", "Natural boosting 25 lr=0.5", "Balanced boosting 25 lr=0.5"),
        ("Boosting\n($\\eta=1.0$)", "Natural boosting 25 lr=1.0", "Balanced boosting 25 lr=1.0"),
    ]

    x = np.arange(len(methods))
    w = 0.35

    nat_f1 = [agg.loc[m[1], "validation_macro_f1_mean"] for m in methods]
    nat_f1_sd = [agg.loc[m[1], "validation_macro_f1_sd"] for m in methods]
    bal_f1 = [agg.loc[m[2], "validation_macro_f1_mean"] for m in methods]
    bal_f1_sd = [agg.loc[m[2], "validation_macro_f1_sd"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(3.4, 2.45))
    ax1.bar(x - w/2, nat_f1, width=w, yerr=nat_f1_sd, capsize=3, color="#4c78a8", label="Natural ($\\lambda=0$)")
    ax1.bar(x + w/2, bal_f1, width=w, yerr=bal_f1_sd, capsize=3, color="#f58518", label="Equal-Quota ($\\lambda=1$)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in methods])
    ax1.set_ylabel("Validation Macro-F1 ($\\bar{f}$)")
    ax1.set_ylim(0.35, 0.54)
    ax1.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_10k_a.pdf", bbox_inches="tight")
    plt.close(fig)

    nat_acc = [agg.loc[m[1], "validation_accuracy_mean"] * 100 for m in methods]
    bal_acc = [agg.loc[m[2], "validation_accuracy_mean"] * 100 for m in methods]
    nat_bacc = [agg.loc[m[1], "validation_balanced_accuracy_mean"] * 100 for m in methods]
    bal_bacc = [agg.loc[m[2], "validation_balanced_accuracy_mean"] * 100 for m in methods]

    fig, ax2 = plt.subplots(figsize=(3.4, 2.45))
    ax2.plot(x, nat_acc, marker="o", linewidth=1.6, color="#4c78a8", label="Natural Accuracy ($a$)")
    ax2.plot(x, bal_acc, marker="o", linestyle="--", linewidth=1.6, color="#f58518", label="Equal-Quota Accuracy ($a$)")
    ax2.plot(x, nat_bacc, marker="s", linewidth=1.6, color="#1b9e77", label="Natural Balanced ($\\bar{s}$)")
    ax2.plot(x, bal_bacc, marker="s", linestyle="--", linewidth=1.6, color="#d95f02", label="Equal-Quota Balanced ($\\bar{s}$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[0] for m in methods])
    ax2.set_ylabel("Validation Percentage (%)")
    ax2.set_ylim(38, 84)
    ax2.legend(loc="center left", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_10k_b.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_trajectories():
    """Figure 4 panels: Validation trajectories across ensemble size t."""
    dev = pd.read_csv(ROOT / "experiments/final-tree-comparison/results/final/development_summary.csv")
    fol = pd.read_csv(ROOT / "experiments/final-tree-comparison/results/returned-followups/followups/summary.csv")

    bag_val = fol[(fol["method"] == "bagging") & (fol["split"] == "validation")].groupby("members").agg(
        f1_mean=("macro_f1", "mean"),
        f1_sd=("macro_f1", "std"),
        acc_mean=("accuracy", "mean"),
        bal_mean=("balanced_accuracy", "mean"),
    )
    boost_val = dev[(dev["method"] == "boosting") & (dev["split"] == "validation")].groupby("members").agg(
        f1_mean=("macro_f1", "mean"),
        f1_sd=("macro_f1", "std"),
        acc_mean=("accuracy", "mean"),
        bal_mean=("balanced_accuracy", "mean"),
    )

    pruned_val = dev[(dev["method"] == "pruned") & (dev["split"] == "validation")]
    unpruned_val = dev[(dev["method"] == "unpruned") & (dev["split"] == "validation")]

    bag_m = bag_val.index.values
    boost_m = boost_val.index.values

    # Panel (a)
    fig, ax1 = plt.subplots(figsize=(3.4, 2.45))
    ax1.errorbar(
        bag_m[:3], bag_val["f1_mean"].values[:3], yerr=bag_val["f1_sd"].values[:3],
        marker="o", linewidth=1.8, capsize=3, color="#1f77b4", label="Bagging (Primary: 5 to 25)"
    )
    ax1.errorbar(
        bag_m[2:], bag_val["f1_mean"].values[2:], yerr=bag_val["f1_sd"].values[2:],
        marker="o", linestyle="--", linewidth=1.6, capsize=3, color="#1f77b4",
        label="Bagging (Extended: 35 to 50)"
    )
    ax1.errorbar(
        boost_m, boost_val["f1_mean"].values, yerr=boost_val["f1_sd"].values,
        marker="s", linewidth=1.8, capsize=3, color="#d62728",
        label="Boosting ($d=8, \\eta=1.0$)"
    )
    ax1.axhline(pruned_val["macro_f1"].mean(), color="#2ca02c", linestyle="-.", linewidth=1.5,
                label="Pruned Tree ($0.6087$)")
    ax1.axhline(unpruned_val["macro_f1"].mean(), color="#7f7f7f", linestyle=":", linewidth=1.5,
                label="Unpruned Tree ($0.5678$)")
    ax1.set_xticks([5, 10, 25, 35, 50])
    ax1.set_xlabel("Number of Ensemble Members ($t$)")
    ax1.set_ylabel("Validation Macro-F1 ($\\bar{f}$)")
    ax1.set_ylim(0.54, 0.62)
    ax1.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_traj_a.pdf", bbox_inches="tight")
    plt.close(fig)

    # Panel (b)
    fig, ax2 = plt.subplots(figsize=(3.4, 2.45))
    ax2.plot(bag_m, bag_val["acc_mean"].values * 100, marker="o", linewidth=1.6, color="#1f77b4",
             label="Bagging Accuracy ($a$)")
    ax2.plot(boost_m, boost_val["acc_mean"].values * 100, marker="s", linewidth=1.6, color="#d62728",
             label="Boosting Accuracy ($a$)")
    ax2.axhline(pruned_val["accuracy"].mean() * 100, color="#2ca02c", linestyle="-.", linewidth=1.4,
                label="Pruned Tree ($a=80.69\\%$)")
    ax2.axhline(unpruned_val["accuracy"].mean() * 100, color="#7f7f7f", linestyle=":", linewidth=1.4,
                label="Unpruned Tree ($a=77.39\\%$)")

    ax2.plot(bag_m, bag_val["bal_mean"].values * 100, marker="^", linestyle="--", linewidth=1.4,
             color="#1f77b4", alpha=0.8, label="Bagging Balanced ($\\bar{s}$)")
    ax2.plot(boost_m, boost_val["bal_mean"].values * 100, marker="v", linestyle="--", linewidth=1.4,
             color="#d62728", alpha=0.8, label="Boosting Balanced ($\\bar{s}$)")
    ax2.set_xticks([5, 10, 25, 35, 50])
    ax2.set_xlabel("Number of Ensemble Members ($t$)")
    ax2.set_ylabel("Validation Percentage (%)")
    ax2.set_ylim(56, 83)
    ax2.legend(loc="center right", frameon=True, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_traj_b.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_generalization_per_class():
    """Figure 5 panels: Per-class F1, Precision, and Recall on Generalization set."""
    gen = pd.read_csv(
        ROOT / "experiments/final-tree-comparison/results/final/generalization_per_class.csv"
    )
    agg = gen.groupby(["method", "class"]).agg(
        f1=("f1", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
    ).reset_index()

    methods = [
        ("unpruned", "Unpruned Tree", "#7f7f7f"),
        ("pruned", "Pruned Tree", "#2ca02c"),
        ("bagging", "Bagging ($t=25$)", "#1f77b4"),
        ("boosting", "Boosting ($t=25$)", "#d62728"),
    ]

    x = np.arange(10)
    w = 0.20
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    # Panel (a)
    fig, ax1 = plt.subplots(figsize=(6.8, 2.15))
    for (m_key, m_label, color), offset in zip(methods, offsets):
        sub = agg[agg["method"] == m_key].sort_values("class")
        ax1.bar(x + offset, sub["f1"].values, width=w, color=color, label=m_label)

    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASS_NAMES)
    ax1.set_xlabel("Traffic Category")
    ax1.set_ylabel("F1-Score ($f_j$)")
    ax1.set_ylim(0, 1.18)
    ax1.legend(loc="upper center", ncol=4, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_perclass_a.pdf", bbox_inches="tight")
    plt.close(fig)

    # Panel (b)
    fig, ax2 = plt.subplots(figsize=(6.8, 2.25))
    for (m_key, m_label, color), offset in zip(methods, offsets):
        sub = agg[agg["method"] == m_key].sort_values("class")
        ax2.bar(x + offset, sub["recall"].values, width=w, color=color, alpha=0.85, label=f"{m_label} Recall")
        ax2.scatter(
            x + offset, sub["precision"].values, marker="D", s=16,
            color="#111111", zorder=5, label="Precision ($p_j$)" if m_key == "unpruned" else None
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_NAMES)
    ax2.set_xlabel("Traffic Category")
    ax2.set_ylabel("Recall ($s_j$) / Precision ($p_j$)")
    ax2.set_ylim(0, 1.24)
    ax2.legend(loc="upper center", ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_perclass_b.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices():
    """Figure 6 panels: Row-normalised recall (%) confusion matrices."""
    eval_dir = ROOT / "experiments/final-tree-comparison/results/final/evaluation"

    def load_mean_cm(method_prefix):
        cms = []
        for s in (42, 43):
            suffix = "n1" if method_prefix in ("pruned", "unpruned") else "n25"
            path = eval_dir / f"generalization-{method_prefix}-s{s}-{suffix}.json"
            with open(path) as f:
                cms.append(np.array(json.load(f)["confusion_matrix"], dtype=float))
        mean_cm = np.mean(cms, axis=0)
        row_sums = mean_cm.sum(axis=1, keepdims=True)
        return (mean_cm / row_sums) * 100.0

    for suffix_name, method_prefix in [("a", "pruned"), ("b", "bagging")]:
        cm = load_mean_cm(method_prefix)
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8.0)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8.0)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                val = cm[i, j]
                if val >= 1.0:
                    text_color = "white" if val > 50 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            color=text_color, fontsize=7.0)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig_cm_{suffix_name}.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    plot_learning_curves()
    plot_depth_curves()
    plot_10k_comparison()
    plot_ensemble_trajectories()
    plot_generalization_per_class()
    plot_confusion_matrices()
    print("Generated all 12 subfigure panels in", FIG_DIR)
