import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


DATA_DIR = "experiments/experiment_3_results"
FILTER_LEVEL = {"cond": 3, "itrv": 2}
DATA_FILE_NAME = lambda c, i: "model-pubmed-bert-sentence_cond-%1i_itrv-%1i.csv" % (c, i)
METRICS = ["ROUGE-Average-F Score", "SciBERT-F Score"]
SUPPL_METRICS = [
    "ROUGE-1-F Score", "ROUGE-2-F Score", "ROUGE-L-F Score",
    "BERT-F Score", "SciBERT-F Score", "Longformer-F Score",
]
METHOD_COLORS = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
METHODS = ["Cluster", "Shuffled Cluster", "LLM", "Shuffled LLM"]


def main():
    plot_one_metric_set(
        metric_set=METRICS,
        output_path="experiments/figure_experiment_3.png",
    )
    plot_one_metric_set(
        metric_set=SUPPL_METRICS,
        output_path="experiments/figure_experiment_3_suppl.png",
    )


def plot_one_metric_set(metric_set, output_path):
    """ Generate a comparison plot between cluster and LLM EC generation methods,
        for a given set of metrics
    """
    # Plot each condition's results
    assert len(metric_set) <= 6, "No more than 6 metrics plotted in one figure"
    n_plot_cols = 3 if len(metric_set) == 6 else len(metric_set)
    n_plot_rows = 2 if len(metric_set) == 6 else 1
    _, axs = plt.subplots(
        n_plot_rows, n_plot_cols,
        figsize=(6 * n_plot_cols, 6 * n_plot_rows)
    )
    if len(metric_set) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for i, metric in enumerate(metric_set):
        plot_legend = (i // n_plot_cols == n_plot_rows - 1 and i % n_plot_cols == n_plot_cols // 2)
        plotted_keys = ["%s %s" % (method, metric) for method in METHODS]
        plot_one_metric(plotted_keys, plot_legend, axs[i], len(axs))
        
    # Save adjusted plot
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(output_path, dpi=300)
    
    
def plot_one_metric(
    plotted_keys: list[str],
    plot_legend: bool,
    ax: "plt.Axis",
    num_axes: int,
) -> float:
    """ Plot results of experiment 3 for one set of condition and intervention levels
    
    Args:
        level_set (dict[str, int]): condition and intervention filter levels
    """
    # Load the data
    csv_file_path = DATA_FILE_NAME(FILTER_LEVEL["cond"], FILTER_LEVEL["itrv"])
    df_raw = pd.read_csv(os.path.join(DATA_DIR, csv_file_path))
    
    # Filter samples where the clustering algorithm converged
    df = df_raw[df_raw["Cluster Quality"] != -1.0]
    mean_converging = df[plotted_keys].mean()
    stderr_converging = df[plotted_keys].std() / np.sqrt(len(df))
    
    # Print exact values
    print("\nProportion converging:")
    print(len(df) / len(df_raw) * 100, "%")
    print("\nMeans:")
    print(mean_converging)
    print("\nStderr:")
    print(stderr_converging, end="\n\n")
    
    # Print some useful values
    plotted_metric = plotted_keys[0].split("Cluster ")[-1]
    cluster_over_llm = mean_converging["Cluster " + plotted_metric] / mean_converging["LLM " + plotted_metric] * 100
    cluster_diff_random = mean_converging["Cluster " + plotted_metric] - mean_converging["Shuffled Cluster " + plotted_metric]
    llm_diff_random = mean_converging["LLM " + plotted_metric] - mean_converging["Shuffled LLM " + plotted_metric]
    cluster_diff_over_llm_diff = cluster_diff_random / llm_diff_random * 100
    print("Cluster makes up %4f%% of LLM performance" % cluster_over_llm)
    print("Cluster vs random makes up %4f%% of LLM vs random" % cluster_diff_over_llm_diff, end="\n\n")
    
    # Initialize bar plot utilities
    max_value = 0.0
    bar_width = 0.15  # should be greater than 0.0 and smaller than 0.25
    def pos_fn(idx):
        group_shift = (idx // 2 - 0.5) * 2 * (1 + 2 * bar_width) / 6
        bar_shift = (idx % 2 - 0.5) * bar_width
        return 0.5 + group_shift + bar_shift
    
    # Statistical tests utilities
    num_comparisons = (num_axes * 3)  # For each metric: Cluster vs. LLM, Cluster vs. Random, LLM vs. Random
    significant = lambda p: "*" if p < 0.05 else "n.s."
    significant_bf = lambda p: "*" if p < 0.05 / num_comparisons else "n.s."
    
    # Plot the bars
    ax_handles = []
    ax_labels = []
    for i, column in enumerate(plotted_keys):
        color = METHOD_COLORS[i % len(METHODS)]
        label = METHODS[i % len(METHODS)] if i < len(METHODS) else None
        handle = ax.bar(
            pos_fn(i), mean_converging[column], bar_width,
            yerr=stderr_converging[column], alpha=0.75, label=label, color=color,
            capsize=4, error_kw={"elinewidth": 2, "capthick":2},
        )
        if label not in ax_labels:
            ax_handles.append(handle)
            ax_labels.append(label)
        
        # Print statistical test results
        if "Shuffled" not in column:
            _, p_value_vs_rand = ttest_rel(
                df[plotted_keys][column],
                df[plotted_keys]["Shuffled %s" % column],
            )
            significance = significant(p_value_vs_rand)
            significance_bf = significant_bf(p_value_vs_rand)
            print(
                "P-value of %s vs random baseline (paired t-test, %i samples): %f (%s, BF-%s)"\
                % (column, len(df), p_value_vs_rand, significance, significance_bf)
                )
            if "Cluster" in column:
                _, p_value_vs_llm = ttest_rel(
                    df[plotted_keys][column],
                    df[plotted_keys][column.replace("Cluster", "LLM")],
                )
                significance = significant(p_value_vs_llm)
                significance_bf = significant_bf(p_value_vs_llm)
                print(
                    "P-value of %s vs LLM (paired t-test, %i samples): %f (%s, BF-%s)"\
                    % (column, len(df), p_value_vs_llm, significance, significance_bf)
                )
        
        # Add individual data points with jitter
        jitter = bar_width / 3 * (np.random.rand(len(df)) - 0.5)
        ax.scatter(
            np.full(len(df), pos_fn(i)) + jitter,
            df[column], color="k", alpha=0.3, s=2,
        )
        this_max = max(df[column].max(), df[column].max())
        if this_max > max_value: max_value = this_max
        
    # Customizing the plot
    print("\n", end="")
    ax.set_xlim([0, 1])
    ax.set_xticks([], [])
    y_label = plotted_keys[0]\
        .split("Cluster ")[1]\
        .replace("-", " ")\
        .replace(" F ", " F1-")
    if num_axes == 1:
        ax.set_ylabel(y_label, fontsize=16)
    else:
        ax.set_title(y_label, fontsize=16)
    if plot_legend:
        if num_axes == 1:
            ax.set_ylim([0.0, max_value * 1.2])
        else:
            bbox_x = 0.5 if num_axes % 2 == 1 or num_axes == 6 else -0.1
            bbox_y = -max_value / 50
        ax.legend(
            ax_handles,
            ax_labels,
            fontsize=16 if num_axes > 1 else 14,
            loc="upper center",
            bbox_to_anchor=(bbox_x, bbox_y) if num_axes > 1 else None,
            ncol=len(ax_handles) if num_axes > 1 else len(ax_handles) // 2,
        )


if __name__ == "__main__":
    main()
    