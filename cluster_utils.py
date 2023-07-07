import os
import torch
import numpy as np
import hdbscan 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
import adjustText
from hdbscan import flat
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from openTSNE import TSNE as oTSNE
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure as hcv_measure,
)


# Metric parameters
DIM_RED_ALGO = 'otsne'  # 'pca', 'tsne', 'ftsne', 'otsne'
RED_DIM = None  # None for no dimensionality reduction
FIG_SIZE = (15, 5)
TEXT_SIZE = 16
N_ANNOTATED_SAMPLES = 8
COLORS = (list(plt.cm.tab20(np.arange(20)[0::2])) + \
          list(plt.cm.tab20(np.arange(20)[1::2]))) * 10
BLACK = (0.0, 0.0, 0.0)
SCATTER_PARAMS = {'s': 50, 'linewidth': 0, 'alpha': 0.5}
LEAF_SEPARATION = 0.3
MIN_CLUSTER_SIZE = 25
N_CLUSTERS = 10  # not used
NA_VALUE = (0.0, 0.0, 0.0)  # unassigned data sample in hdscan
NA_COLOR = np.array([0.0, 0.0, 0.0, 1.0])  # black (never a cluster color)
N_CPUS = min(10, os.cpu_count() // 2)


def plot_clusters(embeddings: torch.Tensor,
                  raw_txt: list[str],
                  labels: list[str],
                  selected_phases: list[str],
                  selected_conditions: list[str],
                  ) -> plt.Figure:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    # Dimensionality reduction of all criterias' embeddings
    print('\nReducing dim of %s eligibility criteria embeddings' % len(labels))
    fig, axs = plt.subplots(1, 3, figsize=FIG_SIZE)
    cluster_data = compute_reduced_repr(embeddings, dim=RED_DIM, algorithm=DIM_RED_ALGO)
    if RED_DIM == 2:
        plot_data = cluster_data
    else:
        plot_data = compute_reduced_repr(embeddings, dim=2, algorithm=DIM_RED_ALGO)
    
    # Eligibility criteria selection (specific condition, specific phase)
    phases = [l['phases'] for l in labels]
    conditions = [l['conditions'] for l in labels]
    statuses = [l['status'] for l in labels]
    selected_indices = filter_data(
        phases, conditions, selected_phases, selected_conditions)
    token_info = {
        'plot_data': plot_data[selected_indices],
        'raw_txt': [t for t, i in zip(raw_txt, selected_indices) if i],
        'class_lbls': [s for s, i in zip(statuses, selected_indices) if i],
    }
    
    # Cluster selected criteria, based on reduced embeddings
    cluster_info = get_cluster_info(cluster_data[selected_indices])
    print('Clustering %s eligibility criteria\n' % sum(selected_indices))
    plot_hierarchy(token_info, cluster_info, axs)
    return fig


def filter_data(phases, conds, selected_phases, selected_conds):
    """ Filter out CTs that do not belong to a given phase and condition
    """
    phase_indices = [any([p in pp for p in selected_phases]) for pp in phases]
    cond_indices = [any([c in cc for c in selected_conds]) for cc in conds]
    combined = [p and c for p, c in zip(phase_indices, cond_indices)]
    return combined


def get_cluster_info(cluster_data):
    """ Cluster data points with hdbscan algorithm and return cluster information
    """
    # Find cluster affordances based on cluster hierarchy
    min_cluster_size = min(len(cluster_data) // 2, MIN_CLUSTER_SIZE)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(cluster_data)
    # if N_CLUSTERS is not None:
    #     try:
    #         clusterer = flat.HDBSCAN_flat(cluster_data,
    #                                       n_clusters=N_CLUSTERS,
    #                                       clusterer=clusterer)
    #     except IndexError:
    #         raise RuntimeError('Clustering algorithm did not converge.')
    
    # Return everything needed
    return {'clusterer': clusterer,
            'cluster_lbls': clusterer.labels_,
            'n_clusters': N_CLUSTERS}


def plot_hierarchy(token_info, cluster_info, axs):
    """ ...
    """
    # Find and measure best match between empirical and theoretical clusters
    cluster_colors, class_colors = align_cluster_colors(token_info, cluster_info)
    cluster_labels, class_labels = compute_labels(cluster_colors, class_colors)
    ari_match = adjusted_rand_score(class_labels, cluster_labels)
    nmi_match = normalized_mutual_info_score(class_labels, cluster_labels)
    homog, compl, v_match = hcv_measure(class_labels, cluster_labels)
    print(' - ari: %.3f\n - nmi: %.3f\n - hom: %.3f\n - comp: %.3f\n - v: %.3f'%
          (ari_match, nmi_match, homog, compl, v_match))
    
    # Visualize theoretical clusters (using, e.g., ICD10-CM code hierarchy)
    plot_data = token_info['plot_data']
    axs[0].scatter(*plot_data.T, c=class_colors, **SCATTER_PARAMS)
    axs[0].set_xticklabels([]); axs[0].set_yticklabels([])
    axs[0].set_xticks([]); axs[0].set_yticks([])
    
    # Add text annotation around some data points
    loop = list(zip(plot_data, token_info['raw_txt'], token_info['class_lbls']))
    np.random.shuffle(loop)
    texts = []
    for d, t, l in loop:
        if 1:  # l == 'terminated':
            wrapped = textwrap.fill(t[:200], 40)
            texts.append(axs[0].text(d[0], d[1], wrapped, fontsize='xx-small'))
        if len(texts) >= N_ANNOTATED_SAMPLES: break
    
    # Adjust text position to avoid overlap (not working bery well)
    arrowprops = dict(arrowstyle='->', color='k', lw=0.5)
    adjustText.adjust_text(
        texts, ax=axs[0],
        x=token_info['plot_data'][:, 0], y=token_info['plot_data'][:, 1],
        arrowprops=arrowprops, force_text=(0.2, 0.4), force_points=(0.0, 0.1) 
    )
    
    # Visualize empirical clusters
    axs[1].scatter(*plot_data.T, c=cluster_colors, **SCATTER_PARAMS)
    axs[1].set_xticklabels([]); axs[1].set_yticklabels([])
    axs[1].set_xticks([]); axs[1].set_yticks([])
    
    # Visualize empirical cluster tree
    cluster_info['clusterer'].condensed_tree_.plot(
        axis=axs[2], leaf_separation=LEAF_SEPARATION, colorbar=True)
    # add_cluster_info(axs[2],
    #                  cluster_info['clusterer'],
    #                  cluster_info['n_clusters'],
    #                  cluster_colors)
    # axs[2].invert_yaxis()
    
    # Add things around the figure
    axs[0].set_title('Data and labels', fontsize=TEXT_SIZE)
    axs[1].set_title('Empirical clusters', fontsize=TEXT_SIZE)
    axs[2].set_title('Empirical cluster tree', fontsize=TEXT_SIZE)
    axs[2].get_yaxis().set_tick_params(left=False)
    axs[2].get_yaxis().set_tick_params(right=False)
    axs[2].set_yticks([])
    axs[2].spines['top'].set_visible(True)
    axs[2].spines['right'].set_visible(True)
    axs[2].spines['bottom'].set_visible(True)
    axs[2].set_ylabel('', fontsize=TEXT_SIZE)


def add_cluster_info(ax: plt.Axes,
                     clusterer: hdbscan.HDBSCAN,
                     n_clusters: int,
                     cluster_colors: list[np.ndarray]):
    """ ...
    """
    # Select the branches that will be highlighted (empirical clusters)
    branch_df = clusterer.condensed_tree_.to_pandas().sort_values('lambda_val')
    branch_df = branch_df[branch_df['child_size'] > 1]
    selected_branches, selected_branch_sizes = [], []
    for _, branch in branch_df.iterrows():
        if len(selected_branches) == n_clusters: break
        selected_branches.append(branch['child'])
        selected_branch_sizes.append(branch['child_size'])
        if branch['parent'] in selected_branches:
            selected_branches.remove(branch['parent'])
            selected_branch_sizes.remove(branch['child_size'])
    
    # Get all cluster bounds and draw a white rectangle to add info on it
    cluster_bounds = clusterer.condensed_tree_.get_plot_data()['cluster_bounds']
    cluster_bottom = min([cluster_bounds[b][-1] for b in selected_branches])
    left, right = ax.get_xlim()
    _, bottom = ax.get_ylim()  # top would be the end of the tree
    rectangle_width = (right - left) * 0.998
    bottom_point = (left + 0.001 * rectangle_width, cluster_bottom - 0.001)
    rectangle_height = cluster_bottom / 15
    rectangle_specs = [bottom_point, rectangle_width, rectangle_height]
    mask = patches.Rectangle(*rectangle_specs, facecolor='white', zorder=10)
    ax.add_patch(mask)
    # top_lim = bottom_point[1] + rectangle_height * 1.025
    # bottom_lim = bottom - 0.11 * (top_lim - bottom)
    # ax.set_ylim([top_lim, bottom_lim])
    
    # # Retrieve colours and match them to the correct clusters
    # assigned_colors = [c for c in cluster_colors if c != (0.0, 0.0, 0.0)]
    # unique_colors = np.unique(assigned_colors, axis=0)
    # sizes_from_colors = [len([ac for ac in assigned_colors if (ac == uc).all()])
    #                      for uc in unique_colors]
    # sizes_from_tree = [int(cluster_bounds[b][1] - cluster_bounds[b][0])
    #                    for b in selected_branches]
    # match_indices = [sizes_from_colors.index(v) for v in sizes_from_tree]
    # unique_colors = unique_colors[match_indices]
    
    # # Plot small circles with the right color, below each branch of the tree
    # for b, c in zip(selected_branches, unique_colors):
    #     anchor_point = (
    #         (cluster_bounds[b][0] + cluster_bounds[b][1]) / 2 * LEAF_SEPARATION,
    #         bottom_point[1] + rectangle_height / 2
    #     )
    #     xlim, ylim = ax.get_xlim(), ax.get_ylim()
    #     zorder = int(1e6 + anchor_point[0])
    #     ax.plot(*anchor_point, 'o', color=c, markersize=10, zorder=zorder)
    #     ax.set_xlim(xlim), ax.set_ylim(ylim)


def align_cluster_colors(token_info, cluster_info):
    """ ...
    """
    # Identify possible colour values
    cluster_lbls = cluster_info['cluster_lbls']
    class_lbls = token_info['class_lbls']
    unique_classes = sorted(list(set(class_lbls)))
    unique_clusters = sorted(list(set(cluster_lbls)))

    # In this case, true labels define the maximum number of colours
    if len(unique_classes) >= len(unique_clusters):
        color_map = best_color_match(cluster_lbls, class_lbls, unique_clusters, unique_classes)
        color_map = {k: unique_classes.index(v) for k, v in color_map.items()}
        cluster_colors = [COLORS[color_map[i]] if i >= 0 else BLACK for i in cluster_lbls]
        class_colors = [COLORS[unique_classes.index(l)] for l in class_lbls]
    
    # In this case, empirical clusters define the maximum number of colours
    else:
        color_map = best_color_match(class_lbls, cluster_lbls, unique_classes, unique_clusters)
        color_map = {unique_classes.index(k): v for k, v in color_map.items()}
        cluster_colors = [COLORS[i] if i >= 0 else BLACK for i in cluster_lbls]
        class_colors = [COLORS[color_map[unique_classes.index(l)]] for l in class_lbls]
    
    # Correct format of unassigned data samples
    cluster_colors = [c if c != NA_VALUE else NA_COLOR for c in cluster_colors]
    
    # Return aligned empirical and theorical clusters
    return cluster_colors, class_colors


def compute_labels(cluster_colors, class_colors):
    """ ...
    """
    # Check for existence of unassigned samples in the empirical clusters
    cluster_colors, class_colors = cluster_colors.copy(), class_colors.copy()
    assigned_sample_ids = np.where(np.any(cluster_colors != NA_COLOR, axis=1))
    unassigned_samples_exist = len(assigned_sample_ids) == len(cluster_colors)

    # Compute cluster and class labels
    if unassigned_samples_exist: class_colors.append(NA_COLOR)
    _, class_labels = np.unique(class_colors, axis=0, return_inverse=True)
    _, cluster_labels = np.unique(cluster_colors, axis=0, return_inverse=True)
    if unassigned_samples_exist: class_labels = class_labels[:-1]

    # Return cluster and class labels
    return cluster_labels, class_labels


def compute_reduced_repr(embeddings: np.ndarray,
                         dim='2',
                         algorithm='pca'
                         ) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """    
    if algorithm == 'pca':
        return PCA().fit_transform(embeddings)[:, :dim]
    
    elif algorithm == 'tsne':
        params = {
            'perplexity': 30.0,
            'learning_rate': 'auto',  # or any value in [10 -> 1000] may be good
            'n_iter': 1000,
            'n_iter_without_progress': 200,
            'metric': 'cosine',
            'init': 'pca',
            'n_jobs': N_CPUS,
            'verbose': 2,
        }
        return TSNE(dim, **params).fit_transform(embeddings)
        
    elif algorithm == 'otsne':
        params = {
            'n_components': dim,
            'n_iter': 1000,
            'n_jobs': N_CPUS,
            'verbose': True,
        }
        return oTSNE(**params).fit(embeddings)
    
    else:
        print('Invalid or no reduction algorithm given. Raw values returned.')
        return embeddings


def best_color_match(src_lbls, tgt_lbls, unique_src_lbls, unique_tgt_lbls):
    """ Find the best match between subcategories, based on cluster memberships
    """
    cost_matrix = np.zeros((len(unique_src_lbls), len(unique_tgt_lbls)))
    for i, src_lbl in enumerate(unique_src_lbls):
        for j, tgt_lbl in enumerate(unique_tgt_lbls):
            count = sum(s == src_lbl and t == tgt_lbl
                        for s, t in zip(src_lbls, tgt_lbls))
            cost_matrix[i, j] = -count
    
    rows, cols = linear_sum_assignment(cost_matrix)
    return {unique_src_lbls[i]: unique_tgt_lbls[j] for i, j in zip(rows, cols)}
