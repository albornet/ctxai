import os
import torch
import numpy as np
import hdbscan 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from openTSNE import TSNE as oTSNE
from scipy.spatial.distance import cdist


DIM_RED_ALGO = 'tsne'  # 'pca', 'tsne', 'otsne'
RED_DIM = None  # None for no dimensionality reduction when clustering
N_CPUS = min(10, os.cpu_count() // 2)  # for t-sne
FIG_SIZE = (15, 5)
TEXT_SIZE = 16
COLORS = np.array((
    list(plt.cm.tab20(np.arange(20)[0::2])) + \
    list(plt.cm.tab20(np.arange(20)[1::2]))) * 10
)
NA_COLOR = np.array([0.0, 0.0, 0.0, 0.1])
NOT_NA_COLOR_ALPHA = 0.8
COLORS[:, -1] = NOT_NA_COLOR_ALPHA
SCATTER_PARAMS = {'s': 50, 'linewidth': 0}
LEAF_SEPARATION = 0.3
MIN_CLUSTER_SIZE = 0.01  # in proportion of the number of data points
MAX_CLUSTER_SIZE = 0.25  # in proportion of the number of data points


def plot_clusters(embeddings: torch.Tensor,
                  raw_txt: list[str],
                  labels: list[str],
                  ) -> plt.Figure:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    # Reduce dimensionality of all criteria's embeddings
    print('\nReducing dim of %s eligibility criteria embeddings' % len(labels))
    fig, axs = plt.subplots(1, 3, figsize=FIG_SIZE)
    cluster_data = compute_reduced_repr(embeddings, dim=RED_DIM, algorithm=DIM_RED_ALGO)
    if RED_DIM == 2:
        plot_data = cluster_data
    else:
        # plot_data = compute_reduced_repr(embeddings, dim=2, algorithm=DIM_RED_ALGO)
        plot_data = compute_reduced_repr(cluster_data, dim=2, algorithm=DIM_RED_ALGO)
    
    # Load plot data, raw data and labels
    token_info = {
        'plot_data': plot_data,
        'raw_txt': raw_txt,
        'class_lbls': [l['status'] for l in labels],
        'ct_paths': [l['ct_path'] for l in labels],
    }
    
    # Cluster selected criteria, based on reduced embeddings
    print('Clustering %s eligibility criteria\n' % len(token_info['plot_data']))
    cluster_info = get_cluster_info(cluster_data)
    generate_cluster_statistics(token_info, cluster_info)
    plot_cluster_hierarchy(token_info, cluster_info, axs)
    return fig


def get_cluster_info(cluster_data):
    """ Cluster data points with hdbscan algorithm and return cluster information
    """
    # Find cluster affordances based on cluster hierarchy
    n_samples = len(cluster_data)
    max_cluster_samples = int(n_samples * MAX_CLUSTER_SIZE)
    min_cluster_samples = int(n_samples * MIN_CLUSTER_SIZE)
    min(len(cluster_data) // 2, min_cluster_samples)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_samples,
        max_cluster_size=max_cluster_samples,
    )
    clusterer.fit(cluster_data)

    # Get centroids and medoids
    n_clusters = clusterer.labels_.max() + 1  # -1 being ignored
    centroids = [clusterer.weighted_cluster_centroid(i) for i in range(n_clusters)]
    medoids = [clusterer.weighted_cluster_medoid(i) for i in range(n_clusters)]
    
    # Get index of closest data point
    centroid_dists = cdist(centroids, cluster_data)
    medoid_dists = cdist(medoids, cluster_data)
    centroid_closest_id = np.argmin(centroid_dists, axis=1)
    medoid_closest_id = np.argmin(medoid_dists, axis=1)
    
    # Return cluster info
    return {
        'clusterer': clusterer,
        'cluster_lbls': clusterer.labels_,
        'centroid_ids': centroid_closest_id,
        'medoid_ids': medoid_closest_id,
    }


def generate_cluster_statistics(token_info, cluster_info):
    """ Lalalalalalala
    """
    centrois = [token_info['raw_txt'][i] for i in cluster_info['centroid_ids']]
    medoids = [token_info['raw_txt'][i] for i in cluster_info['medoid_ids']]
    
    import ipdb; ipdb.set_trace()
    

def plot_cluster_hierarchy(token_info, cluster_info, axs):
    """ Plot theoretical clusters (using labels), empirical clusters (using
        hdbscan), and the empirical cluster tree
    """
    # Assign label colors and evaluate cluster purity (NOT WORKING YET)
    cluster_colors, class_colors = assign_cluster_colors(token_info, cluster_info)
    # cluster_labels, class_labels = compute_labels(cluster_colors, class_colors)
    # ari_match = adjusted_rand_score(class_labels, cluster_labels)
    # nmi_match = normalized_mutual_info_score(class_labels, cluster_labels)
    # homog, compl, v_match = hcv_measure(class_labels, cluster_labels)
    # print(' - ari: %.3f\n - nmi: %.3f\n - hom: %.3f\n - comp: %.3f\n - v: %.3f'%
    #       (ari_match, nmi_match, homog, compl, v_match))
    
    # Visualize theoretical clusters (using, e.g., ICD10-CM code hierarchy)
    plot_data = token_info['plot_data']
    axs[0].scatter(*plot_data.T, c=class_colors, **SCATTER_PARAMS)
    axs[0].set_xticklabels([]); axs[0].set_yticklabels([])
    axs[0].set_xticks([]); axs[0].set_yticks([])
    
    # Visualize empirical clusters
    axs[1].scatter(*plot_data.T, c=cluster_colors, **SCATTER_PARAMS)
    axs[1].set_xticklabels([]); axs[1].set_yticklabels([])
    axs[1].set_xticks([]); axs[1].set_yticks([])
    
    # Visualize empirical cluster tree
    cluster_info['clusterer'].condensed_tree_.plot(
        axis=axs[2], leaf_separation=LEAF_SEPARATION, colorbar=True)
    
    # Polish figure
    n_labels = len(set(token_info['class_lbls']))
    n_clusters = cluster_info['cluster_lbls'].max() + 1
    axs[0].set_title('Data and labels (N = %s)' % n_labels, fontsize=TEXT_SIZE)
    axs[1].set_title('Empirical clusters (M = %s)' % n_clusters, fontsize=TEXT_SIZE)
    axs[2].set_title('Empirical cluster tree', fontsize=TEXT_SIZE)
    axs[2].get_yaxis().set_tick_params(left=False)
    axs[2].get_yaxis().set_tick_params(right=False)
    axs[2].set_ylabel('', fontsize=TEXT_SIZE)
    axs[2].set_yticks([])
    axs[2].spines['top'].set_visible(True)
    axs[2].spines['right'].set_visible(True)
    axs[2].spines['bottom'].set_visible(True)
    
    
def assign_cluster_colors(token_info, cluster_info):
    """ Used to be more complicated, but for now just assign fixed colours
    """
    # Generate class and cluster indices
    token_classes = sorted(set(token_info['class_lbls']))
    cluster_classes = sorted(set(cluster_info['cluster_lbls']))
    token_to_id = {v: i for i, v in enumerate(token_classes)}
    cluster_to_id = {v: i for i, v in enumerate(cluster_classes)}
    
    # Create class and cluster colors for all samples
    cluster_colors = [COLORS[cluster_to_id[c]] if c >= 0 else NA_COLOR
                      for c in cluster_info['cluster_lbls']]
    class_colors = [COLORS[token_to_id[c]] for c in token_info['class_lbls']]
    return cluster_colors, class_colors


def compute_reduced_repr(embeddings: np.ndarray,
                         dim='2',
                         algorithm='pca'
                         ) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """    
    if dim == None or algorithm == None:
        return embeddings
    
    if algorithm == 'pca':
        return PCA().fit_transform(embeddings)[:, :dim]
    
    elif algorithm == 'tsne':
        params = {
            'perplexity': 30.0,
            'learning_rate': 'auto',  # or any value in [10 -> 1000] may be good
            'n_iter': 10000,
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
            'n_iter': 10000,
            'n_jobs': N_CPUS,
            'verbose': True,
        }
        return oTSNE(**params).fit(embeddings)
    
    else:
        raise ValueError('Invalid reduction algorithm.')
    
    
# def filter_data(phases, conds, chosen_phases,
#                 chosen_conds, chose_cond_ids, chose_itrv_ids):
#     """ Filter out CTs that do not belong to a given phase and condition
#     """
#     if chosen_phases == None:
#         phase_ids = [1 for _ in range(len(phases))]
#     else:
#         phase_ids = [any([p in pp for p in chosen_phases]) for pp in phases]
#     if chosen_conds == None:
#         cond_ids = [1 for _ in range(len(conds))]
#     else:
#         cond_ids = [any([c in cc for c in chosen_conds]) for cc in conds]
#     combined = [p and c for p, c in zip(phase_ids, cond_ids)]
#     return combined
