import os
import time
import openai
import torch
import numpy as np
import hdbscan 
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from itertools import zip_longest
from tqdm import tqdm
from openai.error import RateLimitError
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist


CLUSTER_DIM_RED_ALGO = 'tsne'  # 'pca', 'tsne'
PLOT_DIM_RED_ALGO = 'tsne'  # 'pca', 'tsne'
CLUSTER_RED_DIM = 2  # None for no dimensionality reduction when clustering
PLOT_RED_DIM = 2  # either 2 or 3
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
MAX_CLUSTER_SIZE = 0.10  # in proportion of the number of data points


def plot_clusters(embeddings: torch.Tensor,
                  raw_txt: list[str],
                  labels: list[str],
                  ) -> plt.Figure:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    # Reduce dimensionality of all criteria's embeddings
    print('\nReducing dim of %s eligibility criteria embeddings' % len(labels))
    fig = plt.figure(figsize=FIG_SIZE)
    cluster_data = compute_reduced_repr(
        embeddings,
        dim=CLUSTER_RED_DIM,
        algorithm=CLUSTER_DIM_RED_ALGO
    )
    if CLUSTER_RED_DIM == PLOT_RED_DIM\
    and CLUSTER_DIM_RED_ALGO == PLOT_DIM_RED_ALGO:
        plot_data = cluster_data
    else:
        plot_data = compute_reduced_repr(
            cluster_data,  # embeddings,
            dim=PLOT_RED_DIM,
            algorithm=PLOT_DIM_RED_ALGO
        )
    
    # Load plot data, raw data and labels
    token_info = {
        'plot_data': plot_data,
        'raw_txt': raw_txt,
        'ct_lbls': [l['ct_status'] for l in labels],
        'unique_lbls': list(set([l['ct_status'] for l in labels])),
        'ct_paths': [l['ct_path'] for l in labels],
    }
    
    # Cluster selected criteria, based on reduced embeddings
    print('Clustering %s eligibility criteria' % len(token_info['plot_data']))
    cluster_info = get_cluster_info(cluster_data)
    text_report, stat_report = generate_cluster_reports(token_info, cluster_info)
    plot_cluster_hierarchy(token_info, cluster_info, fig, red_dim=PLOT_RED_DIM)
    return fig, text_report, stat_report


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
    
    # Sort data points by how close they are from each cluster medoid
    n_clusters = clusterer.labels_.max() + 1  # -1 being ignored
    medoids = [clusterer.weighted_cluster_medoid(i) for i in range(n_clusters)]   
    medoid_dists = cdist(medoids, cluster_data)
    medoid_sorting = np.argsort(medoid_dists, axis=1)
    
    # For each cluster group, take only samples that belong to the same cluster
    medoid_closest_ids = []
    for cluster_id in range(n_clusters):
        cluster = np.where(clusterer.labels_ == cluster_id)[0]
        sorted_cluster = [i for i in medoid_sorting[cluster_id] if i in cluster]
        medoid_closest_ids.append(sorted_cluster)
        
    # Return cluster info
    return {
        'clusterer': clusterer,
        'n_clusters': n_clusters,
        'cluster_lbls': clusterer.labels_,
        'medoid_ids': medoid_closest_ids,
    }


def generate_cluster_reports(token_info, cluster_info):
    """ Characterize clusters and generate statistics given clinical trials
    """
    # Compute useful statistics
    cluster_prevalences, label_prevalences =\
        compute_cluster_statistics(token_info, cluster_info)
    
    # Identify one "typical" criterion string for each cluster of criteria
    txts_grouped_by_cluster = [
        [token_info['raw_txt'][i] for i in sample_ids]
        for sample_ids in cluster_info['medoid_ids']
    ]  # each cluster group includes criteria sorted by distance to its medoid
    close_medoid_criterion_txts = [txts[:20] for txts in txts_grouped_by_cluster]
    cluster_typicals = pool_cluster_criteria(close_medoid_criterion_txts)
    
    # Generate a report from the computed prevalences and typical criteria
    text_report = generate_text_report(
        txts_grouped_by_cluster,
        cluster_prevalences
    )
    stat_report = generate_stat_report(
        cluster_typicals=cluster_typicals,
        cluster_prevalences=cluster_prevalences,
        label_prevalences=label_prevalences,
        unique_labels=token_info['unique_lbls'],
        n_cts=len(set(token_info['ct_paths'])),
        n_ecs=len(token_info['ct_paths']),
    )
    return text_report, stat_report


def generate_text_report(grouped_by_cluster: list[list[str]],
                         cluster_prevalences: list[float],
                         ) -> list[list[str]]:
    """ Generate a report to write all clusters' criteria in a csv file
    """
    zipped = list(zip(cluster_prevalences, grouped_by_cluster))
    zipped.sort(key=lambda x: x[0], reverse=True)  # most prevalent cluster first
    sorted_groups = [z[1] for z in zipped]  # sorted(grouped_by_cluster, key=len, reverse=True)
    padded_and_transposed = zip_longest(*sorted_groups, fillvalue='')
    text_report = list(map(list, padded_and_transposed))
    headers = ['Cluster #%i (sorted by CT prevalence)' % i
               for i in range(len(grouped_by_cluster))]
    return [headers] + text_report


def generate_stat_report(cluster_typicals: list[str],
                         cluster_prevalences: list[float],
                         label_prevalences: list[dict],
                         unique_labels: list[str],
                         n_cts: int,
                         n_ecs: int,
                         ) -> list[list[str]]:
    """ Write a report for all cluster, using cluster typical representative
        text, CT prevalence between clusters and label prevalence within cluster
    """
    zipped = list(zip(cluster_prevalences, cluster_typicals, label_prevalences))
    zipped.sort(key=lambda x: x[0], reverse=True)  # most prevalent cluster first
    headers = ['Absolute cluster prevalence', 'Cluster representative']
    headers += ['Relative status prevalence (%s)' % s for s in unique_labels]
    report_lines = []
    for abs_prev, txt, rel_prev_ordered_dict in zipped:
        line = ['%i%%' % (abs_prev * 100), txt]
        line.extend(['%i%%' % (v * 100) for v in rel_prev_ordered_dict.values()])
        report_lines.append(line)
    final_line = 'Reporting %s similar CTs, including %s ECs' % (n_cts, n_ecs)
    return [headers] + report_lines + [[final_line]]


def compute_cluster_statistics(token_info, cluster_info):
    """ Compute CT prevalence between clusters & label prevalence within clusters
    """        
    # Compute absolute cluster prevalence by counting clinical trials
    zipped_paths = list(
        zip(token_info['ct_paths'], cluster_info['cluster_lbls'])
    )
    cluster_sample_paths = [
        [p for p, l in zipped_paths if l == cluster_id]
        for cluster_id in range(cluster_info['n_clusters'])
    ]
    n_cts = len(set(token_info['ct_paths']))
    cluster_prevalences = [len(set(ps)) / n_cts for ps in cluster_sample_paths]
    
    # Compute relative label prevalence inside each cluster
    zipped_labels = list(
        zip(token_info['ct_lbls'], cluster_info['cluster_lbls'])
    )
    cluster_sample_labels = [
        [p for p, l in zipped_labels if l == cluster_id]
        for cluster_id in range(cluster_info['n_clusters'])
    ]
    label_prevalences = [
        compute_proportions(c, token_info['unique_lbls'])
        for c in cluster_sample_labels
    ]
    
    # Return computed statistics
    return cluster_prevalences, label_prevalences


def compute_proportions(lbl_list, unique_lbls):
    """ Compute proportion of each CT label within each cluster
        TODO: count only one criterion per CT (here: count all criteria)
    """
    result = OrderedDict([(k, 0.0) for k in unique_lbls])
    counter = Counter(lbl_list)
    total_count = len(lbl_list)
    result.update({lbl: count / total_count for lbl, count in counter.items()})
    return result

   
def pool_cluster_criteria(criterion_txts, method='chatgpt'):
    """ For each cluster, summarize all belonging criteria to a single sentence
    """
    # Take the criterion closest to the cluster medoid
    if method == 'closest':
        pooled_criterion_txts = [txts[0] for txts in criterion_txts]
    
    # Take shortest from the 10 criteria closest to the cluster medoid
    elif method == 'shortest':
        pooled_criterion_txts = [min(txts, key=len) for txts in criterion_txts]
    
    # Use ChatGPT-3.5 to summarize the 10 criteria closest to the cluster medoid
    elif method == 'chatgpt':
        # Authentificate with a valid api-key
        api_path = os.path.join('data', 'api-key-risklick.txt')
        try:
            with open(api_path, 'r') as f: openai.api_key = f.read()
        except:
            raise FileNotFoundError('You must have an api-key at %s' % api_path)
        
        # Define system context and basic prompt
        system_prompt = '\n'.join([
            'You are an expert in the fields of clinical trials and eligibility criteria.',
            'You express yourself succintly, i.e., less than 250 characters per response.',
        ])
        base_user_prompt = '\n'.join([
            'I have a list of eligility criteria. They all look alike.',
            'I need you to generate one clear and short eligibility rule that best covers the list without focusing on the specific details of each criterion.',
            'You should give more weight to the first elements of the list, as they are more likely to be representative.'
            'Your answer should be *one* rule that starts with either "Inclusion criterion - " or "Exclusion criterion - ".',
            'Please just write your answer. Here is the list of criteria (each one is on a new line):\n',
        ])
        
        # Prompt the model and collect answer for each criterion
        pooled_criterion_txts = []
        prompt_loop = tqdm(criterion_txts, 'Prompting GPT to summarize clusters')
        for cluster_criteria in prompt_loop:
            user_prompt = base_user_prompt + '\n'.join(cluster_criteria)
            response = prompt_chatgpt(system_prompt, user_prompt)
            post_processed = 'criterion - '.join(
                [s.capitalize() for s in response.split('criterion - ')]
            )
            pooled_criterion_txts.append(post_processed)
    
    # Handle wrong method name and return the results
    else:
        raise ValueError('Wrong pooling method selected.')
    return pooled_criterion_txts


def prompt_chatgpt(system_prompt, user_prompt, max_retries=5):
    """ Collect answer of chat-gpt, given system and user prompts, and avoiding
        rate limit by waiting an exponentially increasing amount of time
    """
    for i in range(max_retries):  # hard limit
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            return response['choices'][0]['message']['content'].strip()
        except RateLimitError:
            time.sleep(2 ** i)
    return 'No response, open-ai reached rate limit.'


def plot_cluster_hierarchy(token_info: dict,
                           cluster_info: dict,
                           fig: plt.Figure,
                           red_dim: int
                           ) -> None:
    """ Plot theoretical clusters (using labels), empirical clusters (using
        hdbscan), and the empirical cluster tree
    """
    # Load data
    plot_data = token_info['plot_data']
    plot_kwargs = {} if red_dim <= 2 else {'projection': '3d'}
    n_labels = len(set(token_info['ct_lbls']))
    n_clusters = cluster_info['cluster_lbls'].max() + 1
    
    # Assign label colors and evaluate cluster purity (not ready yet)    
    cluster_colors, class_colors = assign_cluster_colors(token_info, cluster_info)
    # cluster_labels, class_labels = compute_labels(cluster_colors, class_colors)
    # ari_match = adjusted_rand_score(class_labels, cluster_labels)
    # nmi_match = normalized_mutual_info_score(class_labels, cluster_labels)
    # homog, compl, v_match = hcv_measure(class_labels, cluster_labels)
    # print(' - ari: %.3f\n - nmi: %.3f\n - hom: %.3f\n - comp: %.3f\n - v: %.3f'%
    #       (ari_match, nmi_match, homog, compl, v_match))
    
    # Visualize theoretical clusters (using, e.g., ICD10-CM code hierarchy)
    ax0 = fig.add_subplot(1, 3, 1, **plot_kwargs)
    ax0.scatter(*plot_data.T, c=class_colors, **SCATTER_PARAMS)
    ax0.set_xticklabels([]); ax0.set_yticklabels([])
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title('Data and labels (N = %s)' % n_labels, fontsize=TEXT_SIZE)
    
    # Visualize empirical clusters
    ax1 = fig.add_subplot(1, 3, 2, **plot_kwargs)
    ax1.scatter(*plot_data.T, c=cluster_colors, **SCATTER_PARAMS)
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title('Empirical clusters (M = %s)' % n_clusters, fontsize=TEXT_SIZE)
    
    # Visualize empirical cluster tree
    ax2 = fig.add_subplot(1, 3, 3)
    cluster_info['clusterer'].condensed_tree_.plot(
        axis=ax2, leaf_separation=LEAF_SEPARATION, colorbar=True)
    ax2.set_title('Empirical cluster tree', fontsize=TEXT_SIZE)
    ax2.get_yaxis().set_tick_params(left=False)
    ax2.get_yaxis().set_tick_params(right=False)
    ax2.set_ylabel('', fontsize=TEXT_SIZE)
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    
    
def assign_cluster_colors(token_info, cluster_info):
    """ Used to be more complicated, but for now just assign fixed colours
    """
    # Generate class and cluster indices
    token_classes = sorted(set(token_info['ct_lbls']))
    cluster_classes = sorted(set(cluster_info['cluster_lbls']))
    token_to_id = {v: i for i, v in enumerate(token_classes)}
    cluster_to_id = {v: i for i, v in enumerate(cluster_classes)}
    
    # Create class and cluster colors for all samples
    cluster_colors = [COLORS[cluster_to_id[c]] if c >= 0 else NA_COLOR
                      for c in cluster_info['cluster_lbls']]
    class_colors = [COLORS[token_to_id[c]] for c in token_info['ct_lbls']]
    return cluster_colors, class_colors


def compute_reduced_repr(embeddings: np.ndarray,
                         dim='2',
                         algorithm='pca'
                         ) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """    
    # No dimensionality reduction
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # ?
    if dim == None or algorithm == None:
        return embeddings
    
    # Simple PCA algorithm
    if algorithm == 'pca':
        return PCA().fit_transform(embeddings)[:, :dim]
    
    # More computationally costly t-SNE algorithm
    elif algorithm == 'tsne':
        max_cluster_samples = int(len(embeddings) * MAX_CLUSTER_SIZE)
        min_cluster_samples = int(len(embeddings) * MIN_CLUSTER_SIZE)
        perplexity = min(100, (min_cluster_samples + max_cluster_samples) // 10)
        params = {
            'n_components': dim,
            'perplexity': perplexity,  # 10.0, 30.0, 100.0, perplexity
            'learning_rate': 'auto',  # or any value in [10 -> 1000] may be good
            'n_iter': 10000,  # 10000, 2000
            'n_iter_without_progress': 1000,  # 200, 1000
            'metric': 'cosine',
            'square_distances': True,  # future default behaviour
            'init': 'random',  # 'pca', 'random'
            'n_jobs': N_CPUS,
            'verbose': 2,
            'method': 'barnes_hut' if dim < 4 else 'exact',  # 'exact' very slow
        }
        return TSNE(**params).fit_transform(embeddings)
    