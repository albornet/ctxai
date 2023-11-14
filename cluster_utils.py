# Mains
import os
import time
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import cluster_config as cfg

# Utils
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure as hcv_measure,
)
from torchmetrics.clustering import DunnIndex
from collections import Counter, OrderedDict
from itertools import zip_longest, product
from tqdm import tqdm

# OpenAI
import openai
from requests.exceptions import ConnectionError
from openai import (
    RateLimitError,
    OpenAIError,
    APIError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

# Optuna, cuml, and dask
import optuna
from optuna.integration.dask import DaskStorage
from optuna.samplers import TPESampler
if not cfg.USE_CUML:  # not supported yet
    import hdbscan
    from sklearn.manifold import TSNE
else:
    import dask.array as da
    from cuml.internals.array import CumlArray as CumlArrayClass
    from distributed.client import Future as FutureArrayClass
    from dask_cuda import LocalCUDACluster
    from dask.distributed import LocalCluster, Client
    from cuml.manifold import TSNE
    from cuml.cluster import hdbscan


def report_clusters(model_type: str,
                    embeddings: torch.Tensor,
                    raw_txt: list[str],
                    metadata: list[str],
                    label_type: str="status",  # status, phase, condition, intervention, label
                    ) -> tuple[str, str, dict]:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    # Reduce data dimensionality and get token information
    plot_data, cluster_data = get_plot_and_cluster_data(embeddings, model_type)
    token_info = get_token_info(plot_data, raw_txt, metadata, label_type)
    
    # Look for best set of hyper-parameters for clustering and cluster
    params = find_best_cluster_params(cluster_data, model_type)
    cluster_info = cluster_criteria(params, cluster_data, token_info, model_type)
    
    # Plot and evaluate clusters and their content
    plot_cluster_hierarchy(token_info, cluster_info, model_type)
    stats, texts, metrics = generate_cluster_reports(token_info, cluster_info, metadata)
    
    # Return all results
    return stats, texts, metrics


def get_token_info(plot_data: np.ndarray,
                   raw_txt: list[str],
                   metadata: list[str],
                   label_type: str,
                   ) -> dict:
    """ Wrap-up raw information about text and plotting
    """
    true_lbls = [l[label_type] for l in metadata]
    token_info = {
        "plot_data": plot_data,
        "raw_txt": raw_txt,
        "true_lbls": true_lbls,
        "unique_lbls": list(set(true_lbls)),
        "paths": [l["path"] for l in metadata],
    }
    return token_info


def get_reduced_data(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Reduce data using t-SNE (or PCA) and return two separate arrays, one for
        clustering and one for plotting
    """
    # Compute reduced representation for clustering
    print("\nReducing dim of %s eligibility criteria embeddings" % len(embeddings))
    cluster_data = compute_reduced_repr(
        embeddings,
        dim=cfg.CLUSTER_RED_DIM,
        algorithm=cfg.CLUSTER_DIM_RED_ALGO,
    )
    
    # Compute reduced representation for plotting
    if cfg.CLUSTER_RED_DIM == cfg.PLOT_RED_DIM\
    and cfg.CLUSTER_DIM_RED_ALGO == cfg.PLOT_DIM_RED_ALGO:
        plot_data = cluster_data
    else:
        plot_data = compute_reduced_repr(
            cluster_data,  # embeddings?
            dim=cfg.PLOT_RED_DIM,
            algorithm=cfg.PLOT_DIM_RED_ALGO,
        )
    
    # Return the results
    return cluster_data, plot_data


def get_plot_and_cluster_data(embeddings: torch.Tensor,
                              model_type: str
                              ) -> tuple[np.ndarray, np.ndarray]:
    """ Compute low-dimensional plot and cluster data using dimensionality
        reduction algorithm, or load already computed results 
    """
    # Load already computed data
    load_path = os.path.join(cfg.OUTPUT_DIR, "reduced_embeddings_%s.pkl" % model_type)
    if cfg.LOAD_REDUCED_EMBEDDINGS:
        with open(load_path, "rb") as f:
            cluster_data = pickle.load(f)
            plot_data = cluster_data
    
    # Recompute reduced dimensionality representation of data and save it
    else:
        cluster_data, plot_data = get_reduced_data(embeddings.numpy())
        with open(load_path, "wb") as f:
            pickle.dump(plot_data, f)
    
    # Return the results
    return plot_data, cluster_data


def cluster_criteria(params: dict,
                     data: np.ndarray,
                     token_info: dict,
                     model_type: str
                     ) -> dict:
    """ Cluster criteria and save the results, or load cluster information from
        a previous run, with specific model embeddings 
    """
    # Load already computed clustering results
    load_path = os.path.join(cfg.OUTPUT_DIR, "cluster_info_%s.pkl" % model_type)
    if cfg.LOAD_CLUSTER_INFO:
        with open(load_path, "rb") as f:
            cluster_info = pickle.load(f)
    
    # Run clustering algorithm with selected hyper-parameters and save results
    else:
        print("Clustering %s eligibility criteria" % len(token_info["plot_data"]))
        cluster_info = clusterize(data=data, mode="primary", params=params)
        # if cluster_info is not None and cfg.DO_SUBCLUSTERIZE:
        #     cluster_info = subclusterize(cluster_info, params=params)
        with open(load_path, "wb") as f:
            pickle.dump(cluster_info, f)
    
    # Return clustering results
    return cluster_info


def find_best_cluster_params(cluster_data: np.ndarray, model_type: str) -> dict:
    """ Use optuna to determine best set of cluster parameters (or load them)
    """
    # Try to load best hyper-parameters and return defaults otherwise
    params_path = os.path.join(cfg.RESULT_DIR, "params_%s.json" % model_type)
    if cfg.LOAD_OPTUNA_RESULTS:
        try:
            with open(params_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print("Clustering parameters not found, using default parameters")
            return cfg.DEFAULT_CLUSTERING_PARAMS

    # Find and save best hyper-parameters
    else:
        print("Looking for best clustering hyper-parameters")
        with LocalCluster(
            n_workers=cfg.NUM_GPUS,
            threads_per_worker=cfg.NUM_PARALLEL_OPTUNA_TRIALS // cfg.NUM_GPUS,
            processes=True,
            device_memory_limit='22GB'
        ) as cluster:
            with Client(cluster, timeout='60s') as client:
                db_path = "sqlite:///%s/optuna_%s.db" % (cfg.RESULT_DIR, model_type)
                study = optuna.create_study(
                    sampler=TPESampler(),
                    direction="maximize",
                    storage=DaskStorage(db_path)
                )
                cluster_data = da.from_array(cluster_data)  # dask array for GPU computations
                objective = lambda trial: objective_fn(trial, cluster_data)
                futures = [
                    client.submit(study.optimize, objective, n_trials=1, pure=False)
                    for _ in range(cfg.N_OPTUNA_TRIALS)
                ]
                client.gather(futures)
                # study.optimize(objective, n_trials=cfg.N_OPTUNA_TRIALS, n_jobs=-1)
                best_params = study.best_params
                
        print("Best params: %s" % best_params)
        with open(params_path, "w") as file:
            json.dump(best_params, file, indent=4)
        return best_params


def objective_fn(trial: optuna.Trial, data: np.ndarray) -> float:
    """ Suggest a new set of hyper-parameters and compute associated metric
    """
    # Perform clustering with a new set of suggested parameters
    params = suggest_parameters(trial)  # if isinstance(data, FutureArrayClass): data = data.result()
    cluster_info = clusterize(data=data, mode="primary", params=params)
    # if cluster_info['n_clusters'] > 1 and cfg.DO_SUBCLUSTERIZE:
    #     cluster_info = subclusterize(cluster_info, params=params)
    
    # Compute metric with clustering results
    if 1 < cluster_info['n_clusters'] < cfg.N_CLUSTER_MAX:
        cluster_lbls = cluster_info["cluster_lbls"]
        metric_1 = silhouette_score(data, cluster_lbls)
        metric_2 = 1.0 - np.count_nonzero(cluster_lbls == -1) / len(cluster_lbls)
        metric = metric_1 + metric_2
    else:
        metric = float("-inf")
    
    # Return metric
    return metric


def suggest_parameters(trial):
    """ Suggest parameters following configured parameter ranges and types
    """
    params = {}
    for name, choices in cfg.OPTUNA_PARAM_RANGES.items():
        if not isinstance(choices, (list, tuple)):
            raise TypeError("Boundary must be a list or a tuple")
        if isinstance(choices[0], (str, bool)):
            params[name] = trial.suggest_categorical(name, choices)
            continue
        low, high = choices
        if isinstance(low, float) and isinstance(high, float):
            params[name] = trial.suggest_float(name, low, high)
        elif isinstance(low, int) and isinstance(high, int):
            params[name] = trial.suggest_int(name, low, high)
        else:
            raise TypeError("Boundary type mismatch")
    
    return params


def set_cluster_params(n_samples: int, mode: str, params: dict) -> dict:
    """ Adapt clustering parameters following dataset size and clustering mode
    """
    # Load base parameters
    max_cluster_size = params["max_cluster_size_%s" % mode]
    min_cluster_size = params["min_cluster_size_%s" % mode]
    min_samples = params["min_samples_%s" % mode]
    
    # Adapter function (int vs float, min and max values)
    default_max_value = n_samples - 1
    def adapt_param_fn(value, min_value, max_value=default_max_value):
        if isinstance(value, float): value = int(n_samples * value)
        return min(max(min_value, value), max_value)
    
    # Return all selected cluster parameters
    return {
        "cluster_selection_method": params["cluster_selection_method"],
        "alpha": params["alpha"],
        "allow_single_cluster": True,
        "max_cluster_size": adapt_param_fn(max_cluster_size, 100),
        "min_cluster_size": adapt_param_fn(min_cluster_size, 10),
        "min_samples": adapt_param_fn(min_samples, 10, 1023),  # 2 ** 10 - 1
    }


def clusterize(data: np.ndarray, mode: str, params: dict) -> dict:
    """ Cluster data points with hdbscan algorithm and return cluster information
    """
    # Identify best cluster parameters given the data and cluster mode
    cluster_params = set_cluster_params(len(data), mode, params)
    
    # Find cluster affordances based on cluster hierarchy
    clusterer = hdbscan.HDBSCAN(**cluster_params)
    clusterer.fit(data)
    os.system("nvidia-smi")
    lbls = clusterer.labels_
    if isinstance(lbls, CumlArrayClass): lbls = lbls.to_host_array()
    n_clusters = lbls.max() + 1  # -1 being ignored
    
    # Put back unclustered samples if secondary mode
    if mode == "secondary":
        unclustered_ids = np.where(lbls == -1)
        lbls[unclustered_ids] = 0

    # Find global ids of all members of each cluster
    cluster_member_ids = {k: np.where(lbls == k)[0] for k in range(-1, n_clusters)}
    
    # Return cluster info
    return {
        "clusterer": clusterer,
        "n_clusters": n_clusters,
        "cluster_data": data,
        "cluster_lbls": lbls,
        "cluster_member_ids": cluster_member_ids,
    }
    
    
def subclusterize(cluster_info: dict, params: dict) -> dict:
    """ Update cluster results by trying to subcluster any computed cluster
    """
    # For each cluster, try to cluster it further with new hdbscan parameters
    for cluster_id in range(cluster_info["n_clusters"]):  # not range(-1, ...)
        subset_ids = np.where(cluster_info["cluster_lbls"] == cluster_id)[0]
        subset_data = cluster_info["cluster_data"][subset_ids]
        new_cluster_info = clusterize(data=subset_data, mode="secondary", params=params)
        
        # If the sub-clustering is successful, record new information
        if new_cluster_info["n_clusters"] > 1:  # new_cluster_info is not None:
            n_new_clusters = new_cluster_info["n_clusters"]
            new_cluster_ids =\
                [cluster_id] +\
                [cluster_info["n_clusters"] + i for i in range(n_new_clusters - 1)]
            cluster_info["n_clusters"] += len(new_cluster_ids)
            
            # And update cluster labels and cluster member ids
            for i, new_cluster_id in enumerate(new_cluster_ids):
                new_member_ids = new_cluster_info["cluster_member_ids"][i]
                new_member_ids = subset_ids[new_member_ids]  # in original clustering
                cluster_info["cluster_lbls"][new_member_ids] = new_cluster_id
                cluster_info["cluster_member_ids"][new_cluster_id] = new_member_ids
    
    # Return updated cluster info
    return cluster_info


def generate_cluster_reports(token_info, cluster_info, metadata):
    """ Characterize clusters and generate statistics given clinical trials
    """
    # Compute useful statistics and metrics
    cluster_prevalences, label_prevalences, sorted_cluster_member_ids =\
        compute_cluster_statistics(token_info, cluster_info)
    
    # Identify one "typical" criterion string for each cluster of criteria
    txts_grouped_by_cluster = [
        [token_info["raw_txt"][i] for i in sample_ids]
        for k, sample_ids in sorted_cluster_member_ids.items() if k != -1
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
        unique_labels=token_info["unique_lbls"],
        n_cts=len(set(token_info["paths"])),
        n_ecs=len(token_info["paths"]),
    )
    cluster_metrics = evaluate_clustering(cluster_info, metadata)
    return text_report, stat_report, cluster_metrics


def compute_cluster_statistics(token_info, cluster_info):
    """ Compute CT prevalence between clusters & label prevalence within clusters
    """        
    # Compute absolute cluster prevalence by counting clinical trials
    zipped_paths = list(zip(token_info["paths"], cluster_info["cluster_lbls"]))
    cluster_sample_paths = [
        [p for p, l in zipped_paths if l == cluster_id]
        for cluster_id in range(cluster_info["n_clusters"])
    ]
    n_cts = len(set(token_info["paths"]))
    cluster_prevalences = [len(set(ps)) / n_cts for ps in cluster_sample_paths]
    
    # Compute relative label prevalence inside each cluster
    zipped_labels = list(zip(token_info["true_lbls"], cluster_info["cluster_lbls"]))
    cluster_sample_labels = [
        [p for p, l in zipped_labels if l == cluster_id]
        for cluster_id in range(cluster_info["n_clusters"])
    ]
    label_prevalences = [
        compute_proportions(c, token_info["unique_lbls"])
        for c in cluster_sample_labels
    ]
    
    # Sort cluster member ids by how close each member is to its cluster medoid
    medoids = compute_weighted_cluster_medoids(cluster_info)
    dist_fn = lambda k, v: cdist([medoids[k]], cluster_info["cluster_data"][v])[0]
    sorted_member_ids = {
        k: [sorted_id for _, sorted_id in sorted(zip(dist_fn(k, v), v))]
        for k, v in cluster_info["cluster_member_ids"].items()  # pas ouf clair
    }
    
    # Return computed statistics
    return cluster_prevalences, label_prevalences, sorted_member_ids


def compute_weighted_cluster_medoids(cluster_info: dict) -> dict:
    """ Compute cluster centroids by averaging samples within each cluster,
        weighting by the sample probability of being in that cluster
    """
    cluster_medoids = {}
    for label in range(-1, cluster_info["n_clusters"]):  # including unassigned
        cluster_ids = np.where(cluster_info["cluster_lbls"] == label)[0]
        cluster_data = cluster_info["cluster_data"][cluster_ids]
        cluster_medoids[label] = compute_medoid(cluster_data)
        
    return cluster_medoids


def compute_medoid(samples: np.ndarray) -> np.ndarray:
    """ Compute medoids of a subset of samples of shape (n_samples, n_features)
        Distance computations are made with dask to mitigate memory requirements
    """
    samples_dask = da.from_array(samples, chunks='auto')
    distance_matrix = da.map_blocks(cdist, samples_dask, samples_dask, dtype=float)
    distance_sum = distance_matrix.sum(axis=0)
    distance_sum_computed = distance_sum.compute()
    medoid_id = np.argmin(distance_sum_computed)
    return samples[medoid_id]


# def compute_medoid(samples: np.ndarray, batch_size: int=1000) -> np.ndarray:
#     """ Compute medoids of a subset of samples of shape (n_samples, n_features)
#         The computation is made incrementally to lower the memory requirements
#     """
#     n_samples = len(samples)
#     distance_sums = np.zeros(n_samples)
#     for i in range(0, n_samples, batch_size):
#         batch = samples[i:i + batch_size]
#         partial_distance_matrix = cdist(batch, samples)
#         distance_sums += partial_distance_matrix.sum(axis=0)
        
#     medoid_id = np.argmin(distance_sums)
#     return samples[medoid_id]


def generate_text_report(grouped_by_cluster: list[list[str]],
                         cluster_prevalences: list[float],
                         ) -> list[list[str]]:
    """ Generate a report to write all clusters" criteria in a csv file
    """
    zipped = list(zip(cluster_prevalences, grouped_by_cluster))
    zipped.sort(key=lambda x: x[0], reverse=True)  # most prevalent cluster first
    sorted_groups = [z[1] for z in zipped]  # sorted(grouped_by_cluster, key=len, reverse=True)
    padded_and_transposed = zip_longest(*sorted_groups, fillvalue="")
    text_report = list(map(list, padded_and_transposed))
    headers = ["Cluster #%i (sorted by CT prevalence)" % i
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
    headers = ["Absolute cluster prevalence", "Cluster representative"]
    headers += ["Relative status prevalence (%s)" % s for s in unique_labels]
    report_lines = []
    for abs_prev, txt, rel_prev_ordered_dict in zipped:
        line = ["%i%%" % (abs_prev * 100), txt]
        line.extend(["%i%%" % (v * 100) for v in rel_prev_ordered_dict.values()])
        report_lines.append(line)
    final_line = "Reporting %s similar CTs, including %s ECs" % (n_cts, n_ecs)
    return [headers] + report_lines + [[final_line]]


def compute_proportions(lbl_list, unique_lbls):
    """ Compute proportion of each CT label within each cluster
        TODO: count only one criterion per CT (here: count all criteria)
    """
    result = OrderedDict([(k, 0.0) for k in unique_lbls])
    counter = Counter(lbl_list)
    total_count = len(lbl_list)
    result.update({lbl: count / total_count for lbl, count in counter.items()})
    return result


def pool_cluster_criteria(criterion_txts):
    """ For each cluster, summarize all belonging criteria to a single sentence
    """
    # Take the criterion closest to the cluster medoid
    if cfg.CLUSTER_SUMMARIZATION_METHOD == "closest":
        pooled_criterion_txts = [txts[0] for txts in criterion_txts]
    
    # Take shortest from the 10 criteria closest to the cluster medoid
    elif cfg.CLUSTER_SUMMARIZATION_METHOD == "shortest":
        pooled_criterion_txts = [min(txts, key=len) for txts in criterion_txts]
    
    # Use ChatGPT-3.5 to summarize the 10 criteria closest to the cluster medoid
    elif cfg.CLUSTER_SUMMARIZATION_METHOD == "chatgpt":
        # Authentificate with a valid api-key
        api_path = os.path.join("data", "api-key-risklick.txt")
        try:
            with open(api_path, "r") as f: openai.api_key = f.read()
        except:
            raise FileNotFoundError("You must have an api-key at %s" % api_path)
        
        # Define system context and basic prompt
        system_prompt = "\n".join([
            "You are an expert in the fields of clinical trials and eligibility criteria.",
            "You express yourself succintly, i.e., less than 250 characters per response.",
        ])
        base_user_prompt = "\n".join([
            "I will show you a list of eligility criteria. They share some level of similarity.",
            "I need you to generate one small tag that best represents the list.",
            "The tag should be short, specific, and concise, and should not focus too much on details or outliers.",
            'Importantly, you should only write your answer, and it should be one single phrase that starts with either "Inclusion criterion - " or "Exclusion criterion - " (but choose only one of them).',
            "Here is the list of criteria (each one is on a new line):\n",
        ])
        
        # Prompt the model and collect answer for each criterion
        pooled_criterion_txts = []
        prompt_loop = tqdm(criterion_txts, "Prompting GPT to summarize clusters")
        for cluster_criteria in prompt_loop:
            user_prompt = base_user_prompt + "\n".join(cluster_criteria)
            response = prompt_chatgpt(system_prompt, user_prompt)
            post_processed = "criterion - ".join(
                [s.capitalize() for s in response.split("criterion - ")]
            )
            pooled_criterion_txts.append(post_processed.replace("\n", " "))
    
    # Handle wrong method name and return the results
    else:
        raise ValueError("Wrong pooling method selected.")
    return pooled_criterion_txts


def prompt_chatgpt(system_prompt, user_prompt, max_retries=5):
    """ Collect answer of chat-gpt, given system and user prompts, and avoiding
        rate limit by waiting an exponentially increasing amount of time
    """
    for i in range(max_retries):  # hard limit
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response["choices"][0]["message"]["content"].strip()

        except (RateLimitError, ConnectionError, OpenAIError, APIError,
                APIStatusError, APITimeoutError, APIConnectionError) as e:
            print("An error occurred: %s. Retrying in %i seconds." % (e, 2 ** i))
            time.sleep(2 ** i)
            
    return "No response, open-ai reached rate limit or another network issue occurred."


def plot_cluster_hierarchy(token_info: dict,
                           cluster_info: dict,
                           model_type: str,
                           subset_ids: list[int]=None,
                           ) -> None:
    """ Plot theoretical clusters (using labels), empirical clusters (using
        hdbscan), and the empirical cluster tree
    """
    # Initialize figure and load data
    fig_path = os.path.join(cfg.RESULT_DIR, "plot_%s.png" % model_type)
    fig = plt.figure(figsize=cfg.FIG_SIZE)
    plot_kwargs = {} if cfg.PLOT_RED_DIM <= 2 else {"projection": "3d"}
    true_labels = token_info["true_lbls"]
    cluster_labels = cluster_info["cluster_lbls"]
    n_labels = len(set(true_labels))
    n_clusters = cluster_labels.max() + 1
    print("Plotting %s criteria embeddings" % len(true_labels))
    
    # Retrieve relevant data
    cluster_colors, class_colors = get_cluster_colors(token_info, cluster_info)
    plot_data = token_info["plot_data"]
    if subset_ids is not None:  # take subset for secondary cluster displays
        plot_data = plot_data[subset_ids]
        class_colors = [class_colors[i] for i in subset_ids[0]]
        
    # Visualize empirical clusters
    ax1 = fig.add_subplot(2, 2, 1, **plot_kwargs)
    ax1.scatter(*plot_data.T, c=cluster_colors, **cfg.SCATTER_PARAMS)
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Data and %s clusters" % n_clusters, fontsize=cfg.TEXT_SIZE)
    
    # Visualize empirical clusters (but only the samples with an assigned cluster)
    cluster_only_indices = np.where(cluster_labels > -1)[0]
    clustered_data = plot_data[cluster_only_indices]
    clustered_colors = [cluster_colors[i] for i in cluster_only_indices]    
    ax2 = fig.add_subplot(2, 2, 2, **plot_kwargs)
    ax2.scatter(*clustered_data.T, c=clustered_colors, **cfg.SCATTER_PARAMS)
    ax2.set_xticklabels([]); ax2.set_yticklabels([])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Data and %s assigned clusters" % (n_clusters), fontsize=cfg.TEXT_SIZE)
    
    # Visualize theoretical clusters (using, e.g., ICD10-CM code hierarchy)
    ax3 = fig.add_subplot(2, 2, 3, **plot_kwargs)
    ax3.scatter(*plot_data.T, c=class_colors, **cfg.SCATTER_PARAMS)
    ax3.set_xticklabels([]); ax3.set_yticklabels([])
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("Data and %s labels" % n_labels, fontsize=cfg.TEXT_SIZE)
    
    # Visualize empirical cluster tree
    if hasattr(cluster_info["clusterer"], "condensed_tree_"):
        ax4 = fig.add_subplot(2, 2, 4)
        cluster_info["clusterer"].condensed_tree_.plot(
            axis=ax4, leaf_separation=cfg.LEAF_SEPARATION, colorbar=True)
        ax4.set_title("Empirical cluster tree", fontsize=cfg.TEXT_SIZE)
        ax4.get_yaxis().set_tick_params(left=False)
        ax4.get_yaxis().set_tick_params(right=False)
        ax4.set_ylabel("", fontsize=cfg.TEXT_SIZE)
        ax4.set_yticks([])
        ax4.spines["top"].set_visible(True)
        ax4.spines["right"].set_visible(True)
        ax4.spines["bottom"].set_visible(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    

def compute_reduced_repr(embeddings: np.ndarray,
                         dim: int,
                         algorithm: str,
                         ) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """    
    # No dimensionality reduction
    # embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # ?
    if dim == None or algorithm == None:
        return embeddings
    
    # Simple PCA algorithm
    if algorithm == "pca":
        return PCA().fit_transform(embeddings)[:, :dim]
    
    # More computationally costly t-SNE algorithm
    elif algorithm == "tsne":
        params = {
            "n_components": dim,
            "method": "barnes_hut" if dim < 4 else "exact",  # "exact" very slow
            "n_iter": cfg.N_ITER_MAX_TSNE,
            "n_iter_without_progress": 1000,
            "learning_rate": max(len(embeddings) / 12.0, 200),
            "metric": "cosine",
            "verbose": True,
            "perplexity": 30.0,
        }
        tsne = TSNE(**params)
        return tsne.fit_transform(embeddings)


def get_cluster_colors(token_info, cluster_info):
    """ Find the best match between clusters and labels to assigns colors
    """
    # Identify possible colour values
    class_lbls = token_info["true_lbls"]
    cluster_lbls = cluster_info["cluster_lbls"]
    unique_classes = list(set(class_lbls))
    unique_clusters = list(set(cluster_lbls))
    
    # In this case, true labels define the maximum number of colours
    if len(unique_classes) >= len(unique_clusters):
        color_map = best_color_match(
            cluster_lbls, class_lbls, unique_clusters, unique_classes)
        color_map = {k: unique_classes.index(v) for k, v in color_map.items()}
        cluster_colors = [cfg.COLORS[color_map[i]]
                          if i != -1 else cfg.NA_COLOR for i in cluster_lbls]
        class_colors = [cfg.COLORS[unique_classes.index(l)]
                        if l != -1 else cfg.NA_COLOR for l in class_lbls]
    
    # In this case, empirical clusters define the maximum number of colours
    else:
        color_map = best_color_match(
            class_lbls, cluster_lbls, unique_classes, unique_clusters)
        color_map = {unique_classes.index(k): v for k, v in color_map.items()}
        cluster_colors = [cfg.COLORS[i]
                          if i >= 0 else cfg.NA_COLOR for i in cluster_lbls]
        class_colors = [cfg.COLORS[color_map[unique_classes.index(l)]]
                        if l != -1 else cfg.NA_COLOR for l in class_lbls]
        
    # Return aligned empirical and theorical clusters
    return cluster_colors, class_colors


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


def evaluate_clustering(cluster_info, metadata):
    """ Run final evaluation of clusters, based on phase(s), condition(s), and
        interventions(s). Duplicate each samples for any combination.    
    """
    # Initialize data
    cluster_metrics = {}
    cluster_data = cluster_info["cluster_data"]
    cluster_lbls = cluster_info["cluster_lbls"]
    phase_lbls = [l["phase"] for l in metadata]
    cond_lbls = [l["condition"] for l in metadata]
    itrv_lbls = [l["intervention"] for l in metadata]
    
    # Create a new sample for each [phase, cond, itrv] label combination
    dupl_cluster_lbls = []
    dupl_true_lbls = []
    for cluster_lbl, phases, conds, itrvs in\
        zip(cluster_lbls, phase_lbls, cond_lbls, itrv_lbls):
        true_lbl_combinations = list(product(phases, conds, itrvs))
        for true_lbl_combination in true_lbl_combinations:
            dupl_cluster_lbls.append(cluster_lbl)
            dupl_true_lbls.append(" - ".join(true_lbl_combination))
    
    # Evaluate clustering quality (label-free)
    cluster_metrics["label_free"] = {
        "sil_score": silhouette_score(cluster_data, cluster_lbls),
        "db_score": davies_bouldin_score(cluster_data, cluster_lbls),
        "dunn_score": dunn_index(cluster_data, cluster_lbls),
    }
    
    # Evaluate clustering quality (label-dependent)
    h, c, v = hcv_measure(dupl_true_lbls, dupl_cluster_lbls)
    cluster_metrics["label_dept"] = {
        "ar_score": adjusted_rand_score(dupl_true_lbls, dupl_cluster_lbls),
        "nm_score": normalized_mutual_info_score(dupl_true_lbls, dupl_cluster_lbls),
        "fm_score": fowlkes_mallows_score(dupl_true_lbls, dupl_cluster_lbls),
        "homogeneity": h,
        "completeness": c,
        "v_measure": v,
    }
    
    # Return all metrics and some info
    cluster_metrics["n_samples"] = len(cluster_lbls)
    cluster_metrics["n_duplicated_samples"] = len(dupl_cluster_lbls)
    return cluster_metrics


def dunn_index(cluster_data: np.ndarray, cluster_lbls: np.ndarray) -> float:
    """ Compute Dunn index using torch-metrics
    """
    dunn = DunnIndex(p=2)
    metric = dunn(torch.from_numpy(cluster_data), torch.from_numpy(cluster_lbls))
    return metric.item()
