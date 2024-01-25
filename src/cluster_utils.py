# Mains
import os
import time
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
try:
    from . import config as cfg
except ImportError:
    import src.config as cfg

# Utils
from scipy.spatial.distance import squareform
from cupyx.scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from itertools import product
from typing import Union
from tqdm import tqdm
from dataclasses import dataclass

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
import dask.array as da
from optuna.samplers import TPESampler
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml.cluster import hdbscan
from cuml.decomposition import PCA
from cuml.manifold import TSNE

# Metrics
from cuml.metrics.cluster import silhouette_score
from torchmetrics.clustering import DunnIndex
from sklearn.metrics import (
    davies_bouldin_score,
    mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)

# Logs and warnings
from cuml.common import logger as cuml_logger
cuml_logger.set_level(cuml_logger.level_error)  # does that work?
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class EligibilityCriterionData:
    raw_text: str
    raw_embedding: list[float]  # length = 784 (or raw_dim, depends on model)
    reduced_embedding: list[float]  # length = 2 (or reduced_dim)


@dataclass
class ClusterInstance:
    id: int
    title: str
    n_samples: int
    prevalence: float
    medoid: list[float]  # length = 2 (or reduced_dim)
    ec_list: list[EligibilityCriterionData]
    
    def __init__(
        self,
        id: int,
        title: str,
        prevalence: float,
        medoid: np.ndarray,
        raw_txts: list[str],
        raw_data: np.ndarray,
        reduced_data: np.ndarray,
    ) -> None:
        """ Build a cluster instance as requested by RisKlick
        """
        self.id = id
        self.title = title
        self.n_samples = len(raw_txts)
        self.prevalence = prevalence
        self.medoid = medoid
        self.ec_list = [
            EligibilityCriterionData(
                raw_text=raw_text,
                raw_embedding=raw_embedding,
                reduced_embedding=reduced_embedding,
            )
            for raw_text, raw_embedding, reduced_embedding,
            in zip(raw_txts, raw_data.tolist(), reduced_data.tolist())
        ]
        
        
@dataclass
class ClusterOutput:
    cluster_plot_path: str
    cluster_metrics: dict[str, float]
    cluster_instances: dict[int, ClusterInstance]
    
    def __init__(
        self,
        token_info: dict,
        cluster_plot_path: str,
        cluster_metrics: dict[str, dict[str, float]],
        cluster_titles: dict[int, str],
        cluster_prevalences: dict[int, float],
        cluster_medoids: dict[int, np.ndarray],
        cluster_sorted_member_ids: dict[int, list[int]],
    ) -> None:
        """ Build output of the clustering algorithm as requested by RisKlick
        """
        self.cluster_plot_path = cluster_plot_path
        self.cluster_metrics = cluster_metrics
        self.cluster_instances = self.get_cluster_instances(
            token_info=token_info,
            cluster_titles=cluster_titles,
            cluster_prevalences=cluster_prevalences,
            cluster_medoids=cluster_medoids,
            cluster_sorted_member_ids=cluster_sorted_member_ids,
        )
    
    @staticmethod
    def get_cluster_instances(
        token_info: dict,
        cluster_titles: dict[int, str],
        cluster_prevalences: dict[int, float],
        cluster_medoids: dict[int, np.ndarray],
        cluster_sorted_member_ids: dict[int, list[int]],
    ) -> list[ClusterInstance]:
        """ Separate data by cluster and build a formatted cluster instance for each
        """
        cluster_instances = {}
        for cluster_id, member_ids in cluster_sorted_member_ids.items():
            
            cluster_instances[cluster_id] = ClusterInstance(
                id=cluster_id,
                title=cluster_titles[cluster_id],
                prevalence=cluster_prevalences[cluster_id],
                medoid=cluster_medoids[cluster_id].tolist(),
                raw_txts=[token_info["raw_txts"][i] for i in member_ids],
                raw_data=token_info["raw_data"][member_ids],
                reduced_data=token_info["plot_data"][member_ids],
            )
            
        return cluster_instances
    
    
@cfg.clean_memory()
def report_clusters(
    model_type: str,
    raw_data: torch.Tensor,
    raw_txts: list[str],
    metadatas: list[str],
    cluster_summarization_params: dict[int, Union[int, str]] | None,
    label_type: str="status",  # status, phase, condition, intervention, label
) -> tuple[str, str, dict]:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    # Reduce data dimensionality
    logging.info(" --- Reducing dimensionality of %s criteria embeddings" % len(raw_data))
    plot_data, cluster_data = get_plot_and_cluster_data(raw_data, model_type)
    
    # Build token information
    logging.info(" --- Collecting criteria texts and labels")
    token_info = get_token_info(raw_txts, metadatas, label_type)
    token_info.update({"plot_data": plot_data, "raw_data": raw_data.numpy()})
    
    # Look for best set of hyper-parameters for clustering
    logging.info(" --- Retrieving best clustering hyper-parameters")
    params = find_best_cluster_params(cluster_data, model_type)
    
    # Proceed to clustering with the best set of hyper-parameters
    logging.info(" --- Clustering criteria with best set of hyper-parameters")
    cluster_info = cluster_criteria(params, cluster_data, model_type)
    
    # Plot criteria embeddings and save plot file path
    logging.info(" --- Plotting reduced criteria embeddings and associated clusters")
    cluster_plot_path = plot_clusters(token_info, cluster_info, model_type)
    
    # Perform label-free and label-dependent evaluation of clustering
    logging.info(" --- Evaluating cluster quality")
    cluster_metrics = evaluate_clustering(cluster_info, metadatas)
    
    # Compute useful statistics and metrics
    logging.info(" --- Computing cluster statistics")
    cluster_prevalences, cluster_medoids, cluster_sorted_member_ids =\
        compute_cluster_statistics(token_info, cluster_info)
    
    # Identify one "typical" criterion string for each cluster of criteria
    logging.info(" --- Generating cluster titles")
    sorted_txts_grouped_by_cluster = {
        cluster_id: [token_info["raw_txts"][i] for i in sample_ids]
        for cluster_id, sample_ids in cluster_sorted_member_ids.items()
    }  # each cluster group includes criteria sorted by distance to its medoid
    cluster_titles = summarize_cluster_criteria(
        txts_grouped_by_cluster=sorted_txts_grouped_by_cluster,
        cluster_summarization_params=cluster_summarization_params,
    )
    
    # Generate final output as a dedicated dataclass
    logging.info(" --- Formatting cluster output")
    return ClusterOutput(
        token_info=token_info,
        cluster_plot_path=cluster_plot_path,
        cluster_metrics=cluster_metrics,
        cluster_titles=cluster_titles,
        cluster_prevalences=cluster_prevalences,
        cluster_medoids=cluster_medoids,
        cluster_sorted_member_ids=cluster_sorted_member_ids,
    )


def get_token_info(
    raw_txts: list[str],
    metadatas: list[str],
    label_type: str,
) -> dict:
    """ Wrap-up raw information about text and plotting
    """
    true_lbls = [l[label_type] for l in metadatas]
    token_info = {
        "n_samples": len(raw_txts),
        "raw_txts": raw_txts,
        "true_lbls": true_lbls,
        "unique_lbls": list(set(true_lbls)),
        "paths": [l["path"] for l in metadatas],
    }
    return token_info


@cfg.clean_memory()
def get_plot_and_cluster_data(
    data: torch.Tensor,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute low-dimensional plot and cluster data using dimensionality
        reduction algorithm, or load already computed results 
    """
    # Retrieve save or load paths for cluster data and plot data
    base_name = "embeddings_%s.pkl" % model_type
    load_path_cluster = os.path.join(cfg.POSTPROCESSED_DIR, "plotted_%s" % base_name)
    load_path_plot = os.path.join(cfg.POSTPROCESSED_DIR, "reduced_%s" % base_name)
    
    # Load already computed data with reduced dimensionality
    if cfg.LOAD_REDUCED_EMBEDDINGS:
        logging.info(" ----- Loading reduced data from previous run")
        with open(load_path_cluster, "rb") as f_cluster:
            cluster_data = pickle.load(f_cluster)
        with open(load_path_plot, "rb") as f_plot:
            plot_data = pickle.load(f_plot)
    
    # Compute reduced dimensionality representation of data and save it
    else:
        
        # Compute reduced representation for clustering
        logging.info(" ----- Running %s algorithm" % cfg.CLUSTER_DIM_RED_ALGO)
        cluster_data = compute_reduced_repr(
            data.numpy(),  # data is torch tensor
            reduced_dim=cfg.CLUSTER_RED_DIM,
            algorithm=cfg.CLUSTER_DIM_RED_ALGO,
        )
        
        # Compute reduced representation for plotting
        if cfg.CLUSTER_RED_DIM == cfg.PLOT_RED_DIM\
        and cfg.CLUSTER_DIM_RED_ALGO == cfg.PLOT_DIM_RED_ALGO:
            plot_data = cluster_data
        else:
            plot_data = compute_reduced_repr(
                cluster_data,  # data.numpy()?
                reduced_dim=cfg.PLOT_RED_DIM,
                algorithm=cfg.PLOT_DIM_RED_ALGO,
            )
        
        # Save data for potential later usage
        with open(load_path_cluster, "wb") as f_cluster:
            pickle.dump(cluster_data, f_cluster)
        with open(load_path_plot, "wb") as f_plot:
            pickle.dump(plot_data, f_plot)
    
    # Return the results
    return plot_data, cluster_data


def compute_reduced_repr(
    data: np.ndarray,
    reduced_dim: int,
    algorithm: str,
    compute_rdm: bool=False,
) -> np.ndarray:
    """ Reduce the dimensionality of high-dimensional concept embeddings
    """
    # No dimensionality reduction
    if reduced_dim == None or algorithm == None:
        return data
    
    # Represent data based on sample similarity instead of data
    if compute_rdm:
        original_dim = data.shape[1]
        data = squareform(pdist(data, "correlation").get())
        params = {"n_components": original_dim}
        pca = PCA(**params)
        data = pca.fit_transform(data)
        
    # Simple PCA algorithm
    if algorithm == "pca":
        params = {"n_components": reduced_dim}
        pca = PCA(**params)
        return pca.fit_transform(data)
    
    # More computationally costly t-SNE algorithm
    elif algorithm == "tsne":
        params = {
            "n_components": reduced_dim,
            "method": "barnes_hut" if reduced_dim < 4 else "exact",
            "n_iter": cfg.N_ITER_MAX_TSNE,
            "n_iter_without_progress": 1000,
            "metric": "cosine",
            "learning_rate": 200.0,
            "verbose": cuml_logger.level_error,
        }
        if data.shape[0] < 36_000:
            n_neighbors = min(int(data.shape[0] / 400 + 1), 90)
            cuml_specific_params = {
                "n_neighbors": n_neighbors,  # CannyLabs CUDA-TSNE default is 32
                "perplexity": 50.0,  # n_neighbors / 3,  # CannyLabs CUDA-TSNE default is 50
                "learning_rate_method": "none",  # not in sklearn and produces bad results
            }
            params.update(cuml_specific_params)
        tsne = TSNE(**params)
        return tsne.fit_transform(data)


@cfg.clean_memory()
def cluster_criteria(
    params: dict,
    data: np.ndarray,
    model_type: str
) -> dict:
    """ Cluster criteria and save the results, or load cluster information from
        a previous run, with specific model embeddings 
    """
    # Load already computed clustering results
    load_path = os.path.join(cfg.POSTPROCESSED_DIR, "cluster_info_%s.pkl" % model_type)
    if cfg.LOAD_CLUSTER_INFO:
        with open(load_path, "rb") as f:
            cluster_info = pickle.load(f)
    
    # Run clustering algorithm with selected hyper-parameters and save results
    else:
        cluster_info = clusterize(data=data, mode="primary", params=params)
        if cluster_info is not None and cfg.DO_SUBCLUSTERIZE:
            cluster_info = subclusterize(cluster_info, params=params)
        with open(load_path, "wb") as f:
            pickle.dump(cluster_info, f)
    
    # Return clustering results
    return cluster_info


@cfg.clean_memory()
def find_best_cluster_params(data: np.ndarray, model_type: str) -> dict:
    """ Use optuna to determine best set of cluster parameters (or load them)
    """
    # Try to load best hyper-parameters and return defaults otherwise
    params_path = os.path.join(cfg.RESULT_DIR, "params_%s.json" % model_type)
    if cfg.LOAD_OPTUNA_RESULTS:
        logging.info(" ----- Loading best hyper-parameters from previous optuna study")
        try:
            with open(params_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logging.warning(" ----- Parameters not found, using default parameters")
            return cfg.DEFAULT_CLUSTERING_PARAMS
        
    # Find and save best hyper-parameters
    else:
        logging.info(" ----- Running optuna study to find best hyper-parameters")
        with LocalCUDACluster(
            n_workers=cfg.NUM_OPTUNA_WORKERS,
            threads_per_worker=cfg.NUM_OPTUNA_THREADS,
            processes=True,
        ) as cluster:
            logging.info(" ----- %s created for study" % cluster)
            with Client(cluster, timeout='120s') as client:
                db_path = "sqlite:///%s/optuna_%s.db" % (cfg.RESULT_DIR, model_type)
                study = optuna.create_study(
                    sampler=TPESampler(),
                    direction="maximize",
                    storage=db_path,
                )
                objective = lambda trial: objective_fn(trial, data)
                study.optimize(
                    func=objective,
                    n_trials=cfg.N_OPTUNA_TRIALS,
                    show_progress_bar=True,
                )
                best_params = study.best_params
        
        logging.info(" ----- Best hyper-parameters: %s" % best_params)
        with open(params_path, "w") as file:
            json.dump(best_params, file, indent=4)
        return best_params


def objective_fn(trial: optuna.Trial, data: np.ndarray) -> float:
    """ Suggest a new set of hyper-parameters and compute associated metric
    """
    # Perform clustering with a new set of suggested parameters
    params = suggest_parameters(trial)  # if isinstance(data, FutureArrayClass): data = data.result()
    try:
        cluster_info = clusterize(data=data, mode="primary", params=params)
        if cluster_info['n_clusters'] > 1 and cfg.DO_SUBCLUSTERIZE:
            cluster_info = subclusterize(cluster_info, params=params)
    except Exception:
        logging.error("Error during clustering. Skipping to next trial.")
        return float("-inf")
    
    # Compute metric with clustering results
    if 1 < cluster_info["n_clusters"] < cfg.N_CLUSTER_MAX:
        cluster_lbls = cluster_info["cluster_lbls"]
        metric_1 = silhouette_score(data, cluster_lbls, chunksize=20_000)
        metric_2 = 1.0 - np.count_nonzero(cluster_lbls == -1) / len(cluster_lbls)
        return metric_1 + metric_2
    else:
        return float("-inf")
    
    
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
    # Identify cluster parameters given the data and cluster mode
    cluster_params = set_cluster_params(len(data), mode, params)
    
    # Find cluster affordances based on cluster hierarchy
    clusterer = hdbscan.HDBSCAN(**cluster_params)
    clusterer.fit(data)
    lbls = clusterer.labels_
    n_clusters = np.max(lbls).item() + 1  # -1 being ignored
    member_ids = {k: np.where(lbls == k)[0].tolist() for k in range(-1, n_clusters)}
        
    # Put back unclustered samples if secondary mode
    if mode == "secondary":
        lbls[np.where(lbls == -1)] = 0
        
    # Return cluster info
    return {
        "clusterer": clusterer,
        "n_clusters": n_clusters,
        "cluster_data": data,
        "cluster_lbls": lbls,
        "cluster_member_ids": member_ids,
    }
    
    
def subclusterize(cluster_info: dict, params: dict) -> dict:
    """ Update cluster results by trying to subcluster any computed cluster
    """
    # Rank cluster ids by cluster size
    cluster_lengths = {k: len(v) for k, v in cluster_info["cluster_member_ids"].items()}
    sorted_lengths = sorted(cluster_lengths.items(), key=lambda x: x[1], reverse=True)
    cluster_ranking = {k: rank for rank, (k, _) in enumerate(sorted_lengths, start=1)}
    threshold = int(np.ceil(cluster_info["n_clusters"] * 0.1))  # 10% largest clusters
    large_cluster_ids = [k for k, rank in cluster_ranking.items() if rank <= threshold]
    if -1 in large_cluster_ids: large_cluster_ids.remove(-1)
    
    # For large clusters, try to cluster it further with new hdbscan parameters
    for cluster_id in large_cluster_ids:
        subset_ids = np.where(cluster_info["cluster_lbls"] == cluster_id)[0]
        subset_data = cluster_info["cluster_data"][subset_ids]
        new_cluster_info = clusterize(data=subset_data, mode="secondary", params=params)
        
        # If the sub-clustering is successful, record new information
        if new_cluster_info["n_clusters"] > 1:  # new_cluster_info is not None:
            n_new_clusters = new_cluster_info["n_clusters"]
            new_cluster_ids =\
                [cluster_id] +\
                [cluster_info["n_clusters"] + i for i in range(n_new_clusters - 1)]
            cluster_info["n_clusters"] += n_new_clusters - 1
            
            # And update cluster labels and cluster member ids
            for i, new_cluster_id in enumerate(new_cluster_ids):
                new_member_ids = new_cluster_info["cluster_member_ids"][i]
                new_member_ids = subset_ids[new_member_ids]  # in original clustering
                cluster_info["cluster_lbls"][new_member_ids] = new_cluster_id
                cluster_info["cluster_member_ids"][new_cluster_id] = new_member_ids
    
    # Return updated cluster info
    return cluster_info


@cfg.clean_memory()
def compute_cluster_statistics(token_info, cluster_info):
    """ Compute CT prevalence between clusters & label prevalence within clusters
    """
    # Put back cluster labels to a list (because they are in a cupy array)
    cluster_lbls = cluster_info["cluster_lbls"].tolist()
    
    # Compute absolute cluster prevalence by counting clinical trials
    zipped_paths = list(zip(token_info["paths"], cluster_lbls))
    cluster_sample_paths = {
        cluster_id: [p for p, l in zipped_paths if l == cluster_id]
        for cluster_id in range(-1, cluster_info["n_clusters"])
    }
    n_cts = len(set(token_info["paths"]))
    cluster_prevalences = {
        # cluster_id: len(paths) / token_info["paths"]
        cluster_id: len(set(paths)) / n_cts
        for cluster_id, paths in cluster_sample_paths.items()
    }
    
    # Sort cluster member ids by how close each member is to its cluster medoid
    cluster_medoids = compute_cluster_medoids(cluster_info)
    cluster_data = cluster_info["cluster_data"]
    cluster_member_ids = cluster_info["cluster_member_ids"]
    cluster_sorted_member_ids = {}
    for k, member_ids in cluster_member_ids.items():
        medoid = cluster_medoids[k]
        members_data = cluster_data[member_ids]
        distances = cdist(medoid[np.newaxis, :], members_data)[0]
        sorted_indices = np.argsort(np.nan_to_num(distances).flatten())
        cluster_sorted_member_ids[k] = [
            member_ids[idx] for idx in sorted_indices.get()
        ]
    
    # Return computed statistics
    return cluster_prevalences, cluster_medoids, cluster_sorted_member_ids


def compute_cluster_medoids(cluster_info: dict) -> dict:
    """ Compute cluster centroids by averaging samples within each cluster,
        weighting by the sample probability of being in that cluster
    """
    cluster_medoids = {}
    for label in range(-1, cluster_info["n_clusters"]):  # including unassigned
        cluster_ids = np.where(cluster_info["cluster_lbls"] == label)[0]
        cluster_data = cluster_info["cluster_data"][cluster_ids]
        cluster_medoids[label] = compute_medoid(cluster_data)
        
    return cluster_medoids


def compute_medoid(data: np.ndarray) -> np.ndarray:
    """ Compute medoids of a subset of samples of shape (n_samples, n_features)
        Distance computations are made with dask to mitigate memory requirements
    """
    dask_data = da.from_array(data, chunks=1_000)
    def compute_distances(chunk): return cdist(chunk, data)
    distances = dask_data.map_blocks(compute_distances, dtype=float)
    sum_distances = distances.sum(axis=1).compute()
    medoid_index = sum_distances.argmin().get()
    return data[medoid_index]


def summarize_cluster_criteria(
    txts_grouped_by_cluster: dict[int, list[str]],
    cluster_summarization_params: dict[str, Union[int, str]] | None,
) -> dict[int, str]:
    """ For each cluster, summarize all belonging criteria to a single sentence
    """
    # Take default clusterization parameters if not provided
    if cluster_summarization_params is None:
        cluster_summarization_params = cfg.DEFAULT_CLUSTER_SUMMARIZATION_PARAMS
    
    # Load N closest to medoid representants for each clusters
    n_representants = cluster_summarization_params["n_representants"]
    cluster_txts = {
        cluster_id: txts[:n_representants]
        for cluster_id, txts in txts_grouped_by_cluster.items()
    }
    cluster_txts[-1] = ["criterion with undefined cluster"] * n_representants
    
    # Take the criterion closest to the cluster medoid
    if cluster_summarization_params["method"] == "closest":
        cluster_titles = {
            cluster_id: txts[0] for cluster_id, txts in cluster_txts.items()
        }
    
    # Take shortest from the 10 criteria closest to the cluster medoid
    elif cluster_summarization_params["method"] == "shortest":
        cluster_titles = {
            cluster_id: min(txts, key=len)
            for cluster_id, txts in cluster_txts.items()
        }
    
    # Use GPT-3.5 to summarize the 10 criteria closest to the cluster medoid
    elif cluster_summarization_params["method"] == "gpt":
        # Authentificate with a valid api-key
        api_path = os.path.join("data", "api-key-risklick.txt")
        try:
            with open(api_path, "r") as f: api_key = f.read()
        except:
            raise FileNotFoundError("You must have an api-key at %s" % api_path)
        client = openai.OpenAI(api_key=api_key)
        
        # Prompt the model and collect answer for each criterion
        cluster_titles = {}
        prompt_loop = tqdm(cluster_txts.items(), "Prompting GPT to summarize clusters")
        for cluster_id, cluster_criteria in prompt_loop:
            user_prompt = \
                cluster_summarization_params["gpt_user_prompt_intro"] + \
                "\n".join(cluster_criteria)
            response = prompt_gpt(
                client=client,
                system_prompt=cluster_summarization_params["gpt_system_prompt"],
                user_prompt=user_prompt,
            )
            post_processed = "criterion - ".join(
                [s.capitalize() for s in response.split("criterion - ")]
            )
            cluster_titles[cluster_id] = post_processed.replace("\n", " ")
            
    # Handle wrong method name
    else:
        raise ValueError("Wrong pooling method selected.")
    
    # Return generated cluster titles
    return cluster_titles


def prompt_gpt(
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    max_retries: int=5
) -> str:
    """ Collect answer of chat-gpt, given system and user prompts, and avoiding
        rate limit by waiting an exponentially increasing amount of time
    """
    for i in range(max_retries):  # hard limit
        try:
            # response = openai.ChatCompletion.create(
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except (RateLimitError, ConnectionError, OpenAIError, APIError,
                APIStatusError, APITimeoutError, APIConnectionError) as e:
            logging.error("An error occurred: %s. Retrying in %i seconds." % (e, 2 ** i))
            time.sleep(2 ** i)
            
    return "No response, open-ai reached rate limit or another network issue occurred."


def plot_clusters(
    token_info: dict,
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
    true_lbls = token_info["true_lbls"]
    cluster_lbls = cluster_info["cluster_lbls"]
    n_labels = len(set(true_lbls))
    n_clusters = cluster_info["n_clusters"]
    s_factor = (280_000 / len(true_lbls)) ** 0.75
    scatter_params = dict(cfg.SCATTER_PARAMS)
    scatter_params.update({"s": cfg.SCATTER_PARAMS["s"] * s_factor})
    
    # Retrieve relevant data
    cluster_colors, class_colors = get_cluster_colors(true_lbls, cluster_lbls)
    plot_data = token_info["plot_data"]
    if subset_ids is not None:  # take subset for secondary cluster displays
        plot_data = plot_data[subset_ids]
        class_colors = [class_colors[i] for i in subset_ids[0]]
        
    # Visualize empirical clusters
    ax1 = fig.add_subplot(2, 2, 1, **plot_kwargs)
    ax1.scatter(*plot_data.T, c=cluster_colors, **scatter_params)
    ax1.set_xticklabels([]); ax1.set_yticklabels([])
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Data and %s clusters" % n_clusters, fontsize=cfg.TEXT_SIZE)
    
    # Visualize empirical clusters (but only the samples with an assigned cluster)
    cluster_only_indices = np.where(cluster_lbls > -1)[0]
    clustered_data = plot_data[cluster_only_indices]
    clustered_colors = [cluster_colors[i] for i in cluster_only_indices]    
    ax2 = fig.add_subplot(2, 2, 2, **plot_kwargs)
    ax2.scatter(*clustered_data.T, c=clustered_colors, **scatter_params)
    ax2.set_xticklabels([]); ax2.set_yticklabels([])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Data and %s assigned clusters" % (n_clusters), fontsize=cfg.TEXT_SIZE)
    
    # Visualize theoretical clusters (using, e.g., ICD10-CM code hierarchy)
    ax3 = fig.add_subplot(2, 2, 3, **plot_kwargs)
    ax3.scatter(*plot_data.T, c=class_colors, **scatter_params)
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
    
    # Save figure and return path to the figure
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600)
    plt.close()
    return fig_path


def get_cluster_colors(true_lbls, cluster_lbls):
    """ Find the best match between clusters and labels to assigns colors
    """
    # Identify possible colour values
    cluster_lbls = cluster_lbls.tolist()
    unique_classes = list(set(true_lbls))
    unique_clusters = list(set(cluster_lbls))
    
    # In this case, true labels define the maximum number of colours
    if len(unique_classes) >= len(unique_clusters):
        color_map = best_color_match(
            cluster_lbls, true_lbls, unique_clusters, unique_classes)
        color_map = {k: unique_classes.index(v) for k, v in color_map.items()}
        cluster_colors = [
            cfg.COLORS[color_map[i]] if i != -1
            else cfg.NA_COLOR for i in cluster_lbls
        ]
        class_colors = [
            cfg.COLORS[unique_classes.index(l)] if l != -1
            else cfg.NA_COLOR for l in true_lbls
        ]
    
    # In this case, empirical clusters define the maximum number of colours
    else:
        color_map = best_color_match(
            true_lbls, cluster_lbls, unique_classes, unique_clusters)
        color_map = {unique_classes.index(k): v for k, v in color_map.items()}
        cluster_colors = [
            cfg.COLORS[i] if i >= 0
            else cfg.NA_COLOR for i in cluster_lbls
        ]
        class_colors = [
            cfg.COLORS[color_map[unique_classes.index(l)]] if l != -1
            else cfg.NA_COLOR for l in true_lbls
        ]
        
    # Return aligned empirical and theorical clusters
    return cluster_colors, class_colors


def best_color_match(src_lbls, tgt_lbls, unique_src_lbls, unique_tgt_lbls):
    """ Find the best match between subcategories, based on cluster memberships
    """
    cost_matrix = np.zeros((len(unique_src_lbls), len(unique_tgt_lbls)))
    for i, src_lbl in enumerate(unique_src_lbls):
        for j, tgt_lbl in enumerate(unique_tgt_lbls):
            count = sum(
                s == src_lbl and t == tgt_lbl for s, t in zip(src_lbls, tgt_lbls)
            )
            cost_matrix[i, j] = -count
    
    rows, cols = linear_sum_assignment(cost_matrix)
    return {unique_src_lbls[i]: unique_tgt_lbls[j] for i, j in zip(rows, cols)}


@cfg.clean_memory()
def evaluate_clustering(cluster_info, metadata):
    """ Run final evaluation of clusters, based on phase(s), condition(s), and
        interventions(s). Duplicate each samples for any combination.    
    """
    # Get relevant data
    cluster_metrics = {}
    cluster_data = cluster_info["cluster_data"]
    cluster_lbls = cluster_info["cluster_lbls"]
    phase_lbls = [l["phase"] for l in metadata]
    cond_lbls = [l["condition"] for l in metadata]
    itrv_lbls = [l["intervention"] for l in metadata]
    
    # Evaluate clustering quality (label-free)
    sil_score = silhouette_score(cluster_data, cluster_lbls, chunksize=20_000)
    cluster_metrics["label_free"] = {
        "Silhouette score": sil_score,
        "DB index": davies_bouldin_score(cluster_data, cluster_lbls),
        "Dunn index": dunn_index(cluster_data, cluster_lbls),
    }
    
    # Create a new sample for each [phase, cond, itrv] label combination
    cluster_lbls = cluster_lbls.tolist()
    dupl_cluster_lbls = []
    dupl_true_lbls = []
    for cluster_lbl, phases, conds, itrvs in\
        zip(cluster_lbls, phase_lbls, cond_lbls, itrv_lbls):
        true_lbl_combinations = list(product(phases, conds, itrvs))
        for true_lbl_combination in true_lbl_combinations:
            dupl_cluster_lbls.append(cluster_lbl)
            dupl_true_lbls.append(" - ".join(true_lbl_combination))
    
    # Create a set of int labels for label-dependent metrics
    encoder = LabelEncoder()
    true_lbls = encoder.fit_transform(dupl_true_lbls).astype(np.int32)
    pred_lbls = np.array(dupl_cluster_lbls, dtype=np.int32)
    
    # Evaluate clustering quality (label-dependent)
    homogeneity, completeness, v_measure = \
        homogeneity_completeness_v_measure(true_lbls, pred_lbls)
    cluster_metrics["label_dept"] = {
        "Homogeneity": homogeneity,
        "Completeness": completeness,
        "V measure": v_measure,
        "MI score": mutual_info_score(true_lbls, pred_lbls),
        "AMI score": adjusted_mutual_info_score(true_lbls, pred_lbls),
        "AR score": adjusted_rand_score(true_lbls, pred_lbls),
    }
    
    # Return all metrics and some info
    cluster_metrics["n_samples"] = len(cluster_lbls)
    cluster_metrics["n_duplicated_samples"] = len(dupl_cluster_lbls)
    return cluster_metrics


def dunn_index(cluster_data: np.ndarray, cluster_lbls: np.ndarray) -> float:
    """ Compute Dunn index using torch-metrics
    """
    dunn = DunnIndex(p=2)
    metric = dunn(
        torch.as_tensor(cluster_data, device="cuda"),
        torch.as_tensor(cluster_lbls, device="cuda"),
    )
    return metric.item()
