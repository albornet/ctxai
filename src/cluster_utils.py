# Mains
import os
import time
import json
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
try:
    from . import config as cfg
except ImportError:
    import config as cfg

# Utils
from scipy.spatial.distance import squareform
from cupyx.scipy.spatial.distance import cdist, pdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from itertools import product
from typing import Union
from tqdm import tqdm
from dataclasses import dataclass, asdict

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

# Logs and warnings (does any of this works?)
from cuml.common import logger as cuml_logger
cuml_logger.set_level(cuml_logger.level_error)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


@dataclass
class EligibilityCriterionData:
    ct_id: str
    raw_text: str
    reduced_embedding: list[float]  # length = 2 (or reduced_dim)
    raw_embedding: list[float] | None=None  # length = 784 (or raw_dim, depends on model)


@dataclass
class ClusterInstance:
    cluster_id: int
    title: str
    n_samples: int
    prevalence: float
    medoid: list[float]  # length = 2 (or reduced_dim)
    ec_list: list[EligibilityCriterionData]  # length = n_samples
    
    def __init__(
        self,
        cluster_id: int,
        ct_ids: list[str],
        title: str,
        prevalence: float,
        medoid: np.ndarray,
        raw_txts: list[str],
        raw_data: np.ndarray,
        reduced_data: np.ndarray,
    ) -> None:
        """ Build a cluster instance as requested by RisKlick
        """
        self.cluster_id = cluster_id
        self.title = title
        self.n_samples = len(raw_txts)
        self.prevalence = prevalence
        self.medoid = medoid
        self.ec_list = [
            EligibilityCriterionData(
                ct_id=ct_id,
                raw_text=raw_text,
                reduced_embedding=reduced_embedding,
                # raw_embedding=raw_embedding,  # not used by RisKlick and heavy
            )
            for ct_id, raw_text, raw_embedding, reduced_embedding,
            in zip(ct_ids, raw_txts, raw_data.tolist(), reduced_data.tolist())
        ]
        
        
@dataclass
class ClusterOutput:
    model_id: str
    user_id: str | None
    project_id: str | None
    unique_id: str
    cluster_metrics: dict[str, float]
    cluster_instances: list[ClusterInstance]
    plot_path: str
    json_path: str
    
    def __init__(
        self,
        token_info: dict,
        cluster_info: dict,
        user_id: str | None=None,
        project_id: str | None=None,
    ) -> None:
        """ Build output of the clustering algorithm as requested by RisKlick
        """
        self.model_id = token_info["model_id"]
        self.project_id = project_id
        self.user_id = user_id
        self.unique_id = "user-%s_project-%s_model-%s" % \
            (self.user_id, self.project_id, self.model_id)
        self.cluster_metrics = cluster_info["metrics"]
        self.cluster_instances = self.get_cluster_instances(
            token_info=token_info, cluster_info=cluster_info,
        )
        self.plot_clusters()  # sets self.plot_path
        self.write_to_json()  # sets self.json_path
    
    @staticmethod
    def get_cluster_instances(
        token_info: dict,
        cluster_info: dict,
    ) -> list[ClusterInstance]:
        """ Separate data by cluster and build a formatted cluster instance for each
        """
        cluster_instances = []
        for cluster_id, member_ids in cluster_info["sorted_member_ids"].items():
            
            cluster_instances.append(
                ClusterInstance(
                    cluster_id=cluster_id,
                    ct_ids=[token_info["ct_ids"][i] for i in member_ids],
                    title=cluster_info["titles"][cluster_id],
                    prevalence=cluster_info["prevalences"][cluster_id],
                    medoid=cluster_info["medoids"][cluster_id].tolist(),
                    raw_txts=[token_info["raw_txts"][i] for i in member_ids],
                    raw_data=token_info["raw_data"][member_ids],
                    reduced_data=token_info["plot_data"][member_ids],
                )
            )
            
        return cluster_instances
    
    def plot_clusters(self, font_size: float=21.0) -> None:
        """ Plot the top 20 instances of empirical clusters (skipping undefined cluster)
        """
        # Retrieve top-20 data and titles and format them for plotly.scatter
        defined_cluster = [c for c in self.cluster_instances if c.cluster_id != -1]
        top_20_clusters = sorted(defined_cluster, key=lambda x: x.n_samples, reverse=True)[:20]
        ys, xs, titles = [], [], []
        legend_line_count = 0
        for cluster in top_20_clusters:
            title, title_line_count = self.format_title_for_legend(cluster.title)
            legend_line_count += title_line_count
            for ec in cluster.ec_list:
                xs.append(ec.reduced_embedding[0])
                ys.append(ec.reduced_embedding[1])
                titles.append(title)
        plot_df = pd.DataFrame({"x": xs, "y": ys, "title": titles})
        
        # Plot cluster data
        fig = go.Figure()
        for k, title in enumerate(plot_df["title"].unique()):
            cluster_df = plot_df[plot_df["title"] == title]
            symbol = ["circle", "x"][int(k >= 10)]
            fig.add_scatter(
                x=cluster_df["x"], y=cluster_df["y"], name=title,
                opacity=0.8, mode="markers", marker_symbol=symbol,
            )
        
        # Polish figure
        fig.update_traces(marker=dict(
            size=font_size / 2,
            line=dict(color="black", width=1)
        ))
        fig.update_xaxes(
            title_text="tSNE-1", linecolor="black", linewidth=0.5,
            title_font=dict(size=font_size, family="TeX Gyre Pagella"),
            tickfont=dict(size=font_size, family="TeX Gyre Pagella")
        )
        fig.update_yaxes(
            title_text="tSNE-2", linecolor="black", linewidth=0.5,
            title_font=dict(size=font_size, family="TeX Gyre Pagella"),
            tickfont=dict(size=font_size, family="TeX Gyre Pagella")
        )
        width, height = 1440, 720
        legend_font_size = max(1, font_size * 20 / legend_line_count)
        fig.update_layout(
            legend=dict(
                yanchor="auto", xanchor="left", title_text="",
                font=dict(size=legend_font_size, family="TeX Gyre Pagella"),
            ),
            width=width, height=height, plot_bgcolor="white",
            margin=dict(l=60, r=width * 0.45, t=30, b=80),
        )
        
        # Save image and sets plot_path
        plot_name = "%s_cluster_plot.png" % self.unique_id
        plot_path = os.path.join(cfg.RESULT_DIR, plot_name)
        fig.write_image(plot_path, engine="kaleido",scale=2)
        self.plot_path = plot_path
    
    @staticmethod
    def format_title_for_legend(
        title: str,
        max_length: int=75,
        max_line_count: int=2
    ) -> tuple[str, int]:
        """ Try to format the title in the legend of the ctxai cluster plot
        """
        # Shorten criterion type information
        title = title.replace("\n", " ").replace("<br>", " ") \
            .replace("Inclusion -", "IN -").replace("Inclusion criterion", "IN") \
            .replace("inclusion -", "IN -").replace("inclusion criterion", "IN") \
            .replace("Exclusion -", "EX -").replace("Exclusion criterion", "EX") \
            .replace("exclusion -", "EX -").replace("exclusion criterion", "EX")
        
        # Let the title as it is if its length is ok
        if len(title) <= max_length:
            return title.strip(), 1
        
        # Split the title into words and process
        words = title.split()
        shortened_text = ""
        current_line_length = 0
        line_count = 1
        for word in words:
            
            # Check if adding the next word would exceed the maximum length
            if current_line_length + len(word) > max_length:
                if line_count == max_line_count:  # replace remaining text by "..."
                    shortened_text = shortened_text.rstrip() + "..."
                    break
                else:  # Move to the next line
                    shortened_text += "<br>" + word + " "
                    current_line_length = len(word) + 1
                    line_count += 1
            else:
                shortened_text += word + " "
                current_line_length += len(word) + 1
        
        return shortened_text.strip(), line_count
        
    def write_to_json(self) -> str:
        """ Convert cluster output to a dictionary and write it to a json file,
            after generating a unique file name given by the project and user ids
        """
        # Define file name and sets json_path
        file_name = "%s_ec_clustering.json" % self.unique_id
        json_path = os.path.join(cfg.RESULT_DIR, file_name)
        self.json_path = json_path
        
        # Save data as a json file
        cluster_output_dict = asdict(self)
        json_data = json.dumps(cluster_output_dict, indent=4)
        with open(json_path, 'w') as file:
            file.write(json_data)
            
            
            
@cfg.clean_memory()
def report_clusters(
    model_id: str,
    raw_data: torch.Tensor,
    raw_txts: list[str],
    metadatas: list[str],
    label_type: str="status",  # status, phase, condition, intervention, label
    cluster_summarization_params: dict[int, Union[int, str]] | None=None,
    user_id: str | None=None,
    project_id: str | None=None,
) -> tuple[str, str, dict]:
    """ Reduce the dimensionality of concept embeddings for different categories
        and log a scatter plot of the low-dimensional data to tensorboard
    """
    # Reduce data dimensionality
    logging.info(" --- Reducing dimensionality of %s criteria embeddings" % len(raw_data))
    plot_data, cluster_data = get_plot_and_cluster_data(raw_data, model_id)
    
    # Build token information
    logging.info(" --- Collecting criteria texts, labels, and embeddings")
    token_info = get_token_info(
        raw_txts, raw_data, plot_data, model_id, label_type, metadatas,
    )
    
    # Look for best set of hyper-parameters for clustering
    logging.info(" --- Retrieving best clustering hyper-parameters")
    params = find_best_cluster_params(cluster_data, model_id)
    
    # Proceed to clustering with the best set of hyper-parameters
    logging.info(" --- Clustering criteria with best set of hyper-parameters")
    cluster_info = cluster_criteria(params, cluster_data, model_id)
    
    # Perform label-free and label-dependent evaluation of clustering
    logging.info(" --- Evaluating cluster quality")
    evaluate_clustering(cluster_info, metadatas)
    
    # Compute useful statistics and metrics
    logging.info(" --- Computing cluster statistics")
    compute_cluster_statistics(token_info, cluster_info)
    
    # Identify one "typical" criterion string for each cluster of criteria
    logging.info(" --- Generating cluster titles")
    compute_cluster_titles(token_info, cluster_info, cluster_summarization_params)
    
    # Generate formatted output as a dedicated dataclass
    logging.info(" --- Formatting cluster output and plotting data")
    cluster_output = ClusterOutput(
        token_info=token_info, cluster_info=cluster_info,
        user_id=user_id, project_id=project_id,
    )
        
    # Return final cluster output data structure
    return cluster_output


def get_token_info(
    raw_txts: list[str],
    raw_data: torch.Tensor,
    plot_data: np.ndarray,
    model_id: str,
    label_type: str,
    metadatas: list[str],
) -> dict:
    """ Wrap-up raw information about text and plotting
    """
    true_lbls = [l[label_type] for l in metadatas]
    token_info = {
        "n_samples": len(raw_txts),
        "raw_txts": raw_txts,
        "raw_data": raw_data.numpy(),
        "plot_data": plot_data,
        "model_id": model_id,
        "true_lbls": true_lbls,
        "unique_lbls": list(set(true_lbls)),
        "ct_ids": [l["path"][0] for l in metadatas],
    }
    return token_info


@cfg.clean_memory()
def get_plot_and_cluster_data(
    data: torch.Tensor,
    model_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """ Compute low-dimensional plot and cluster data using dimensionality
        reduction algorithm, or load already computed results 
    """
    # Retrieve save or load paths for cluster data and plot data
    base_name = "embeddings_%s.pkl" % model_id
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
                "perplexity": 50.0,  # CannyLabs CUDA-TSNE default is 50
                "learning_rate_method": "none",  # not in sklearn and produces bad results
            }
            params.update(cuml_specific_params)
        tsne = TSNE(**params)
        return tsne.fit_transform(data)


@cfg.clean_memory()
def cluster_criteria(
    params: dict,
    data: np.ndarray,
    model_id: str
) -> dict:
    """ Cluster criteria and save the results, or load cluster information from
        a previous run, with specific model embeddings 
    """
    # Load already computed clustering results
    load_path = os.path.join(cfg.POSTPROCESSED_DIR, "cluster_info_%s.pkl" % model_id)
    if cfg.LOAD_CLUSTER_RESULTS:
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
def find_best_cluster_params(data: np.ndarray, model_id: str) -> dict:
    """ Use optuna to determine best set of cluster parameters (or load them)
    """
    # Try to load best hyper-parameters and return defaults otherwise
    params_path = os.path.join(cfg.RESULT_DIR, "params_%s.json" % model_id)
    if cfg.LOAD_OPTUNA_RESULTS:
        logging.info(" ----- Loading best hyper-parameters from previous optuna study")
        try:
            with open(params_path, "r") as file:
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
            with Client(cluster, timeout="120s") as client:
                db_path = "sqlite:///%s/optuna_%s.db" % (cfg.RESULT_DIR, model_id)
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
        if cluster_info["n_clusters"] > 1 and cfg.DO_SUBCLUSTERIZE:
            cluster_info = subclusterize(cluster_info, params=params)
    except Exception:
        logging.error("Error during clustering. Skipping to next trial.")
        return float("-inf")
    
    # Compute metric with clustering results
    if 1 < cluster_info["n_clusters"] < cfg.N_CLUSTER_MAX:
        cluster_lbls = cluster_info["cluster_ids"]
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
    cluster_ids = clusterer.labels_
    n_clusters = np.max(cluster_ids).item() + 1  # -1 being ignored
    member_ids = {
        k: np.where(cluster_ids == k)[0].tolist() for k in range(-1, n_clusters)
    }
        
    # Put back unclustered samples if secondary mode
    if mode == "secondary":
        cluster_ids[np.where(cluster_ids == -1)] = 0
        
    # Return cluster info
    return {
        "clusterer": clusterer,
        "n_clusters": n_clusters,
        "cluster_data": data,
        "cluster_ids": cluster_ids,
        "member_ids": member_ids,
    }
    
    
def subclusterize(cluster_info: dict, params: dict) -> dict:
    """ Update cluster results by trying to subcluster any computed cluster
    """
    # Rank cluster ids by cluster size
    cluster_lengths = {k: len(v) for k, v in cluster_info["member_ids"].items()}
    sorted_lengths = sorted(cluster_lengths.items(), key=lambda x: x[1], reverse=True)
    cluster_ranking = {k: rank for rank, (k, _) in enumerate(sorted_lengths, start=1)}
    threshold = int(np.ceil(cluster_info["n_clusters"] * 0.1))  # 10% largest clusters
    large_cluster_ids = [k for k, rank in cluster_ranking.items() if rank <= threshold]
    if -1 in large_cluster_ids: large_cluster_ids.remove(-1)
    
    # For large clusters, try to cluster it further with new hdbscan parameters
    for cluster_id in large_cluster_ids:
        subset_ids = np.where(cluster_info["cluster_ids"] == cluster_id)[0]
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
                new_member_ids = new_cluster_info["member_ids"][i]
                new_member_ids = subset_ids[new_member_ids]  # in original clustering
                cluster_info["cluster_ids"][new_member_ids] = new_cluster_id
                cluster_info["member_ids"][new_cluster_id] = new_member_ids
    
    # Return updated cluster info
    return cluster_info


@cfg.clean_memory()
def compute_cluster_statistics(token_info: dict, cluster_info: dict) -> None:
    """ Compute CT prevalence between clusters & label prevalence within clusters
    """
    # Put back cluster labels to a list (because they are in a cupy array)
    cluster_lbls = cluster_info["cluster_ids"].tolist()
    
    # Compute absolute cluster prevalence by counting clinical trials
    zipped_paths = list(zip(token_info["ct_ids"], cluster_lbls))
    cluster_sample_paths = {
        cluster_id: [p for p, l in zipped_paths if l == cluster_id]
        for cluster_id in range(-1, cluster_info["n_clusters"])
    }
    n_cts = len(set(token_info["ct_ids"]))
    cluster_prevalences = {
        # cluster_id: len(paths) / token_info["ct_ids"]
        cluster_id: len(set(paths)) / n_cts
        for cluster_id, paths in cluster_sample_paths.items()
    }
    
    # Sort cluster member ids by how close each member is to its cluster medoid
    cluster_medoids = compute_cluster_medoids(cluster_info)
    cluster_data = cluster_info["cluster_data"]
    cluster_member_ids = cluster_info["member_ids"]
    cluster_sorted_member_ids = {}
    for k, member_ids in cluster_member_ids.items():
        medoid = cluster_medoids[k]
        members_data = cluster_data[member_ids]
        distances = cdist(medoid[np.newaxis, :], members_data)[0]
        sorted_indices = np.argsort(np.nan_to_num(distances).flatten())
        cluster_sorted_member_ids[k] = [
            member_ids[idx] for idx in sorted_indices.get()
        ]
    
    # Update main cluster info dict with computed statistics
    cluster_info.update({
        "prevalences": cluster_prevalences,
        "medoids": cluster_medoids,
        "sorted_member_ids": cluster_sorted_member_ids,
    })


def compute_cluster_medoids(cluster_info: dict) -> dict:
    """ Compute cluster centroids by averaging samples within each cluster,
        weighting by the sample probability of being in that cluster
    """
    cluster_medoids = {}
    for label in range(-1, cluster_info["n_clusters"]):  # including unassigned
        cluster_ids = np.where(cluster_info["cluster_ids"] == label)[0]
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


@cfg.clean_memory()
def evaluate_clustering(cluster_info: dict, metadatas: dict):
    """ Run final evaluation of clusters, based on phase(s), condition(s), and
        interventions(s). Duplicate each samples for any combination.    
    """
    # Get relevant data
    cluster_metrics = {}
    cluster_data = cluster_info["cluster_data"]
    cluster_lbls = cluster_info["cluster_ids"]
    phase_lbls = [l["phase"] for l in metadatas]
    cond_lbls = [l["condition"] for l in metadatas]
    itrv_lbls = [l["intervention"] for l in metadatas]
    
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
    
    # Update cluster_info main dict with cluster_metrics
    cluster_metrics["n_samples"] = len(cluster_lbls)
    cluster_metrics["n_duplicated_samples"] = len(dupl_cluster_lbls)
    cluster_info.update({"metrics": cluster_metrics})


def dunn_index(cluster_data: np.ndarray, cluster_lbls: np.ndarray) -> float:
    """ Compute Dunn index using torch-metrics
    """
    dunn = DunnIndex(p=2)
    metric = dunn(
        torch.as_tensor(cluster_data, device="cuda"),
        torch.as_tensor(cluster_lbls, device="cuda"),
    )
    return metric.item()


def compute_cluster_titles(token_info: dict,
    cluster_info,
    cluster_summarization_params: dict[str, Union[int, str]] | None=None,
) -> dict[int, str]:
    """ For each cluster, summarize all belonging criteria to a single sentence
    """
    # Compute clusters sorted by distance to the cluster medoid
    sorted_txts_grouped_by_cluster = {
        cluster_id: [token_info["raw_txts"][i] for i in sample_ids]
        for cluster_id, sample_ids in cluster_info["sorted_member_ids"].items()
    }

    # Take default clusterization parameters if not provided
    if cluster_summarization_params is None:
        cluster_summarization_params = cfg.DEFAULT_CLUSTER_SUMMARIZATION_PARAMS
    
    # Load N closest to medoid representants for each clusters
    n_representants = cluster_summarization_params["n_representants"]
    cluster_txts = {
        cluster_id: txts[:n_representants]
        for cluster_id, txts in sorted_txts_grouped_by_cluster.items()
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
    
    # Update main cluster info dict with generated cluster titles
    cluster_info.update({"titles": cluster_titles})


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


# def plot_clusters_original_way(
#     token_info: dict,
#     cluster_info: dict,
#     model_id: str,
#     subset_ids: list[int]=None,
# ) -> None:
#     """ Plot theoretical clusters (using labels), empirical clusters (using
#         hdbscan), and the empirical cluster tree
#     """
#     # Initialize figure and load data
#     fig_path = os.path.join(cfg.RESULT_DIR, "plot_%s.png" % model_id)
#     fig = plt.figure(figsize=cfg.FIG_SIZE)
#     plot_kwargs = {} if cfg.PLOT_RED_DIM <= 2 else {"projection": "3d"}
#     true_lbls = token_info["true_lbls"]
#     cluster_lbls = cluster_info["cluster_ids"]
#     n_labels = len(set(true_lbls))
#     n_clusters = cluster_info["n_clusters"]
#     s_factor = (280_000 / len(true_lbls)) ** 0.75
#     scatter_params = dict(cfg.SCATTER_PARAMS)
#     scatter_params.update({"s": cfg.SCATTER_PARAMS["s"] * s_factor})
    
#     # Retrieve relevant data
#     cluster_colors, class_colors = get_cluster_colors(true_lbls, cluster_lbls)
#     plot_data = token_info["plot_data"]
#     if subset_ids is not None:  # take subset for secondary cluster displays
#         plot_data = plot_data[subset_ids]
#         class_colors = [class_colors[i] for i in subset_ids[0]]
        
#     # Visualize empirical clusters
#     ax1 = fig.add_subplot(2, 2, 1, **plot_kwargs)
#     ax1.scatter(*plot_data.T, c=cluster_colors, **scatter_params)
#     ax1.set_xticklabels([]); ax1.set_yticklabels([])
#     ax1.set_xticks([]); ax1.set_yticks([])
#     ax1.set_title("Data and %s clusters" % n_clusters, fontsize=cfg.TEXT_SIZE)
    
#     # Visualize empirical clusters (but only the samples with an assigned cluster)
#     cluster_only_indices = np.where(cluster_lbls > -1)[0]
#     clustered_data = plot_data[cluster_only_indices]
#     clustered_colors = [cluster_colors[i] for i in cluster_only_indices]    
#     ax2 = fig.add_subplot(2, 2, 2, **plot_kwargs)
#     ax2.scatter(*clustered_data.T, c=clustered_colors, **scatter_params)
#     ax2.set_xticklabels([]); ax2.set_yticklabels([])
#     ax2.set_xticks([]); ax2.set_yticks([])
#     ax2.set_title("Data and %s assigned clusters" % (n_clusters), fontsize=cfg.TEXT_SIZE)
    
#     # Visualize theoretical clusters (using, e.g., ICD10-CM code hierarchy)
#     ax3 = fig.add_subplot(2, 2, 3, **plot_kwargs)
#     ax3.scatter(*plot_data.T, c=class_colors, **scatter_params)
#     ax3.set_xticklabels([]); ax3.set_yticklabels([])
#     ax3.set_xticks([]); ax3.set_yticks([])
#     ax3.set_title("Data and %s labels" % n_labels, fontsize=cfg.TEXT_SIZE)
    
#     # Visualize empirical cluster tree
#     if hasattr(cluster_info["clusterer"], "condensed_tree_"):
#         ax4 = fig.add_subplot(2, 2, 4)
#         cluster_info["clusterer"].condensed_tree_.plot(
#             axis=ax4, leaf_separation=cfg.LEAF_SEPARATION, colorbar=True)
#         ax4.set_title("Empirical cluster tree", fontsize=cfg.TEXT_SIZE)
#         ax4.get_yaxis().set_tick_params(left=False)
#         ax4.get_yaxis().set_tick_params(right=False)
#         ax4.set_ylabel("", fontsize=cfg.TEXT_SIZE)
#         ax4.set_yticks([])
#         ax4.spines["top"].set_visible(True)
#         ax4.spines["right"].set_visible(True)
#         ax4.spines["bottom"].set_visible(True)
    
#     # Save figure and return path to the figure
#     plt.tight_layout()
#     plt.savefig(fig_path, dpi=600)
#     plt.close()
#     return fig_path


# def get_cluster_colors(true_lbls, cluster_lbls):
#     """ Find the best match between clusters and labels to assigns colors
#     """
#     # Identify possible colour values
#     cluster_lbls = cluster_lbls.tolist()
#     unique_classes = list(set(true_lbls))
#     unique_clusters = list(set(cluster_lbls))
    
#     # In this case, true labels define the maximum number of colours
#     if len(unique_classes) >= len(unique_clusters):
#         color_map = best_color_match(
#             cluster_lbls, true_lbls, unique_clusters, unique_classes)
#         color_map = {k: unique_classes.index(v) for k, v in color_map.items()}
#         cluster_colors = [
#             cfg.COLORS[color_map[i]] if i != -1
#             else cfg.NA_COLOR for i in cluster_lbls
#         ]
#         class_colors = [
#             cfg.COLORS[unique_classes.index(l)] if l != -1
#             else cfg.NA_COLOR for l in true_lbls
#         ]
    
#     # In this case, empirical clusters define the maximum number of colours
#     else:
#         color_map = best_color_match(
#             true_lbls, cluster_lbls, unique_classes, unique_clusters)
#         color_map = {unique_classes.index(k): v for k, v in color_map.items()}
#         cluster_colors = [
#             cfg.COLORS[i] if i >= 0
#             else cfg.NA_COLOR for i in cluster_lbls
#         ]
#         class_colors = [
#             cfg.COLORS[color_map[unique_classes.index(l)]] if l != -1
#             else cfg.NA_COLOR for l in true_lbls
#         ]
        
#     # Return aligned empirical and theorical clusters
#     return cluster_colors, class_colors


# def best_color_match(src_lbls, tgt_lbls, unique_src_lbls, unique_tgt_lbls):
#     """ Find the best match between subcategories, based on cluster memberships
#     """
#     cost_matrix = np.zeros((len(unique_src_lbls), len(unique_tgt_lbls)))
#     for i, src_lbl in enumerate(unique_src_lbls):
#         for j, tgt_lbl in enumerate(unique_tgt_lbls):
#             count = sum(
#                 s == src_lbl and t == tgt_lbl for s, t in zip(src_lbls, tgt_lbls)
#             )
#             cost_matrix[i, j] = -count
    
#     rows, cols = linear_sum_assignment(cost_matrix)
#     return {unique_src_lbls[i]: unique_tgt_lbls[j] for i, j in zip(rows, cols)}
