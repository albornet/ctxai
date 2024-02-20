# Config
import os
try:
    import config
except:
    from . import config
cfg = config.get_config()

# Utils
import re
import csv
import json
import numpy as np
import pandas as pd
import plotly.express as px
from itertools import product
from collections import defaultdict
from dataclasses import dataclass, asdict
from bertopic import BERTopic

# Optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.samplers import TPESampler, RandomSampler
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml import HDBSCAN
from cuml.metrics.cluster import silhouette_score

# Evaluation
import torch
import dask.array as da
from sklearn.preprocessing import LabelEncoder
from cupyx.scipy.spatial.distance import cdist
from torchmetrics.clustering import DunnIndex
from sklearn.metrics import (
    davies_bouldin_score,
    mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)

# Dimensionality reduction
from umap import UMAP
from cuml import UMAP as CUML_UMAP
from sklearn.decomposition import PCA
from cuml.decomposition import PCA as CUML_PCA
from sklearn.manifold import TSNE
from cuml.manifold import TSNE as CUML_TSNE
from cuml.common import logger as cuml_logger
from bertopic.dimensionality import BaseDimensionalityReduction


class ClusterGeneration:
    def __init__(
        self,
        do_subclusterize: bool=cfg["DO_SUBCLUSTERIZE"],
        n_cluster_max: int=cfg["N_CLUSTER_MAX"],
        optuna_param_ranges: dict=cfg["OPTUNA_PARAM_RANGES"],
        n_optuna_trials: int=cfg["N_OPTUNA_TRIALS"],
        optuna_sampler: str=cfg["OPTUNA_SAMPLER"],
        random_state: int=cfg["RANDOM_STATE"],
    ):
        """ Initialize clustering model based on HDBSCAN, but including
            subclustering and hyper-parameter tuning
        """
        self.cluster_info = None
        self.do_subclusterize = do_subclusterize
        self.n_cluster_max = n_cluster_max
        self.optuna_param_ranges = optuna_param_ranges
        self.n_optuna_trials = n_optuna_trials
        if optuna_sampler == "tpe":
            self.sampler = TPESampler(seed=random_state)
        else:
            self.sampler = RandomSampler(seed=random_state)
    
    def fit(self, X: np.ndarray) -> dict:
        """ Use optuna to determine best set of cluster parameters
        """
        with LocalCUDACluster(n_workers=1, processes=True) as cluster:
            with Client(cluster, timeout="120s") as client:
                study = optuna.create_study(
                    sampler=self.sampler,
                    direction="maximize"
                )
                objective = lambda trial: self.objective_fn(trial, X)
                study.optimize(
                    func=objective,
                    n_trials=self.n_optuna_trials,
                    show_progress_bar=True,
                )
                
        self.best_hyper_params = study.best_params
        self.labels_ = self.predict(X)  # not sur if I'm doing this right (but it works)
        return self
        
    def predict(self, X: np.ndarray) -> list[int]:
        """ Cluster samples and return cluster labels for each sample
        """
        params = self.best_hyper_params
        self.cluster_info = self.clusterize(data=X, mode="primary", params=params)
        if self.cluster_info is not None and self.do_subclusterize:
            self.cluster_info = self.subclusterize(self.cluster_info, params=params)
        return self.cluster_info["clusterer"].labels_

    def objective_fn(self, trial: optuna.Trial, data: np.ndarray) -> float:
        """ Suggest a new set of hyper-parameters and compute associated metric
        """
        # Perform clustering with a new set of suggested parameters
        params = self.suggest_parameters(trial)
        try:
            cluster_info = self.clusterize(data=data, mode="primary", params=params)
            if cluster_info["n_clusters"] > 1 and self.do_subclusterize:
                cluster_info = self.subclusterize(cluster_info, params=params)
        except Exception:
            return float("-inf")
        
        # Return metric from the clustering results
        if 1 < cluster_info["n_clusters"] < self.n_cluster_max:
            cluster_lbls = cluster_info["cluster_ids"]
            metric = 0.0
            metric += 1.0 * silhouette_score(data, cluster_lbls, chunksize=20_000)
            no_lbl_rate = np.count_nonzero(cluster_lbls == -1) / len(cluster_lbls)
            metric += 1.0 * (1.0 - no_lbl_rate)
            return metric
        else:
            return float("-inf")
    
    def suggest_parameters(self, trial: optuna.Trial) -> dict:
        """ Suggest parameters following configured parameter ranges and types
        """
        params = {}
        for name, choices in self.optuna_param_ranges.items():
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

    def set_cluster_params(self, n_samples: int, mode: str, params: dict) -> dict:
        """ Adapt clustering parameters following dataset size and clustering mode
        """
        # Load base parameters
        max_cluster_size = params["max_cluster_size_%s" % mode]
        min_cluster_size = params["min_cluster_size_%s" % mode]
        min_samples = params["min_samples_%s" % mode]
        cluster_selection_method = params["cluster_selection_method_%s" % mode]
        alpha = params["alpha_%s" % mode]
        
        # Adapter function (int vs float, min and max values)
        default_max_value = n_samples - 1
        def adapt_param_fn(value, min_value, max_value=default_max_value):
            if isinstance(value, float): value = int(n_samples * value)
            return min(max(min_value, value), max_value)
        
        # Return all selected cluster parameters
        return {
            "cluster_selection_method": cluster_selection_method,
            "alpha": alpha,
            "allow_single_cluster": True,
            "max_cluster_size": adapt_param_fn(max_cluster_size, 100),
            "min_cluster_size": adapt_param_fn(min_cluster_size, 10),
            "min_samples": adapt_param_fn(min_samples, 10, 1023),  # 2 ** 10 - 1
        }

    def clusterize(self, data: np.ndarray, mode: str, params: dict) -> dict:
        """ Cluster data points with hdbscan algorithm and return cluster information
        """
        # Identify cluster parameters given the data and cluster mode
        cluster_params = self.set_cluster_params(len(data), mode, params)
        
        # Find cluster affordances based on cluster hierarchy
        clusterer = HDBSCAN(**cluster_params)
        clusterer.fit(data)
        cluster_ids = clusterer.labels_
        n_clusters = np.max(cluster_ids).item() + 1  # -1 not counted in n_clusters
        member_ids = {
            k: np.where(cluster_ids == k)[0].tolist() for k in range(-1, n_clusters)
        }
        
        # Put back unclustered samples if secondary mode
        if mode == "secondary":
            cluster_ids[np.where(cluster_ids == -1)] = 0
            
        # Return cluster info
        return {
            "clusterer": clusterer,
            "cluster_data": data,
            "n_clusters": n_clusters,
            "cluster_ids": cluster_ids,
            "member_ids": member_ids,
        }
     
    def subclusterize(self, cluster_info: dict, params: dict) -> dict:
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
            new_cluster_info = \
                self.clusterize(data=subset_data, mode="secondary", params=params)
            
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
          

@dataclass
class EligibilityCriterionData:
    ct_id: str
    raw_text: str
    reduced_embedding: list[float]  # length = 2 (or plot_dim)


@dataclass
class ClusterInstance:
    cluster_id: int
    title: str
    n_samples: int
    prevalence: float
    medoid: list[float]  # length = 2 (or plot_dim)
    ec_list: list[EligibilityCriterionData]  # length = n_samples
    
    def __init__(
        self,
        cluster_id: int,
        ct_ids: list[str],
        title: str,
        prevalence: float,
        medoid: np.ndarray,
        raw_txts: list[str],
        plot_data: np.ndarray,
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
                reduced_embedding=plot_embedding,
            )
            for ct_id, raw_text, plot_embedding,
            in zip(ct_ids, raw_txts, plot_data.tolist())
        ]
        

@dataclass
class ClusterOutput:
    output_dir: str
    embed_model_id: str
    user_id: str | None
    project_id: str | None
    cluster_metrics: dict[str, float]
    cluster_instances: list[ClusterInstance]
    visualization_paths: dict[str, dict[str, str]]
    raw_ec_list_path: str
    json_path: str
    
    def __init__(
        self,
        output_base_dir: str,
        topic_model: BERTopic,
        raw_txts: list[str],
        metadatas: list[str],
        embed_model_id: str | None=None,
        user_id: str | None=None,
        project_id: str | None=None,
    ):
        """ Initialize a class to evaluate the clusters generated by an instance
            of ClusterGeneration, given a set of labels
        """
        # Identify where results come from are where they are stored
        self.embed_model_id = embed_model_id
        self.user_id = user_id
        self.project_id = project_id
        self.output_dir = os.path.join(
            output_base_dir,
            "user-%s" % user_id,
            "project-%s" % project_id,
            "embed-model-%s" % embed_model_id
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Raw data and labels for each eligibility criterion
        self.raw_txts = raw_txts
        self.n_samples = len(raw_txts)
        self.ct_ids = [m["path"][0] for m in metadatas]
        self.phases = [l["phase"] for l in metadatas]
        self.conds = [l["condition"] for l in metadatas]
        self.itrvs = [l["intervention"] for l in metadatas]
        
        # Cluster data and evaluation
        self.cluster_info = self.get_cluster_info(topic_model)
        self.cluster_titles = self.get_cluster_titles(topic_model)
        self.plot_data = self.get_plot_data()
        self.cluster_metrics = self.evaluate_clustering()        
        self.statistics = self.compute_cluster_statistics()
        self.cluster_instances = self.get_cluster_instances()
        
        # Cluster visualization and reports
        self.visualization_paths = self.plot_clusters() 
        self.raw_ec_list_path = self.write_raw_ec_list()
        self.write_to_json()  # this sets self.json_path
    
    def get_cluster_info(self, topic_model: BERTopic) -> dict:
        """ Re-align cluster ids from ClusterGeneration object to BERTopic topics
            BERTopic sorts topics, and hence disaligns with original cluster ids
        """
        original_cluster_info = topic_model.hdbscan_model.cluster_info
        cluster_ids = np.array(topic_model.topics_)
        n_clusters = cluster_ids.max() + 1
        member_ids = {
            k: np.where(cluster_ids == k)[0].tolist()
            for k in range(-1, n_clusters)
        }
        return {
            "clusterer": original_cluster_info["clusterer"],
            "cluster_data": original_cluster_info["cluster_data"],
            "n_clusters": n_clusters,
            "cluster_ids": cluster_ids,
            "member_ids": member_ids,
        }
    
    def get_cluster_titles(self, topic_model: BERTopic) -> dict:
        """ Format titles from raw BERTopic representation model output
        """
        # Helper function that adapts to diverse representation formats
        def get_title(representation) -> str:
            """ Find cluster title from its bertopic representation
            """
            if isinstance(representation, str):
                return representation
            elif isinstance(representation, (tuple, list)):
                titles = [get_title(r) for r in representation]
                return "-".join([t for t in titles if t])
        
        # Return formatted BERTopic representatinos
        formatted_titles = {
            k: get_title(v)
            for k, v in topic_model.topic_representations_.items()
        }
        return formatted_titles
    
    def get_cluster_instances(self) -> list[ClusterInstance]:
        """ Separate data by cluster and build a formatted cluster instance for each
        """
        cluster_instances = []
        for cluster_id, member_ids in self.statistics["sorted_member_ids"].items():
            
            cluster_instances.append(
                ClusterInstance(
                    cluster_id=cluster_id,
                    ct_ids=[self.ct_ids[i] for i in member_ids],
                    title=self.cluster_titles[cluster_id],
                    prevalence=self.statistics["prevalences"][cluster_id],
                    medoid=self.statistics["medoids"][cluster_id].tolist(),
                    raw_txts=[self.raw_txts[i] for i in member_ids],
                    plot_data=self.plot_data[member_ids],
                )
            )
        
        # Function sorting clusters by number of samples, and with "-1" last
        def custom_sort_key(cluster_instance: ClusterInstance, sort_by: str="size"):
            if cluster_instance.cluster_id == -1:
                return 0  # to go to the last position
            if sort_by == "size":
                return cluster_instance.n_samples
            elif sort_by == "prevalence":
                return cluster_instance.prevalence
        
        # Return cluster instances (sorting helps printing data structure)
        return sorted(cluster_instances, key=custom_sort_key, reverse=True)
    
    def get_plot_data(self) -> np.ndarray:
        """ Compute low-dimensional plot data using t-SNE algorithm, or take it
            directly from the cluster data if it has the right dimension
        """
        if cfg["CLUSTER_RED_DIM"] == cfg["PLOT_RED_DIM"] \
        and cfg["CLUSTER_DIM_RED_ALGO"] == cfg["PLOT_DIM_RED_ALGO"] \
        or cfg["CLUSTER_DIM_RED_ALGO"] is None:
            return self.cluster_info["cluster_data"]
        else:
            dim_red_model = get_dim_red_model(
                cfg["PLOT_DIM_RED_ALGO"],
                cfg["PLOT_RED_DIM"],
                self.n_samples,
            )
            return dim_red_model.fit_transform(self.cluster_info["cluster_data"])
    
    def compute_cluster_statistics(self) -> dict:
        """ Compute CT prevalence between clusters & label prevalence within clusters
        """
        # Match clinical trial ids to cluster ids for all criteria
        cluster_ids = self.cluster_info["cluster_ids"].tolist()
        zipped_paths = list(zip(self.ct_ids, cluster_ids))
        
        # Compute absolute cluster prevalence by counting clinical trials
        n_cts = len(set(self.ct_ids))
        cluster_sample_paths = {
            cluster_id: [p for p, l in zipped_paths if l == cluster_id]
            for cluster_id in range(-1, self.cluster_info["n_clusters"])
        }
        cluster_prevalences = {
            cluster_id: len(set(paths)) / n_cts  # len(paths) / token_info["ct_ids"]
            for cluster_id, paths in cluster_sample_paths.items()
        }
        
        # Sort cluster member ids by how close each member is to its cluster medoid
        cluster_medoids = self.compute_cluster_medoids(self.cluster_info)
        cluster_data = self.cluster_info["cluster_data"]
        cluster_member_ids = self.cluster_info["member_ids"]
        cluster_sorted_member_ids = {}
        for k, member_ids in cluster_member_ids.items():
            medoid = cluster_medoids[k]
            members_data = cluster_data[member_ids]
            distances = cdist(medoid[np.newaxis, :], members_data)[0]
            sorted_indices = np.argsort(np.nan_to_num(distances).flatten())
            cluster_sorted_member_ids[k] = [
                member_ids[idx] for idx in sorted_indices.get()
            ]
        
        return {
            "prevalences": cluster_prevalences,
            "medoids": cluster_medoids,
            "sorted_member_ids": cluster_sorted_member_ids,
        }
    
    def compute_cluster_medoids(self, cluster_info: dict) -> dict:
        """ Compute cluster centroids by averaging samples within each cluster,
            weighting by the sample probability of being in that cluster
        """
        cluster_medoids = {}
        for label in range(-1, cluster_info["n_clusters"]):  # including unassigned
            cluster_ids = np.where(cluster_info["cluster_ids"] == label)[0]
            cluster_data = cluster_info["cluster_data"][cluster_ids]
            cluster_medoids[label] = self.compute_medoid(cluster_data)
            
        return cluster_medoids
    
    @staticmethod    
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
    
    def evaluate_clustering(self):
        """ Run final evaluation of clusters, based on phase(s), condition(s), and
            interventions(s). Duplicate each samples for any combination.    
        """
        # Get relevant data
        cluster_metrics = {}
        cluster_data = self.cluster_info["cluster_data"]
        cluster_lbls = self.cluster_info["cluster_ids"]
        
        # Evaluate clustering quality (label-free)
        sil_score = silhouette_score(cluster_data, cluster_lbls, chunksize=20_000)
        cluster_metrics["label_free"] = {
            "Silhouette score": sil_score,
            "DB index": davies_bouldin_score(cluster_data, cluster_lbls),
            "Dunn index": self.dunn_index(cluster_data, cluster_lbls),
        }
        
        # Create a new sample for each [phase, cond, itrv] label combination
        cluster_lbls = cluster_lbls.tolist()
        dupl_cluster_lbls = []
        dupl_true_lbls = []
        for cluster_lbl, phases, conds, itrvs in\
            zip(cluster_lbls, self.phases, self.conds, self.itrvs):
            true_lbl_combinations = list(product(phases, conds, itrvs))
            for true_lbl_combination in true_lbl_combinations:
                dupl_cluster_lbls.append(cluster_lbl)
                dupl_true_lbls.append("- ".join(true_lbl_combination))
        
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
        
        return cluster_metrics  # cluster_info.update({"metrics": cluster_metrics})
    
    @staticmethod
    def dunn_index(cluster_data: np.ndarray, cluster_lbls: np.ndarray) -> float:
        """ Compute Dunn index using torch-metrics
        """
        dunn = DunnIndex(p=2)
        metric = dunn(
            torch.as_tensor(cluster_data, device="cuda"),
            torch.as_tensor(cluster_lbls, device="cuda"),
        )
        return metric.item()
    
    def plot_clusters(self, font_size: float=21.0, top_k: int=20) -> None:
        """ Plot clusters as a coloured scatter plot, both for all cluster,
            (without labels) and for the top_k largest clusters (with labels)
        """
        # Initialize variables
        assert top_k <= 20, "top_k must not be greater than 20"
        visualization_paths = defaultdict(lambda: {})
        is_3d = len(self.cluster_instances[0].ec_list[0].reduced_embedding) == 3
        
        # Generate a visualization for top_k clusters and all clusters
        for do_top_k in [True, False]:
            
            # Retrieve cluster data (clusters are already sorted by n_sample)
            clusters = [c for c in self.cluster_instances]  # if c.cluster_id != -1]
            symbol_seq = ["circle", "square", "diamond", "x"]
            size_seq = [1.0, 0.7, 0.7, 0.5] if is_3d else [1.0, 1.0, 1.0, 1.0]
            symbol_map = {k: symbol_seq[k % len(symbol_seq)] for k in range(100)}
            size_map = {k: size_seq[k % len(size_seq)] for k in range(100)}
            plotly_colors = px.colors.qualitative.Plotly
            if do_top_k: clusters = clusters[:top_k]
            
            # Format data for dataframe
            ys, xs, zs, raw_texts = [], [], [], []
            labels, hover_names, ids, symbols, sizes = [], [], [], [], []
            color_map = {}
            legend_line_count = 0
            for k, cluster in enumerate(clusters):
                
                # Cluster data
                label, label_line_count = self.format_text(
                    cluster.title, max_length=100, max_line_count=2,
                )
                legend_line_count += label_line_count
                labels.extend([label] * len(cluster.ec_list))
                ids.extend([cluster.cluster_id] * len(cluster.ec_list))
                hover_names.extend([self.format_text(
                    cluster.title, max_length=40, max_line_count=10,
                )[0]] * len(cluster.ec_list))
                
                # Eligibility criteria data
                xs.extend([ec.reduced_embedding[0] for ec in cluster.ec_list])
                ys.extend([ec.reduced_embedding[1] for ec in cluster.ec_list])
                if is_3d:
                    zs.extend([ec.reduced_embedding[2] for ec in cluster.ec_list])
                raw_texts.extend([self.format_text(
                    ec.raw_text, max_length=35, max_line_count=10,
                )[0] for ec in cluster.ec_list])
                
                # Eligibility criteria markers
                symbol = k // 10 if cluster.cluster_id != -1 else 0
                symbols.extend([symbol] * len(cluster.ec_list))
                size = size_map[k // 10] if cluster.cluster_id != -1 else 0.1
                sizes.extend([size] * len(cluster.ec_list))
                color = plotly_colors[k % 10] if cluster.cluster_id != -1 else "white"
                color_map[label] = color
                
            # Build dataframe for plotly.scatter
            plot_df = pd.DataFrame({
                "x": xs, "y": ys, "raw_text": raw_texts, "label": labels,
                "id": ids,  "hover_name": hover_names, "symbol": symbols,
                "size": sizes,
            })
            if is_3d: plot_df["z"] = zs
            
            # Plot cluster data using px.scatter
            hover_data = {
                "label": False, "hover_name": False, "raw_text": True,
                "symbol": False, "size": False, "x": ":.2f", "y": ":.2f",
            }
            if not is_3d:  # i.e., is_2d
                fig = px.scatter(
                    plot_df, x="x", y="y", opacity=1.0,
                    color="label", color_discrete_map=color_map,
                    labels={"label": "Cluster labels"}, size="size",
                    symbol="symbol", symbol_map=symbol_map,
                    hover_name="hover_name", hover_data=hover_data,
                )
            else:
                hover_data.update({"z": ":.2f"})
                fig = px.scatter_3d(
                    plot_df, x="x", y="y", z="z", opacity=1.0,
                    color="label", color_discrete_map=color_map,
                    labels={"label": "Cluster labels"}, size="size",
                    symbol="symbol", symbol_map=symbol_map,
                    hover_name="hover_name", hover_data=hover_data,
                )
            
            # Polish figure
            width, height = (1540, 720) if do_top_k else (720, 720)
            legend_font_size = max(1, font_size * 20 / legend_line_count)
            legend_font_size = min(font_size, legend_font_size)  # not bigger
            fig.update_traces(
                marker=dict(line=dict(color="black", width=1)),
            )
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
            fig.update_layout(
                width=width, height=height, plot_bgcolor="white",
                margin=dict(l=20, r=width * 0.45, t=20, b=20),
                legend=dict(
                    yanchor="middle", y=0.5, xanchor="left",
                    title_text="", itemsizing="constant",
                    font=dict(size=legend_font_size, family="TeX Gyre Pagella"),
                )
            )
            
            # Small improvements that depend on what is plotted
            if do_top_k:
                for trace in fig.data:
                    trace.name = trace.name.replace(', 0', '').replace(', 1', '')
            else:
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
            if not is_3d:  # i.e., is_2d
                for trace in fig.data:
                    if 'size' in trace.marker:
                        trace.marker.size = [s / 3 for s in trace.marker.size]
            
            # Save image and sets plot_path
            plot_tag = "top_%i" % top_k if do_top_k else "all"
            plot_name = "cluster_plot_%s.png" % plot_tag
            plot_path = os.path.join(self.output_dir, plot_name)
            html_path = plot_path.replace("png", "html")
            fig.write_image(plot_path, engine="kaleido", scale=2)
            fig.write_html(html_path)
            visualization_paths[plot_tag]["png"] = plot_path
            visualization_paths[plot_tag]["html"] = html_path
        
        return dict(visualization_paths)
    
    @staticmethod
    def format_text(
        text: str,
        max_length: int=70,
        max_line_count: int=2
    ) -> tuple[str, int]:
        """ Format text by cutting it into lines and trimming it if too long
        """
        # Shorten criterion type information
        text = text.replace("\n", " ").replace("<br>", " ") \
            .replace("Inclusion -", "IN:").replace("Inclusion criterion -", "IN:") \
            .replace("inclusion -", "IN:").replace("inclusion criterion -", "IN:") \
            .replace("Exclusion -", "EX:").replace("Exclusion criterion -", "EX:") \
            .replace("exclusion -", "EX:").replace("exclusion criterion -", "EX:")
        
        # Let the text as it is if its length is ok
        if len(text) <= max_length:
            return text, 1
        
        # Split the title into words and build the formatted text
        words = re.findall(r'\w+-|\w+', text)
        shortened_text = ""
        current_line_length = 0
        line_count = 1
        for word in words:
            
            # Check if adding the next word would exceed the maximum length
            added_space = (" " if word[-1] != "-" else "")
            if current_line_length + len(word) > max_length:
                if line_count == max_line_count:  # replace remaining text by "..."
                    shortened_text = shortened_text.rstrip() + "..."
                    break
                else:  # Move to the next line
                    shortened_text += "<br>" + word + added_space
                    current_line_length = len(word) + len(added_space)
                    line_count += 1
            else:
                shortened_text += word + added_space
                current_line_length += len(word) + len(added_space)
        
        return shortened_text.strip(), line_count
    
    def write_raw_ec_list(self) -> str:
        """ Generate a raw list of criteria grouped by cluster
        """
        # Open the CSV file in write mode
        raw_ec_list_path = os.path.join(self.output_dir, "raw_ec_list.csv")
        with open(raw_ec_list_path, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                "Cluster id", "Cluster prevalence",
                "Cluster Title", "Eligibility Criteria",
            ])
            
            # Iterate through ClusterOutput objects and their ClusterInstances
            for cluster_instance in self.cluster_instances:
                title = cluster_instance.title
                cluster_id = cluster_instance.cluster_id
                prevalence = cluster_instance.prevalence
                for ec_data in cluster_instance.ec_list:
                    ec_text = ec_data.raw_text
                    csv_writer.writerow([cluster_id, prevalence, title, ec_text])
        
        return raw_ec_list_path
    
    def write_to_json(self) -> str:
        """ Convert cluster output to a dictionary and write it to a json file,
            after generating a unique file name given by the project and user ids
        """
        # Define file name and sets json_path
        json_path = os.path.join(self.output_dir, "ec_clustering.json")
        self.json_path = json_path  # need to set it here for asdict(self)
        
        # Save data as a json file
        cluster_output_dict = asdict(self)
        json_data = json.dumps(cluster_output_dict, indent=4)
        with open(json_path, "w") as file:
            file.write(json_data)
        
        return json_path
    

class CUML_TSNEForBERTopic(CUML_TSNE):
    def transform(self, X):
        reduced_X = self.fit_transform(X)
        return reduced_X.to_host_array()


class TSNEForBERTopic(TSNE):
    def transform(self, X):
        reduced_X = self.fit_transform(X)
        return reduced_X


def get_dim_red_model(algorithm: str, dim: int, n_samples: int):
    """ Create a dimensionality reduction model for BERTopic
    """
    # No dimensionality reduction
    if algorithm is None:
        return BaseDimensionalityReduction()
    
    # Uniform manifold approximation and projection
    elif algorithm == "umap":
        return CUML_UMAP(
            n_components=dim,
            random_state=cfg["RANDOM_STATE"],
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
        )
    
    # Principal component analysis
    elif algorithm == "pca":
        return CUML_PCA(
            n_components=dim,
            random_state=cfg["RANDOM_STATE"],
        )
    
    # t-distributed stochastic neighbor embedding
    elif algorithm == "tsne":
        params = {
            "n_components": dim,
            "random_state": cfg["RANDOM_STATE"],
            "method": "barnes_hut" if dim < 4 else "exact",  # "fft" or "barnes_hut"?
            "n_iter": cfg["N_ITER_MAX_TSNE"],
            "n_iter_without_progress": 1000,
            "metric": "cosine",
            "learning_rate": 200.0,
        }
        if n_samples < 36_000 and dim == 2:
            n_neighbors = min(int(n_samples / 400 + 1), 90)
            small_cuml_specific_params = {
                "n_neighbors": n_neighbors,  # CannyLabs CUDA-TSNE default is 32
                "perplexity": 50.0,  # CannyLabs CUDA-TSNE default is 50
                "learning_rate_method": "none",  # not in sklearn and produces bad results
            }
            params.update(small_cuml_specific_params)
        if dim == 2:
            params.update({"verbose": cuml_logger.level_error})
            return CUML_TSNEForBERTopic(**params)
        else:
            params.update({"verbose": False})
            return TSNEForBERTopic(**params)  # cuml_tsne only available for dim = 2
    
    # Invalid name
    else:
        raise ValueError("Invalid name for dimensionality reduction algorithm")
