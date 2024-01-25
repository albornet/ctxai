import os
import re
import csv
import torch
import logging
from cluster_utils import compute_medoid
from tqdm import tqdm
from typing import Union
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# General parameters
MODEL = "pubmed-bert-token"  # "pubmed-bert-token", "pubmed-bert-sentence"
N_SAMPLES = 1000

# User-defined paths
DATA_FORMAT_USER = "ctxai"
RESULT_DIR_USER = os.path.join("results", DATA_FORMAT_USER)
EMBEDDING_DIR_USER = os.path.join("data", "postprocessed", DATA_FORMAT_USER)
CLUSTER_STAT_PATH_USER = os.path.join(RESULT_DIR_USER, "stat_%s.csv" % MODEL)
CLUSTER_TEXT_PATH_USER = os.path.join(RESULT_DIR_USER, "text_%s.csv" % MODEL)
EMBEDDING_PATH_USER = os.path.join(EMBEDDING_DIR_USER, "embeddings_%s.pt" % MODEL)
OUTPUT_PATH_USER = os.path.join(RESULT_DIR_USER, "list_user_method_%s.csv" % MODEL)

# General ct.gov paths
DATA_FORMAT_CTGOV = "json"
RESULT_DIR_CTGOV = os.path.join("results", DATA_FORMAT_CTGOV)
EMBEDDING_DIR_CTGOV = os.path.join("data", "postprocessed", DATA_FORMAT_CTGOV)
CLUSTER_STAT_PATH_CTGOV = os.path.join(RESULT_DIR_CTGOV, "stat_%s.csv" % MODEL)
CLUSTER_TEXT_PATH_CTGOV = os.path.join(RESULT_DIR_CTGOV, "text_%s.csv" % MODEL)
EMBEDDING_PATH_CTGOV = os.path.join(EMBEDDING_DIR_CTGOV, "embeddings_%s.pt" % MODEL)
OUTPUT_PATH_CTGOV = os.path.join(RESULT_DIR_USER, "list_ctgov_method_%s.csv" % MODEL)


def main() -> None:
    """ Generate cluster lists using user and ct.gov data
    """
    # Set logging level and format
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname).1s %(asctime)s] %(message)s",
    )
    
    # Load filtered data (user-defined)
    logging.info("Loading data filtered by user")
    cluster_stats_user = load_stats(CLUSTER_STAT_PATH_USER)
    cluster_texts_and_embeddings_user = load_texts_and_embeddings(
        CLUSTER_TEXT_PATH_USER, EMBEDDING_PATH_USER,
    )
    
    # Load general data (whole ct.gov)
    logging.info("Loading ct.gov data")
    cluster_stats_ctgov = load_stats(CLUSTER_STAT_PATH_CTGOV)
    cluster_texts_and_embeddings_ctgov = load_texts_and_embeddings(
        CLUSTER_TEXT_PATH_CTGOV, EMBEDDING_PATH_CTGOV,
    )
    
    # Generate cluster lists using the "user-only" way
    logging.info("Generating sample-cluster pairs with user data")
    generate_cluster_lists(
        cluster_stats_user,
        cluster_texts_and_embeddings_user,
        OUTPUT_PATH_USER,
    )
    
    # Generate cluster lists using the whole ct.gov data as reference
    logging.info("Generating sample-cluster pairs by referring to ct.gov data")
    generate_cluster_lists_by_referring_to_ctgov(
        cluster_stats_user,
        cluster_texts_and_embeddings_user,
        cluster_stats_ctgov,
        cluster_texts_and_embeddings_ctgov,
        OUTPUT_PATH_CTGOV,
    )
    
    
def load_stats(csv_file_path: str) -> dict[int, dict[str, str]]:
    """ Load cluster statistics from a CSV file.
        :param csv_file_path: path to the CSV file containing cluster statistics
        :return: dict mapping cluster ids to corresponding statistics
    """
    data = {}
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                cluster_id = row.pop("Cluster id")
                data[int(cluster_id)] = row
            except ValueError:  # last row is something else
                pass
    return data


def load_texts_and_embeddings(
    csv_file_path: str,
    embedding_path: str,
) -> dict[int, dict[str, Union[str, torch.Tensor]]]:
    """ Load texts and their embeddings from specified files.
        :param csv_file_path: path to the CSV file containing text data
        :param embedding_path: path to the file containing the embeddings
        :return: dict mapping cluster ids to corresponding texts and embeddings
    """
    embeddings = torch.load(embedding_path)
    data = defaultdict(list)
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = [int(cluster_id) for cluster_id in next(reader)]
        for row in reader:
            for header, field in zip(headers, row):
                if len(field) > 0:
                    sample_id, text = parse_member_id_and_text(field)
                    embedding = embeddings[sample_id]
                    data[header].append({"text": text, "embedding": embedding})
    return data


def parse_member_id_and_text(xml_string: str) -> tuple[int, str]:
    """ Extract member id and text content from an XML string
        :param xml_string: xml string to parse
        :return: member id and corresponding text
    """
    sample_id_match = re.search(r"<member_id>(\d+)</member_id>", xml_string)
    text_match = re.search(r"<text>(.*?)</text>", xml_string)
    sample_id = sample_id_match.group(1) if sample_id_match else None
    text = text_match.group(1) if text_match else None
    return int(sample_id), text


def generate_cluster_lists(
    cluster_stats: dict[int, dict[str, str]],
    cluster_texts_and_embeddings: dict[int, dict[str, Union[str, torch.Tensor]]],
    result_path: str,
    sort_by_key: str="prevalence",
    n_max_samples_per_cluster: int=20,
    n_max_samples_to_write: int=1000,
) -> None:
    """ Generate a csv file listing sample clusters
        :param cluster_stats: dict for cluster statistics.
        :param cluster_texts_and_embeddings: dict for cluster texts and embeddings
        :param result_path: path to save the resulting CSV file
        :param sort_by_key: cluster with largest values of this keys written first
        :param n_samples_per_cluster: number of samples per cluster to include
    """
    # Sort cluster dictionary
    cluster_stats = dict(sorted(
        cluster_stats.items(),
        key=lambda item: float(item[1][sort_by_key]),
        reverse=True,
    ))
    if -1 in cluster_stats.keys():  # undertermined cluster last
        cluster_stats.update({-1: cluster_stats.pop(-1)})
    
    # Write dictionary content to csv file
    n_samples_written = 0
    with open(result_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["title", sort_by_key, "criterion", "good?"])
        for cluster_id, stat in cluster_stats.items():
            metric = stat[sort_by_key]
            title = stat["title"]
            samples = cluster_texts_and_embeddings[cluster_id]
            if n_max_samples_per_cluster is not None:
                samples = samples[:n_max_samples_per_cluster]
            for sample in samples:
                writer.writerow([title, metric, sample["text"], "yes/no?"])
                n_samples_written += 1
                if n_samples_written >= n_max_samples_to_write: return
                
                
def generate_cluster_lists_by_referring_to_ctgov(
    cluster_stats_user: dict[int, dict[str, str]],
    cluster_texts_and_embeddings_user: dict[int, list[dict[str, Union[str, torch.Tensor]]]],
    cluster_stats_ctgov: dict[int, dict[str, str]],
    cluster_texts_and_embeddings_ctgov: dict[int, list[dict[str, Union[str, torch.Tensor]]]],
    result_path: str,
) -> None:
    """ Generate cluster lists by comparing user data with ct.gov data
        :param cluster_stats_user: user-specific cluster statistics
        :param cluster_texts_and_embeddings_user: user-specific texts and embeddings
        :param cluster_stats_ctgov: ct.gov cluster statistics
        :param cluster_texts_and_embeddings_ctgov: ct.gov texts and embeddings
        :param result_path: path to save the resulting CSV file
    """
    # Compute cluster centroids from the ct.gov clutering data
    logging.info("Computing cluster representants from ct.gov data")
    # cluster_representants_ctgov = compute_cluster_centroids(cluster_texts_and_embeddings_ctgov)
    cluster_representants_ctgov = compute_cluster_medoids(cluster_texts_and_embeddings_ctgov)
    cluster_ids_ctgov = list(cluster_representants_ctgov.keys())
    representants_ctgov = torch.stack(list(cluster_representants_ctgov.values()))
    
    # Infer cluster ids for the user dataset, using the ct.gov clusters
    inferred_cluster_info = defaultdict(list)
    for cluster_id, cluster_data in cluster_texts_and_embeddings_user.items():
        for sample in cluster_data:
            # Infer cluster id, based on ct.gov data
            sample_embedding = sample["embedding"].unsqueeze(dim=0)
            similarities = cosine_similarity(sample_embedding, representants_ctgov)
            max_index = torch.argmax(torch.tensor(similarities))
            inferred_cluster_id = cluster_ids_ctgov[max_index]
            
            # Add sample to the new cluster info dictionary
            sample.update({"similarity": similarities[0, max_index]})
            inferred_cluster_info[inferred_cluster_id].append(sample)
    
    # Sort samples based on simliarity to their inferred cluster centroid
    for cluster_id, cluster_data in inferred_cluster_info.items():
        cluster_data.sort(key=lambda x: x["similarity"], reverse=True)
        
    # Find cluster specificty based on sample importance and cluster prevalence
    n_samples = sum([len(data) for data in inferred_cluster_info.values()])
    inferred_cluster_stats = {}
    for cluster_id, cluster_data in inferred_cluster_info.items():
        prevalence = len(cluster_data) / n_samples
        ctgov_prevalence = cluster_stats_ctgov[inferred_cluster_id]["prevalence"]
        stats = {
            "prevalence": prevalence,
            "specificity": prevalence / float(ctgov_prevalence),
            "title": cluster_stats_ctgov[cluster_id]["title"]
        }
        inferred_cluster_stats[cluster_id] = stats
        
    # Generate cluster lists using the "user-only" way
    generate_cluster_lists(
        inferred_cluster_stats,
        inferred_cluster_info,
        result_path,
        sort_by_key="specificity",
    )


def compute_cluster_centroids(
    cluster_texts_and_embeddings: dict[int, dict[str, Union[str, torch.Tensor]]],
) -> dict[int, torch.Tensor]:
    """ Compute centroids for each cluster based on embeddings
        :param cluster_texts_and_embeddings: dict of cluster texts and embeddings
        :return: dict mapping cluster ids to the centroid of their embeddings
    """
    cluster_centroids = {}
    for cluster_id, cluster_data in cluster_texts_and_embeddings.items():
        if cluster_id != -1:
            cluster_embeddings = [sample["embedding"] for sample in cluster_data]
            cluster_centroid = torch.mean(torch.stack(cluster_embeddings), dim=0)
            cluster_centroids[cluster_id] = cluster_centroid
    return cluster_centroids


def compute_cluster_medoids(
    cluster_texts_and_embeddings: dict[int, dict[str, Union[str, torch.Tensor]]],
) -> dict[int, torch.Tensor]:
    """ Compute cluster centroids by averaging samples within each cluster,
        weighting by the sample probability of being in that cluster
    """
    cluster_centroids = {}
    for cluster_id, cluster_data in tqdm(cluster_texts_and_embeddings.items(), leave=False):
        if cluster_id != -1:
            cluster_embeddings = [sample["embedding"] for sample in cluster_data]
            cluster_embeddings = torch.stack(cluster_embeddings).numpy()
            cluster_medoid = compute_medoid(cluster_embeddings)
            cluster_centroids[cluster_id] = torch.from_numpy(cluster_medoid)
    return cluster_centroids


if __name__ == "__main__":
    main()
    