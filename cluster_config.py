import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


# General boolean parameters
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', action='store_true', help='Script run on HPC')
args = parser.parse_args()
NUM_WORKERS = 40 if args.hpc else 12
LOAD_FINAL_RESULTS = False
LOAD_EMBEDDINGS = False
LOAD_REDUCED_EMBEDDINGS = False
LOAD_CLUSTER_INFO = False
DO_OPTUNA = True


# Eligibility crietria embedding model parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MAX_SELECTED_SAMPLES = 1_000_000
NUM_STEPS = MAX_SELECTED_SAMPLES // BATCH_SIZE
MODEL_STR_MAP = {
    "pubmed-bert-sentence": "pritamdeka/S-PubMedBert-MS-MARCO",
    "transformer-sentence": "sentence-transformers/all-mpnet-base-v2",
    "bert-sentence": "efederici/sentence-bert-base",
    "pubmed-bert-token": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "bioct-bert-token": "domenicrosati/ClinicalTrialBioBert-NLI4CT",
    "roberta": "roberta-large",
    "bert": "bert-large-uncased",
}


# Eligibility criteria dataset parameters
RAW_INPUT_FORMAT = "json"  # "json", "xlsx", "dict"
CSV_FILE_MASK = "*criteria.csv"  # "*criteria.csv", "*example.csv"
DATA_DIR = "data"
INPUT_DIR = os.path.join(DATA_DIR, "preprocessed", RAW_INPUT_FORMAT)
OUTPUT_DIR = os.path.join(DATA_DIR, "postprocessed", RAW_INPUT_FORMAT)
RESULT_DIR = os.path.join("results", RAW_INPUT_FORMAT)
with open(os.path.join(DATA_DIR, "mesh_crosswalk_inverted.json"), "r") as f:
    MESH_CROSSWALK_INVERTED = json.load(f)


# Eligibility criteria labelling parameters
CHOSEN_STATUSES = ["completed", "terminated"]  # ["completed", "suspended", "withdrawn", "terminated", "unknown status"]  # [] to ignore this section filter
CHOSEN_CRITERIA = []  # ["in"]  # [] to ignore this selection filter
CHOSEN_PHASES = []  # ["Phase 2"]  # [] to ignore this selection filter
CHOSEN_COND_IDS = ["C04"]  # [] to ignore this selection filter
CHOSEN_ITRV_IDS = []  # ["D02"]  # [] to ignore this selection filter
CHOSEN_COND_LVL = 4
CHOSEN_ITRV_LVL = 3
# STATUS_MAP = {
#     "completed": "good",
#     "suspended": "bad",
#     "withdrawn": "bad",
#     "terminated": "bad",
#     "unknown status": "bad",
#     "recruiting": "bad",
#     "not yet recruiting": "bad",
#     "active, not recruiting": "bad",
# }
STATUS_MAP = None  # to use raw labels


# Clustering algorithm parameters
CLUSTER_DIM_RED_ALGO = "tsne"  # "pca", "tsne"
PLOT_DIM_RED_ALGO = "tsne"  # "pca", "tsne"
CLUSTER_RED_DIM = 2  # None for no dimensionality reduction when clustering
PLOT_RED_DIM = 2  # either 2 or 3
DO_SUBCLUSTERIZE = True  # if True, try to cluster further each computed cluster
N_ITER_MAX_TSNE = 100_000  # 1_000, 2_000, 10_000
CLUSTER_SUMMARIZATION_METHOD = "chatgpt" if args.hpc else "closest"


# Cluster plot parameters
FIG_SIZE = (11, 11)
TEXT_SIZE = 16
COLORS = np.array((
    list(plt.cm.tab20(np.arange(20)[0::2])) + \
    list(plt.cm.tab20(np.arange(20)[1::2]))) * 5
)
NA_COLOR_ALPHA = 0.1
NOT_NA_COLOR_ALPHA = 0.8
NA_COLOR = np.array([0.0, 0.0, 0.0, NA_COLOR_ALPHA])
COLORS[:, -1] = NOT_NA_COLOR_ALPHA
SCATTER_PARAMS = {"s": 50, "linewidth": 0}
LEAF_SEPARATION = 0.3


# Cluster algorithm hyper-optimization parameters (optuna)
N_OPTUNA_TRIALS = 100
OPTUNA_PARAM_RANGES = {
    "max_cluster_size_primary": [0.02, 0.1],
    "min_cluster_size_primary": [0.0, 0.01],
    "min_samples_primary": [0.0, 0.004],
    "max_cluster_size_secondary": [0.3, 1.0],
    "min_cluster_size_secondary": [0.0, 0.4],
    "min_samples_secondary": [0.0, 0.004],
    "alpha": [0.5, 5.0],
    "cluster_selection_method": ["eom"],
}
DEFAULT_CLUSTERING_PARAMS = {
    'max_cluster_size_primary': 0.046785350806582304,
    'min_cluster_size_primary': 0.006506556037527307,
    'min_samples_primary': 1.1262890212484896e-05,
    'max_cluster_size_secondary': 0.8623302183161973,
    'min_cluster_size_secondary': 0.18352927484472079,
    'min_samples_secondary': 0.0038758800804370368,
    'alpha': 2.8146278364855495,
    'cluster_selection_method': 'eom',
}
