import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess
import json


def get_gpu_count():
    """ Count the number available GPU devices based on the output of nvidia-smi
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '-L'], text=True)
        gpu_count = len(output.strip().split('\n'))
        return gpu_count
    except subprocess.CalledProcessError:
        return 0


# General parameters
parser = argparse.ArgumentParser()
parser.add_argument('--hpc', action='store_true', help='Script run on HPC')
args = parser.parse_args()
USE_CUML = True  # todo: include version without cuml for small data
LOAD_EMBEDDINGS = False
LOAD_REDUCED_EMBEDDINGS = False
LOAD_OPTUNA_RESULTS = False
LOAD_CLUSTER_INFO = False
LOAD_FINAL_RESULTS = False
NUM_GPUS = get_gpu_count()
NUM_OPTUNA_WORKERS = 1
NUM_OPTUNA_THREADS = 1


# Eligibility criteria embedding model parameters
DEVICE = "cuda:0" if NUM_GPUS > 0 else "cpu"
BATCH_SIZE = 256 if args.hpc else 64
MAX_SELECTED_SAMPLES = 280_000
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
STATUS_MAP = None  # to use raw labels


# Dimensionality reduction parameters
CLUSTER_DIM_RED_ALGO = "tsne"  # "pca", "tsne"
PLOT_DIM_RED_ALGO = "tsne"  # "pca", "tsne"
CLUSTER_RED_DIM = 2  # None for no dimensionality reduction when clustering
PLOT_RED_DIM = 2  # either 2 or 3
N_ITER_MAX_TSNE = 10_000 if args.hpc else 10_000


# Clustering algorithm and hyper-optimization (optuna) parameters
N_OPTUNA_TRIALS = 100
N_CLUSTER_MAX = 500
DO_SUBCLUSTERIZE = True  # if True, try to cluster further each computed cluster
CLUSTER_SUMMARIZATION_METHOD = "closest" if args.hpc else "closest"
OPTUNA_PARAM_RANGES = {
    "max_cluster_size_primary": [0.02, 0.1],
    "min_cluster_size_primary": [0.0007, 0.01],
    "min_samples_primary": [0.0, 0.0007],
    "max_cluster_size_secondary": [0.3, 1.0],
    "min_cluster_size_secondary": [0.001, 0.4],
    "min_samples_secondary": [0.0, 0.001],
    "alpha": [0.5, 5.0],
    "cluster_selection_method": ["eom"],
}
DEFAULT_CLUSTERING_PARAMS = {
    'max_cluster_size_primary': 0.059104588066699,
    'min_cluster_size_primary': 0.004393240987952118,
    'min_samples_primary': 0.00022394924184848854,
    'max_cluster_size_secondary': 0.9714541246953673,
    'min_cluster_size_secondary': 0.1735510702949971,
    'min_samples_secondary': 0.00031720162384443566,
    'alpha': 0.6993816357610149,
    'cluster_selection_method': 'eom'
}


# Cluster plot parameters
FIG_SIZE = (11, 11)
TEXT_SIZE = 16
COLORS = np.array((
    list(plt.cm.tab20(np.arange(20)[0::2])) + \
    list(plt.cm.tab20(np.arange(20)[1::2]))) * (N_CLUSTER_MAX // 20 + 1)
)
NA_COLOR_ALPHA = 0.1
NOT_NA_COLOR_ALPHA = 0.8
NA_COLOR = np.array([0.0, 0.0, 0.0, NA_COLOR_ALPHA])
COLORS[:, -1] = NOT_NA_COLOR_ALPHA
SCATTER_PARAMS = {"s": 0.33, "linewidth": 0}
LEAF_SEPARATION = 0.3
