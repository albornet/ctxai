import os
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gc
import cupy as cp


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
LOAD_PREPROCESSED_DATA = True  # False
LOAD_EMBEDDINGS = True  # False
LOAD_REDUCED_EMBEDDINGS = True  # False
LOAD_OPTUNA_RESULTS = True  # False
LOAD_CLUSTER_RESULTS = True  # False
NUM_GPUS = get_gpu_count()
NUM_OPTUNA_WORKERS = 1
NUM_OPTUNA_THREADS = 1


# Directories and data format
BASE_DATA_DIR = "data"
RAW_INPUT_FORMAT = "ctxai"  # "json", "ctxai"
PREPROCESSED_DIR = os.path.join(BASE_DATA_DIR, "preprocessed", RAW_INPUT_FORMAT)  # results saved here after parsing criteria
POSTPROCESSED_DIR = os.path.join(BASE_DATA_DIR, "postprocessed", RAW_INPUT_FORMAT)  # results saved here during and after clustering criteria


# Eligility criteria parsing parameters
TOO_MANY_WORKERS = max(os.cpu_count() - 4, os.cpu_count() // 4)
NUM_PARSE_WORKERS = 12  # 0 for no multi processing at the parsing level
NUM_PARSE_WORKERS = min(NUM_PARSE_WORKERS, TOO_MANY_WORKERS)


# Eligibility criteria embedding model parameters
BATCH_SIZE = 64
MAX_SELECTED_SAMPLES = 280_000 if RAW_INPUT_FORMAT == "json" else 7_000
DEVICE = "cuda:0" if NUM_GPUS > 0 else "cpu"
NUM_STEPS = MAX_SELECTED_SAMPLES // BATCH_SIZE
DEFAULT_MODEL_ID = "pubmed-bert-sentence"
MODEL_STR_MAP = {
    "pubmed-bert-sentence": "pritamdeka/S-PubMedBert-MS-MARCO",
    # "transformer-sentence": "sentence-transformers/all-mpnet-base-v2",
    # "bert-sentence": "efederici/sentence-bert-base",
    # "pubmed-bert-token": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    # "bioct-bert-token": "domenicrosati/ClinicalTrialBioBert-NLI4CT",
    # "roberta": "roberta-large",
    # "bert": "bert-large-uncased",
}


# Eligibility criteria dataset parameters
PREPROCESSED_DATA_HEADERS = [
    "criteria paragraph", "complexity", "ct path", "label", "phases",
    "conditions", "condition_ids", "intervention_ids", "category", "context",
    "subcontext", "individual criterion",
]
PREPROCESSED_FILE_MASK = "*criteria.csv"
RESULT_DIR = os.path.join("results", RAW_INPUT_FORMAT)
with open(os.path.join(BASE_DATA_DIR, "mesh_crosswalk_inverted.json"), "r") as f:
    MESH_CROSSWALK_INVERTED = json.load(f)


# Eligibility criteria labelling parameters ([] to ignore filters)
CHOSEN_STATUSES = []  # ["completed", "terminated"]  # ["completed", "suspended", "withdrawn", "terminated", "unknown status"]  # [] to ignore this section filter
CHOSEN_CRITERIA = []  # ["in"]  # [] to ignore this selection filter
CHOSEN_PHASES = []  # ["Phase 2"]  # [] to ignore this selection filter
# Infections ["C01"] // Neoplasms ["C04"] // Cardiovascular Diseases ["C14"] // Immune System Diseases ["C20"]
CHOSEN_COND_IDS = ["C04"] if RAW_INPUT_FORMAT == "json" else []  # TODO: CARDIOLOGY? ETC. LOOK FOR OTHER C0N
CHOSEN_ITRV_IDS = []  # ["D02"]  # [] to ignore this selection filter
CHOSEN_COND_LVL = None  # 4  # None to ignore this one
CHOSEN_ITRV_LVL = None  # 3  # None to ignore this one
STATUS_MAP = None


# Dimensionality reduction parameters
CLUSTER_DIM_RED_ALGO = "tsne"  # "pca", "tsne"
PLOT_DIM_RED_ALGO = "tsne"  # "pca", "tsne"
CLUSTER_RED_DIM = 2  # None for no dimensionality reduction when clustering
PLOT_RED_DIM = 2  # either 2 or 3
N_ITER_MAX_TSNE = 25_000
REPRESENTATION_METRIC = None  # None, "correlation", "euclidean"


# Clustering algorithm and hyper-optimization (optuna) parameters
N_OPTUNA_TRIALS = 100
N_CLUSTER_MAX = 500
DO_SUBCLUSTERIZE = True  # if True, try to cluster further each computed cluster
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


# Default parameters for cluster title generation
DEFAULT_CLUSTER_SUMMARIZATION_PARAMS = {
    "method": "gpt",  # "closest", "shortest", "gpt"
    "n_representants": 20,
    "gpt_system_prompt": " ".join([  # only used if method == gpt
        "You are an expert in the fields of clinical trials and eligibility criteria.",
        "You express yourself succintly, i.e., less than 100 characters per response.",
    ]),
    "gpt_user_prompt_intro": " ".join([  # only used if method == gpt
        "I will show you a list of eligility criteria. They share some level of similarity.",
        "Your task is to create a single, short tag that effecively summarizes the list.",
        "Importantly, the tag should be very short and concise, i.e., a few words, and under 100 characters",
        "You can use medical abbreviations and you should avoid focusing on details or outliers.",
        "Write only your answer, starting with 'Inclusion - ' or 'Exclusion - '.",
        "Summarize the criteria list as a whole, choosing either 'Inclusion' or 'Exclusion' for your tag.",
        "Here is the list of criteria (each one is on a new line):\n",
    ])
}


# # Cluster plot parameters (was used for an old plot)
# FIG_SIZE = (11, 11)
# TEXT_SIZE = 16
# COLORS = np.array((
#     list(plt.cm.tab20(np.arange(20)[0::2])) + \
#     list(plt.cm.tab20(np.arange(20)[1::2]))) * (N_CLUSTER_MAX // 20 + 1)
# )
# NA_COLOR_ALPHA = 0.1
# NOT_NA_COLOR_ALPHA = 0.8
# NA_COLOR = np.array([0.0, 0.0, 0.0, NA_COLOR_ALPHA])
# COLORS[:, -1] = NOT_NA_COLOR_ALPHA
# SCATTER_PARAMS = {"s": 0.33, "linewidth": 0}
# LEAF_SEPARATION = 0.3


def clean_memory_fn():
    """ Try to remove unused variables in GPU and CPU, after each model run
    """
    gc.collect()
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


def clean_memory():
    """ Decorator to clean memory before and after a function call
    """
    def decorator(original_function):
        def wrapper(*args, **kwargs):
            clean_memory_fn()
            result = original_function(*args, **kwargs)
            clean_memory_fn()
            return result
        return wrapper
    return decorator

