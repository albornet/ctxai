import os
import ast
import csv
import json
import pickle
import glob
import shutil
import numpy as np
import pandas as pd
import torch
import torchdata.datapipes.iter as dpi
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from transformers import AutoModel, AutoTokenizer
from collections import Counter
from cluster_utils import report_clusters


MODEL_STR_MAP = {
    # "bert": "bert-large-uncased",
    # "roberta": "roberta-large",
    # "bert-sentence": "efederici/sentence-bert-base",
    # "pubmed-bert-token": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "pubmed-bert-sentence": "pritamdeka/S-PubMedBert-MS-MARCO",
    # "transformer-sentence": "sentence-transformers/all-mpnet-base-v2",
    # "bioct-bert-token": "domenicrosati/ClinicalTrialBioBert-NLI4CT",
}

RAW_INPUT_FORMAT = "json"  # "json", "xlsx", "dict"
CSV_FILE_MASK = "*criteria.csv"  # "*criteria.csv", "*example.csv"
LOAD_DATA = False
LOAD_RESULTS = False

DATA_DIR = "data"
INPUT_DIR = os.path.join(DATA_DIR, "preprocessed", RAW_INPUT_FORMAT)
OUTPUT_DIR = os.path.join(DATA_DIR, "postprocessed", RAW_INPUT_FORMAT)
RESULT_DIR = os.path.join("results", RAW_INPUT_FORMAT)
STAT_REPORT_OUTPUT_PATH = os.path.join(RESULT_DIR, "cluster_report_stat.csv")
TEXT_REPORT_OUTPUT_PATH = os.path.join(RESULT_DIR, "cluster_report_text.csv")
with open(os.path.join(DATA_DIR, "mesh_crosswalk_inverted.json"), "r") as f:
    MESH_CROSSWALK_INVERTED = json.load(f)

CHOSEN_STATUSES = ["completed", "terminated"]  # ["completed", "suspended", "withdrawn", "terminated", "unknown status"]  # [] to ignore this section filter
CHOSEN_CRITERIA = []  # ["in"]  # [] to ignore this selection filter
CHOSEN_PHASES = []  # ["Phase 2"]  # [] to ignore this selection filter
CHOSEN_COND_IDS = ["C04"]  # [] to ignore this selection filter
CHOSEN_ITRV_IDS = []  # ["D02"]  # [] to ignore this selection filter
CHOSEN_COND_LVL = 4
CHOSEN_ITRV_LVL = 3

ENCODING = "utf-8"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 20  # 0 for no multiprocessing
# NUM_WORKERS = min(NUM_WORKERS, os.cpu_count() // 2)
PREFETCH_FACTOR = None if NUM_WORKERS == 0 else 1  # more?
BATCH_SIZE = 64
MAX_SELECTED_SAMPLES = 100_000  # 1_000_000
NUM_STEPS = MAX_SELECTED_SAMPLES // BATCH_SIZE

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


def main():
    input_dir = split_csv_for_multiprocessing(INPUT_DIR, NUM_WORKERS)
    if LOAD_RESULTS:
        with open(os.path.join(RESULT_DIR, "model_comparison.pkl"), "rb") as f:
            cluster_metrics = pickle.load(f)
    else:
        cluster_metrics = {}
        for model_type in MODEL_STR_MAP.keys():
            cluster_metrics[model_type] = run_one_model(model_type, input_dir)        
    plot_model_comparison(cluster_metrics)


def run_one_model(model_type, input_dir):
    """ Cluster eligibility criteria using embeddings from one language model
    """
    # Initialize language model and data pipeline
    if not LOAD_DATA:
        model, tokenizer, pooling_fn = get_model_pipeline(model_type)
        ds = get_dataset(input_dir, tokenizer)
        rs = MultiProcessingReadingService(num_workers=NUM_WORKERS)
        dl = DataLoader2(ds, reading_service=rs)
    
    # Populate tensor with eligibility criteria embeddings
    if not LOAD_DATA:
        raw_txts, metadatas = [], []
        embeddings = torch.empty((0, model.config.hidden_size))
        data_loop = tqdm(enumerate(dl), total=NUM_STEPS, leave=False)
        for i, (encoded, raw_txt, metadata) in data_loop:
            # Compute embeddings for this batch
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            ec_embeddings = pooling_fn(encoded, outputs)
            
            # Record model outputs (tensor), input texts and corresponding labels
            embeddings = torch.cat((embeddings, ec_embeddings.cpu()), dim=0)
            raw_txts.extend(raw_txt)
            metadatas.extend(metadata)
            if i >= NUM_STEPS - 1: break
        
    # Save / load data
    if LOAD_DATA:
        embeddings, raw_txts, metadatas = load_data(OUTPUT_DIR, model_type)
    else:
        save_data(OUTPUT_DIR, model_type, embeddings, raw_txts, metadatas)
    
    # Generate clusters and save results and metrics for model comparison
    os.makedirs(RESULT_DIR, exist_ok=True)
    for f in os.listdir(RESULT_DIR):
        if ".csv" in f:
            os.remove(os.path.join(RESULT_DIR, f))
    text_report, stat_report, cluster_metrics =\
        report_clusters(RESULT_DIR, OUTPUT_DIR, model_type, embeddings, raw_txts, metadatas)
    
    # Generate csv report from raw text results
    with open(STAT_REPORT_OUTPUT_PATH, "w", newline="") as file:
        csv.writer(file).writerows(stat_report)
    with open(TEXT_REPORT_OUTPUT_PATH, "w", newline="") as file:
        csv.writer(file).writerows(text_report)
    
    # Return metrics for this model
    return cluster_metrics


def get_model_pipeline(model_type):
    """ Select a model and the corresponding tokenizer and embed function
    """
    # Model generates token-level embeddings, and output [cls] (+ linear + tanh)
    if "-sentence" not in model_type:
        def pooling_fn(encoded_input, model_output):
            return model_output["pooler_output"]
        
    # Model generates sentence-level embeddings directly
    else:
        def pooling_fn(encoded_input, model_output):
            token_embeddings = model_output[0]
            attn_mask = encoded_input["attention_mask"].unsqueeze(-1)
            input_expanded = attn_mask.expand(token_embeddings.size()).float()
            token_sum = torch.sum(token_embeddings * input_expanded, dim=1)
            return token_sum / torch.clamp(input_expanded.sum(1), min=1e-9)
            
    # Return model (sent to correct device) and tokenizer
    model_str = MODEL_STR_MAP[model_type]
    model = AutoModel.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    return model.to(DEVICE), tokenizer, pooling_fn


def get_dataset(data_dir, tokenizer):
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    dataset = dpi.FileLister(data_dir, recursive=True, masks=CSV_FILE_MASK)\
        .open_files(mode="t")\
        .sharding_filter()\
        .parse_csv()\
        .filter_clinical_trial()\
        .batch(batch_size=BATCH_SIZE)\
        .tokenize(tokenizer=tokenizer)
    return dataset


def save_data(dir_, model_type, embeddings, raw_txts, labels):
    """ Simple saving function for model predictions
    """
    os.makedirs(dir_, exist_ok=True)
    torch.save(embeddings, os.path.join(dir_, "embeddings_%s.pt" % model_type))
    with open(os.path.join(dir_, "raw_txts.pkl"), "wb") as f:
        pickle.dump(raw_txts, f)
    with open(os.path.join(dir_, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)


def load_data(dir_, model_type):
    """ Simple loading function for model predictions
    """
    embeddings = torch.load(os.path.join(dir_, "embeddings_%s.pt" % model_type))
    with open(os.path.join(dir_, "raw_txts.pkl"), "rb") as f:
        raw_txts = pickle.load(f)
    with open(os.path.join(dir_, "labels.pkl"), "rb") as f:
        labels = pickle.load(f)
    return embeddings, raw_txts, labels


@functional_datapipe("filter_clinical_trial")
class ClinicalTrialFilter(dpi.IterDataPipe):
    def __init__(self, dp: dpi.IterDataPipe) -> None:
        """ Custom data pipeline to extract input text and label from CT csv file
            and to filter out trials that do not belong to a phase and condition
        """
        self.dp = dp
        all_column_names = next(iter(self.dp))
        cols = ["individual criterion", "phases", "ct path", "condition_ids",
                "intervention_ids", "category", "context", "subcontext", "label"]
        assert all([c in all_column_names for c in cols])
        self.col_id = {c: all_column_names.index(c) for c in cols}
        
    def __iter__(self):
        for i, sample in enumerate(self.dp):
            # Filter out unwanted lines of the csv file
            if i == 0: continue
            ct_metadata, ct_not_filtered = self._filter_fn(sample)
            if not ct_not_filtered: continue
            
            # Yield sample and metadata ("labels") if all is good
            yield self._build_input_text(sample), ct_metadata
            
    def _filter_fn(self, sample: dict[str, str]):
        """ Filter out CTs that do not belong to a given phase and condition
        """
        # Initialize metadata
        ct_path = sample[self.col_id["ct path"]],
        ct_status = sample[self.col_id["label"]].lower()
        metadata = {"path": ct_path, "status": ct_status}
        
        # Load relevant data
        ct_phases = ast.literal_eval(sample[self.col_id["phases"]])
        ct_cond_ids = ast.literal_eval(sample[self.col_id["condition_ids"]])
        ct_itrv_ids = ast.literal_eval(sample[self.col_id["intervention_ids"]])
        ct_cond_ids = [c for cc in ct_cond_ids for c in cc]  # flatten
        ct_itrv_ids = [i for ii in ct_itrv_ids for i in ii]  # flatten
        ct_category = sample[self.col_id["category"]]
        
        # Check criterion phases
        if len(CHOSEN_PHASES) > 0:
            if all([p not in CHOSEN_PHASES for p in ct_phases]):
                return metadata, False
        
        # Check criterion conditions
        cond_lbls = self._get_cond_itrv_labels(ct_cond_ids, CHOSEN_COND_IDS, CHOSEN_COND_LVL)
        if len(cond_lbls) == 0:
            return metadata, False
        
        # Check criterion interventions
        itrv_lbls = self._get_cond_itrv_labels(ct_itrv_ids, CHOSEN_ITRV_IDS, CHOSEN_ITRV_LVL)
        if len(itrv_lbls) == 0:
            return metadata, False
        
        # Check criterion status
        if len(CHOSEN_STATUSES) > 0:
            if ct_status not in CHOSEN_STATUSES:
                return metadata, False
        
        # Check criterion type
        if len(CHOSEN_CRITERIA) > 0:
            if ct_category not in CHOSEN_CRITERIA:
                return metadata, False
        
        # Update metadata
        metadata["phase"] = ct_phases
        metadata["condition_ids"] = ct_cond_ids
        metadata["condition"] = cond_lbls
        metadata["intervention_ids"] = ct_itrv_ids
        metadata["intervention"] = itrv_lbls
        metadata["label"] = self._get_unique_label(ct_phases, cond_lbls, itrv_lbls)

        # Accept to yield criterion if it passes all filters
        return metadata, True
    
    @staticmethod
    def _get_unique_label(phases: list[str],
                          cond_lbls: list[str],
                          itrv_lbls: list[str]
                          ) -> str:
        """ Build a single label for any combination of phase, condition, and
            intervention
        """
        phase_lbl = " - ".join(sorted(phases))
        cond_lbl = " - ".join(sorted(cond_lbls))
        itrv_lbl = " - ".join(sorted(itrv_lbls))
        return " --- ".join([phase_lbl, cond_lbl, itrv_lbl])
        
    @staticmethod
    def _get_cond_itrv_labels(ct_ids: list[str],
                              chosen_ids: list[str],
                              level: int,
                              ) -> list[str]:
        """ Construct a list of unique mesh tree labels for a list of condition
            or intervention mesh codes, aiming a specific level in the hierachy
        """
        # Filter condition or intervention ids
        if len(chosen_ids) > 0:
            ct_ids = [c for c in ct_ids if any([c.startswith(i) for i in chosen_ids])]
        
        # Select only the ones that have enough depth
        n_chars = level * 4 - 1  # format: at least "abc.def.ghi.jkl"
        cut_ct_ids = [c[:n_chars] for c in ct_ids if len(c.split(".")) >= level]
        
        # Map ids to non-code labels
        labels = [MESH_CROSSWALK_INVERTED[c] for c in cut_ct_ids]
        is_a_code = lambda lbl: (sum(c.isdigit() for c in lbl) + 1 == len(lbl))
        labels = [l for l in labels if not is_a_code(l)]
        
        # Return unique values
        return list(set(labels))
    
    def _build_input_text(self, sample):
        """ Retrieve criterion and contextual information
        """
        term_sequence = (
            # sample[self.col_id["category"]] + "clusion criterion",
            # sample[self.col_id["context"]],
            # sample[self.col_id["subcontext"]],
            sample[self.col_id["individual criterion"]],
        )
        to_join = (s for s in term_sequence if len(s) > 0)
        return " - ".join(to_join).lower()
    

@functional_datapipe("tokenize")
class Tokenizer(dpi.IterDataPipe):
    def __init__(self, dp: dpi.IterDataPipe, tokenizer):
        """ Custom data pipeline to tokenize an batch of input strings, keeping
            the corresponding labels and returning input and label batches
        """
        self.dp = dp
        self.tokenizer = tokenizer
        
    def __iter__(self):
        for batch in self.dp:
            input_batch, metadata_batch = zip(*batch)
            yield self.tokenize_fn(input_batch), input_batch, metadata_batch
    
    def tokenize_fn(self, batch):
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        
def plot_model_comparison(cluster_metrics):
    """ Generate a comparison plot between models, based on how model embeddings
        produce good clusters
    """
    # aRRACH
    def filter_fn(d: dict):
        PLOTTED_METRICS = ["sil", "db", "nm", "h", "c", "v"]
        d = {k: v for k, v in d.items() if k in PLOTTED_METRICS}
        d = {k: v / 10 if k == "db" else v for k, v in d.items()}
        return d
    cluster_metrics = {k: filter_fn(v) for k, v in cluster_metrics.items()}
    
    # Retrieve model list from the data
    labels = list(next(iter(cluster_metrics.values())).keys())
    num_models = len(cluster_metrics.keys())
    width = 0.6 / num_models  # Adjust width based on number of models
    
    # Plot cluster_metrics for each model
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (model_name, metrics) in enumerate(cluster_metrics.items()):
        
        # Plot metric values
        x_values = [i + idx * width for i, _ in enumerate(labels)]
        y_values = list(metrics.values())
        rects = ax.bar(x_values, y_values, width, label=model_name)
        
        # Auto-label the bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                text="{}".format(round(height, 2)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom",
            )

    # Adjustments to the plot
    ax.set_ylabel("Scores", fontsize="x-large")
    ax.set_title("Comparison of models for each metric", fontsize="x-large")
    ax.set_xticks([i + width * (num_models - 1) / 2 for i in range(len(labels))])
    ax.set_xticklabels(labels, fontsize="large")
    ax.legend(fontsize="large")
    plt.plot([-0.1, len(cluster_metrics) + 0.6], [0, 0], color="k")
    fig.tight_layout()
    
    # Save raw data and figure
    if not LOAD_RESULTS:
        with open(os.path.join(RESULT_DIR, "model_comparison.pkl"), "wb") as f:
            pickle.dump(cluster_metrics, f)
    plt.savefig(os.path.join(RESULT_DIR, "model_comparison.png"), dpi=300)


def print_data_summary(metadatas: list[dict[str, str]]) -> None:
    """ Debug function to find the best combination of filters and levels
    """
    # What filters were used
    print("Filters:")
    print("\t- Status: %s" % CHOSEN_STATUSES)
    print("\t- Type: %s" % CHOSEN_CRITERIA)
    print("\t- Phase: %s" % CHOSEN_PHASES)
    print("\t- Cond: %s" % CHOSEN_COND_IDS)
    print("\t- Itrv: %s" % CHOSEN_ITRV_IDS)
    
    # What level were used
    print("Levels:")
    print("\t- Cond: %s" % CHOSEN_COND_LVL)
    print("\t- Itrv: %s" % CHOSEN_ITRV_LVL)
    
    # Get labels
    ct_paths = [d["path"] for d in metadatas]
    phase_labels = [" - ".join(d["phase"]) for d in metadatas]
    cond_labels = [" - ".join(d["condition"]) for d in metadatas]
    itrv_labels = [" - ".join(d["intervention"]) for d in metadatas]
    single_labels = [d["label"] for d in metadatas]
    
    # Compute number of ECs per label (skipping small label groups)
    thresh_ec = len(metadatas) // 1000  # each label must contain a least 0.1% ECs
    single_label_counts = Counter(single_labels)
    ecs_per_label = [c for c in single_label_counts.values() if c > thresh_ec]
    
    # Compute number of CTs per label (skipping small label groups)
    thresh_ct = len(set(ct_paths)) // 1000  # each label must contain a least 0.1% CTs
    ct_paths_per_label = {}
    for single_label, ct_path in zip(single_labels, ct_paths):
        if single_label not in ct_paths_per_label:
            ct_paths_per_label[single_label] = set()
        ct_paths_per_label[single_label].add(ct_path)
    cts_per_label = [len(p) for p in ct_paths_per_label.values() if len(p) > thresh_ct]
    
    # Resulting numbers
    print("Numbers:")
    print("\t- N ECs: %s" % len(metadatas))
    print("\t- N CTs: %s" % len(set(ct_paths)))
    print("\t- N phases: %s" % len(set(phase_labels)))
    print("\t- N conds: %s" % len(set(cond_labels)))
    print("\t- N itrvs: %s" % len(set(itrv_labels)))
    print("\t- N labels: %s" % len(set(single_labels)))
    
    # Resulting statistics
    print("Statistics:")
    print("\t- Minimum EC per label: %s" % thresh_ec)
    print("\t- N labels above EC threshold: %s" % len(ecs_per_label))
    print("\t- N ECs above threshold: %s" % sum(ecs_per_label))
    print("\t- Average ECs per label: %s" % np.mean(ecs_per_label))
    print("\t- Median ECs per label: %s" % np.median(ecs_per_label))
    print("\t- Minimum CT per label: %s" % thresh_ct)
    print("\t- N labels above CT threshold: %s" % len(cts_per_label))
    print("\t- N CTs above threshold: %s" % sum(cts_per_label))
    print("\t- Average CTs per label: %s" % np.mean(cts_per_label))
    print("\t- Median CTs per label: %s" % np.median(cts_per_label))


def split_csv_for_multiprocessing(input_dir, num_workers):
    """ Split one big csv file into N different smaller files to be processed
        by multiple processes
    """
    # Check split file directory and number of workers 
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    output_dir = os.path.join(input_dir, 'mp')
    if len(csv_files) != 1: raise RuntimeError("Too many csv files")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    if num_workers <= 1: return input_dir  # original directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the big file into smaller pieces
    print("Loading csv file for multiprocessing split")
    csv_path = csv_files[0]
    csv_name = os.path.split(csv_path)[-1]
    df = pd.read_csv(csv_path)
    batch_size = len(df) // num_workers
    for i in tqdm(range(num_workers), desc="Splitting csv for multiprocessing"):
        start_index = i * batch_size
        end_index = start_index + batch_size if i < num_workers - 1 else None
        batch_df = df[start_index:end_index]
        output_path = os.path.join(output_dir, '%03i_%s' % (i + 1, csv_name))
        batch_df.to_csv(output_path, index=False)
    
    # Return updated directory
    return output_dir
            
        
if __name__ == "__main__":
    main()
    