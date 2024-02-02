import os
import ast
import pickle
import glob
import shutil
import logging
import pandas as pd
import torch
import torchdata.datapipes.iter as dpi
import warnings
warnings.filterwarnings(
    action="ignore", category=UserWarning, message="TypedStorage is deprecated"
)
import matplotlib.pyplot as plt
from typing import Union
from tqdm import tqdm
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2 import DataLoader2, InProcessReadingService
from transformers import AutoModel, AutoTokenizer
try:
    from . import config as cfg
    from .cluster_utils import ClusterOutput, report_clusters
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger = logging.getLogger("cluster")
except ImportError:  # cluster_data.py file run as a script
    import sys
    import config as cfg
    from cluster_utils import ClusterOutput, report_clusters
    logger = logging.getLogger("cluster")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main():
    """ If ran as a script, call cluster_data for several models and write
        a summary of results to a directory given by cfg.RESULT_DIR
    """
    # Generate results using one model, then save the results
    cluster_metrics = {}
    for model_id in cfg.MODEL_STR_MAP.keys():
        logger.info(" - Starting with %s" % model_id)
        cluster_output = cluster_data_fn(
            model_id=model_id,
            cluster_summarization_params=cfg.DEFAULT_CLUSTER_SUMMARIZATION_PARAMS,
        )
        cluster_metrics[model_id] = cluster_output.cluster_metrics
        logger.info(" - Done with %s" % model_id)
    
    # Plot final results (comparison of different model embeddings)
    plot_model_comparison(cluster_metrics)
    logger.info("Model comparison finished!")


def cluster_data_fn(
    model_id: str | None,
    user_id: str | None=None,
    project_id: str | None=None,
    cluster_summarization_params: dict[int, Union[int, str]] | None=None,
) -> ClusterOutput:
    """ Cluster eligibility criteria using embeddings from one language model
    """
    # Take default model if not provided
    if model_id is None:
        model_id = cfg.DEFAULT_MODEL_ID
        
    # Save / load data
    logger.info(" - Retrieving elibility criteria embeddings with %s" % model_id)
    if cfg.LOAD_EMBEDDINGS:
        embeddings, raw_txts, metadatas = load_embeddings(
            output_dir=cfg.POSTPROCESSED_DIR, model_id=model_id,
        )
    else:
        embeddings, raw_txts, metadatas = generate_embeddings(
            input_dir=cfg.PREPROCESSED_DIR, model_id=model_id,
        )
        save_embeddings(
            output_dir=cfg.POSTPROCESSED_DIR,
            model_id=model_id,
            embeddings=embeddings,
            raw_txts=raw_txts,
            metadatas=metadatas,
        )
    
    # Generate clusters and report them as a formatted data structure
    logger.info(" - Running clustering algorithm on eligibility criteria embeddings")
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    return report_clusters(
        model_id=model_id,
        raw_data=embeddings,
        raw_txts=raw_txts,
        metadatas=metadatas,
        cluster_summarization_params=cluster_summarization_params,
        user_id=user_id,
        project_id=project_id,
    )
    

def get_model_pipeline(model_id):
    """ Select a model and the corresponding tokenizer and embed function
    """
    # Model generates token-level embeddings, and output [cls] (+ linear + tanh)
    if "-sentence" not in model_id:
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
    model_str = cfg.MODEL_STR_MAP[model_id]
    model = AutoModel.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    return model.to(cfg.DEVICE), tokenizer, pooling_fn


def get_dataset(data_dir, tokenizer):
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    ds = dpi.FileLister(data_dir, recursive=True, masks=cfg.PREPROCESSED_FILE_MASK)\
        .open_files(mode="t", encoding="utf-8")\
        .parse_csv()\
        .sharding_filter()\
        .filter_clinical_trial()\
        .batch(batch_size=cfg.BATCH_SIZE)\
        .tokenize(tokenizer=tokenizer)
    return ds


def generate_embeddings(
    input_dir: str,
    model_id: str
) -> tuple[torch.Tensor, list[str], list[dict]]:
    """ Generate a set of embeddigns from data in a given input directory, using
        a given model
    """
    # Load model and data pipeline
    logger.info(" --- Running model to generate embeddings")
    model, tokenizer, pooling_fn = get_model_pipeline(model_id)
    ds = get_dataset(input_dir, tokenizer)
    rs = InProcessReadingService()
    dl = DataLoader2(ds, reading_service=rs)
    
    # Go through data pipeline
    raw_txts, metadatas = [], []
    embeddings = torch.empty((0, model.config.hidden_size))
    for i, (encoded, raw_txt, metadata) in tqdm(
        iterable=enumerate(dl),
        total=cfg.NUM_STEPS,
        leave=False,
        desc="Processing eligibility criteria dataset"
    ):
        
        # Compute embeddings for this batch
        encoded = {k: v.to(cfg.DEVICE) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        ec_embeddings = pooling_fn(encoded, outputs)
        
        # Record model outputs (tensor), input texts and corresponding labels
        embeddings = torch.cat((embeddings, ec_embeddings.cpu()), dim=0)
        raw_txts.extend(raw_txt)
        metadatas.extend(metadata)
        if i >= cfg.NUM_STEPS - 1: break
    
    # Make sure gpu memory is made free for report_cluster
    torch.cuda.empty_cache()
    
    # Return embeddings, as well as raw text data and some metadata 
    return embeddings, raw_txts, metadatas
    

def save_embeddings(output_dir, model_id, embeddings, raw_txts, metadatas):
    """ Simple saving function for model predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "embeddings_%s.pt" % model_id)
    torch.save(embeddings, ckpt_path)
    with open(os.path.join(output_dir, "raw_txts.pkl"), "wb") as f:
        pickle.dump(raw_txts, f)
    with open(os.path.join(output_dir, "metadatas.pkl"), "wb") as f:
        pickle.dump(metadatas, f)


def load_embeddings(output_dir, model_id):
    """ Simple loading function for model predictions
    """
    logger.info(" --- Loading embeddings from previous run")
    ckpt_path = os.path.join(output_dir, "embeddings_%s.pt" % model_id)
    embeddings = torch.load(ckpt_path)
    with open(os.path.join(output_dir, "raw_txts.pkl"), "rb") as f:
        raw_txts = pickle.load(f)
    with open(os.path.join(output_dir, "metadatas.pkl"), "rb") as f:
        metadatas = pickle.load(f)
    return embeddings, raw_txts, metadatas


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
        self.yielded_input_texts = []
        
    def __iter__(self):
        for i, sample in enumerate(self.dp):
            # Filter out unwanted lines of the csv file
            if i == 0: continue
            ct_metadata, ct_not_filtered = self._filter_fn(sample)
            if not ct_not_filtered: continue
            
            # Yield sample and metadata ("labels") if all is good
            input_text = self._build_input_text(sample)
            if input_text not in self.yielded_input_texts:
                self.yielded_input_texts.append(input_text)
                yield input_text, ct_metadata
            
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
        if len(cfg.CHOSEN_PHASES) > 0:
            if all([p not in cfg.CHOSEN_PHASES for p in ct_phases]):
                return metadata, False
        
        # Check criterion conditions
        cond_lbls = self._get_cond_itrv_labels(ct_cond_ids, cfg.CHOSEN_COND_IDS, cfg.CHOSEN_COND_LVL)
        if len(cond_lbls) == 0:
            return metadata, False
        
        # Check criterion interventions
        itrv_lbls = self._get_cond_itrv_labels(ct_itrv_ids, cfg.CHOSEN_ITRV_IDS, cfg.CHOSEN_ITRV_LVL)
        if len(itrv_lbls) == 0:
            return metadata, False
        
        # Check criterion status
        if len(cfg.CHOSEN_STATUSES) > 0:
            if ct_status not in cfg.CHOSEN_STATUSES:
                return metadata, False
        
        # Check criterion type
        if len(cfg.CHOSEN_CRITERIA) > 0:
            if ct_category not in cfg.CHOSEN_CRITERIA:
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
        # Case where condition or intervention is not important
        if level is None: return ["N/A"]
        
        # Filter condition or intervention ids
        if len(chosen_ids) > 0:
            ct_ids = [c for c in ct_ids if any([c.startswith(i) for i in chosen_ids])]
        
        # Select only the ones that have enough depth
        n_chars = level * 4 - 1  # format: at least "abc.def.ghi.jkl"
        cut_ct_ids = [c[:n_chars] for c in ct_ids if len(c.split(".")) >= level]
        
        # Map ids to non-code labels
        labels = [cfg.MESH_CROSSWALK_INVERTED[c] for c in cut_ct_ids]
        is_a_code = lambda lbl: (sum(c.isdigit() for c in lbl) + 1 == len(lbl))
        labels = [l for l in labels if not is_a_code(l)]
        
        # Return unique values
        return list(set(labels))
    
    def _build_input_text(self, sample):
        """ Retrieve criterion and contextual information
        """
        term_sequence = (
            sample[self.col_id["category"]] + "clusion criterion",
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
        
        
def plot_model_comparison(metrics):
    """ Generate a comparison plot between models, based on how model embeddings
        produce good clusters
    """
    # Load, parse, ,and format data
    to_plot = ["Silhouette score", "DB index", "Dunn index", "MI score",
               "AMI score", "Homogeneity", "Completeness", "V measure"]
    def filter_fn(d: dict[str, dict]) -> dict:
        d_free, d_dept = d["label_free"], d["label_dept"]
        d_free = {k: v for k, v in d_free.items() if k in to_plot}
        d_free = {k: v / 10 if k == "DB index" else v for k, v in d_free.items()}
        d_dept = {k: v for k, v in d_dept.items() if k in to_plot}
        d_dept = {k: v / 5 if k == "MI score" else v for k, v in d_dept.items()}
        d_free.update(d_dept)
        return d_free
    metrics = {k: filter_fn(v) for k, v in metrics.items()}
    
    # Retrieve metric labels and model names
    labels = list(next(iter(metrics.values())).keys())
    labels = ["%s / 10.0" % l if l == "DB index" else l for l in labels]
    labels = ["%s / 5.0" % l if l == "MI score" else l for l in labels]
    num_models = len(metrics.keys())
    width = 0.8 / num_models  # Adjust width based on number of models
    
    # Plot metrics for each model
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, (model_name, metrics) in enumerate(metrics.items()):
        
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
                rotation=90,
            )
            
    # Adjustments to the plot
    ax.set_title("Comparison of models for each metric", fontsize="x-large")
    ax.set_xticks([i + width * (num_models - 1) / 2 for i in range(len(labels))])
    ax.set_xticklabels(labels, fontsize="large", rotation=22.5)
    ax.set_ylabel("Scores", fontsize="x-large")
    ax.legend(fontsize="large", loc="upper right", ncol=4)
    ax.plot([-0.1, len(metrics) - 0.1], [0, 0], color="k")
    fig.tight_layout()
    
    # Save final figure
    figure_path = os.path.join(cfg.RESULT_DIR, "model_comparison.png")
    plt.savefig(figure_path, dpi=300)


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
    logging.info(" --- Loading csv file for multiprocessing split")
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
    