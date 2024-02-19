# Config
import os
import sys
try:
    import config
except:
    from . import config
import logging
cfg = config.get_config()
logger = logging.getLogger("cluster")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    from cluster_utils import ClusterGeneration, ClusterOutput, get_dim_red_model
    from preprocess_utils import EligibilityCriteriaFilter, set_seeds
except:
    from .cluster_utils import ClusterGeneration, ClusterOutput, get_dim_red_model
    from .preprocess_utils import EligibilityCriteriaFilter, set_seeds

# Embedding
import torch
import torchdata.datapipes.iter as dpi
from torchdata.datapipes import functional_datapipe
from torchdata.dataloader2 import DataLoader2, InProcessReadingService
from transformers import AutoModel, AutoTokenizer

# Clustering and representation
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import OpenAI
from sklearn.feature_extraction.text import CountVectorizer


def main():
    """ If ran as a script, call cluster_data for several models and write
        a summary of results to the output directory
    """
    # Logging
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Generate results using one model, then save the results
    cluster_metrics = {}
    for embed_model_id in cfg["EMBEDDING_MODEL_ID_MAP"].keys():
        logger.info("- Starting with %s" % embed_model_id)
        cluster_output = cluster_data_fn(embed_model_id=embed_model_id)
        cluster_metrics[embed_model_id] = cluster_output.cluster_metrics
        logger.info("- Done with %s" % embed_model_id)
    
    # Plot final results (comparison of different model embeddings)
    output_path = os.path.join(
        cfg["RESULT_DIR"],
        cfg["RAW_INPUT_FORMAT"],
        "user-%s" % cfg["USER_ID"],
        "project-%s" % cfg["PROJECT_ID"],
        "model-comparison.png",
    )
    plot_model_comparison(cluster_metrics, output_path)
    logger.info("Model comparison finished!")


def cluster_data_fn(
    embed_model_id: str,
    user_id: str | None=None,
    project_id: str | None=None,
) -> ClusterOutput:
    """ Cluster eligibility criteria using embeddings from one language model
    """
    # Initialization
    output_base_dir = os.path.join(cfg["RESULT_DIR"], cfg["RAW_INPUT_FORMAT"])
    set_seeds(cfg["RANDOM_STATE"])  # ensure reproducibility
    
    # Generate or load elibibility criterion texts, embeddings, and metadatas
    logger.info("- Retrieving elibility criteria embeddings with %s" % embed_model_id)
    embeddings, raw_txts, metadatas = get_embeddings(embed_model_id)
    n_samples = len(raw_txts)
    
    # Create BERTopic model
    topic_model = BERTopic(
        umap_model=get_dim_red_model(
            cfg["CLUSTER_DIM_RED_ALGO"],
            cfg["CLUSTER_RED_DIM"],
            n_samples,
        ),
        hdbscan_model=ClusterGeneration(),
        vectorizer_model=CountVectorizer(stop_words="english"),
        ctfidf_model=ClassTfidfTransformer(),
        representation_model=get_representation_model(),
        verbose=True,
    )
    
    # Train model and use it for topic evaluation
    logger.info("- Running bertopic algorithm on eligibility criteria embeddings")
    topics, probs = topic_model.fit_transform(raw_txts, embeddings)
    topics = topic_model.reduce_outliers(raw_txts, topics)
    
    # Generate results from the trained model and predictions
    logger.info("- Writing clustering results with bertopic titles")
    return ClusterOutput(
        output_base_dir=output_base_dir,
        embed_model_id=embed_model_id,
        user_id=user_id,
        project_id=project_id,
        raw_txts=raw_txts,
        metadatas=metadatas,
        cluster_info=topic_model.hdbscan_model.cluster_info,
        cluster_representations=topic_model.topic_representations_,
    )
    
    
def get_representation_model():
    """ Get a model to represent each cluster-topic with a title
    """
    if cfg["CLUSTER_REPRESENTATION_MODEL"] == "gpt":
        api_path = os.path.join("data", "api-key-risklick.txt")
        try:
            with open(api_path, "r") as f: api_key = f.read()
        except:
            raise FileNotFoundError("You must have an api-key at %s" % api_path)
        client = OpenAI(api_key=api_key)
        return OpenAI(
                client=client,
                model="gpt-3.5-turbo",
                exponential_backoff=True, chat=True,
                prompt=cfg["CLUSTER_REPRESENTATION_GPT_PROMPT"],
            )
    # TODO N-KEYWORDS
    else:
        return None


def get_embeddings(embed_model_id: str) -> tuple[np.ndarray, list[str], dict]:
    """ Generate and save embeddings or load them from a previous run
    """
    preprocessed_dir = os.path.join(
        cfg["BASE_DATA_DIR"],
        cfg["PREPROCESSED_SUBDIR"],
        cfg["RAW_INPUT_FORMAT"],
    )
    postprocessed_dir = os.path.join(
        cfg["BASE_DATA_DIR"],
        cfg["POSTPROCESSED_SUBDIR"],
        cfg["RAW_INPUT_FORMAT"],
    )
    if cfg["LOAD_EMBEDDINGS"]:
        embeddings, raw_txts, metadatas = load_embeddings(
            output_dir=postprocessed_dir,
            embed_model_id=embed_model_id,
        )
    else:
        embeddings, raw_txts, metadatas = generate_embeddings(
            input_dir=preprocessed_dir,
            embed_model_id=embed_model_id,
        )
        save_embeddings(
            output_dir=postprocessed_dir,
            embed_model_id=embed_model_id,
            embeddings=embeddings,
            raw_txts=raw_txts,
            metadatas=metadatas,
        )
    return embeddings.numpy(), raw_txts, metadatas


def generate_embeddings(
    input_dir: str,
    embed_model_id: str,
) -> tuple[torch.Tensor, list[str], list[dict]]:
    """ Generate a set of embeddigns from data in a given input directory, using
        a given model
    """
    # Load model and data pipeline
    logger.info("--- Running model to generate embeddings")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer, pooling_fn = get_model_pipeline(embed_model_id)
    model = model.to(device)
    ds = get_dataset(input_dir, tokenizer)
    rs = InProcessReadingService()
    dl = DataLoader2(ds, reading_service=rs)
    
    # Go through data pipeline
    raw_txts, metadatas = [], []
    embeddings = torch.empty((0, model.config.hidden_size))
    for i, (encoded, raw_txt, metadata) in tqdm(
        iterable=enumerate(dl), leave=False,
        desc="Processing eligibility criteria dataset"
    ):
        
        # Compute embeddings for this batch
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        ec_embeddings = pooling_fn(encoded, outputs)
        
        # Record model outputs (tensor), input texts and corresponding labels
        embeddings = torch.cat((embeddings, ec_embeddings.cpu()), dim=0)
        raw_txts.extend(raw_txt)
        metadatas.extend(metadata)
        if len(raw_txts) > cfg["MAX_ELIGIBILITY_CRITERIA_SAMPLES"]: break
    
    # Make sure gpu memory is made free for report_cluster
    torch.cuda.empty_cache()
    
    # Return embeddings, as well as raw text data and some metadata 
    return embeddings, raw_txts, metadatas
    

def save_embeddings(output_dir, embed_model_id, embeddings, raw_txts, metadatas):
    """ Simple saving function for model predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "embeddings_%s.pt" % embed_model_id)
    torch.save(embeddings, ckpt_path)
    with open(os.path.join(output_dir, "raw_txts.pkl"), "wb") as f:
        pickle.dump(raw_txts, f)
    with open(os.path.join(output_dir, "metadatas.pkl"), "wb") as f:
        pickle.dump(metadatas, f)


def load_embeddings(output_dir, embed_model_id):
    """ Simple loading function for model predictions
    """
    logger.info("--- Loading embeddings from previous run")
    ckpt_path = os.path.join(output_dir, "embeddings_%s.pt" % embed_model_id)
    embeddings = torch.load(ckpt_path)
    with open(os.path.join(output_dir, "raw_txts.pkl"), "rb") as f:
        raw_txts = pickle.load(f)
    with open(os.path.join(output_dir, "metadatas.pkl"), "rb") as f:
        metadatas = pickle.load(f)
    return embeddings, raw_txts, metadatas


def get_model_pipeline(embed_model_id: str):
    """ Select a model and the corresponding tokenizer and embed function
    """
    # Model generates token-level embeddings, and output [cls] (+ linear + tanh)
    if "-sentence" not in embed_model_id:
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
    model_str = cfg["EMBEDDING_MODEL_ID_MAP"][embed_model_id]
    model = AutoModel.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    return model, tokenizer, pooling_fn


def get_dataset(data_dir, tokenizer):
    """ Create a pipe from file names to processed data, as a sequence of basic
        processing functions, with sharding implemented at the file level
    """
    ds = dpi.FileLister(
        data_dir, recursive=True, masks=cfg["PREPROCESSED_FILE_MASK"],
    )\
        .open_files(mode="t", encoding="utf-8")\
        .parse_csv()\
        .sharding_filter()\
        .filter_eligibility_criteria()\
        .batch(batch_size=cfg["EMBEDDING_BATCH_SIZE"])\
        .tokenize(tokenizer=tokenizer)
    return ds


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
        
     
def plot_model_comparison(metrics: dict, output_path: str):
    """ Generate a comparison plot between models, based on how model embeddings
        produce good clusters
    """
    # Load, parse, and format data
    to_plot = [
        "Silhouette score", "DB index", "Dunn index", "MI score",
        "AMI score", "Homogeneity", "Completeness", "V measure",
    ]
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
    
    # Save final figure
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)

    
if __name__ == "__main__":
    main()
    