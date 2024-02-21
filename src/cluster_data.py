# Config
import os
import sys
try:
    import config
except:
    from . import config

import logging
logger = logging.getLogger("CTxAI")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Utils
import matplotlib.pyplot as plt
try:
    from cluster_utils import ClusterGeneration, ClusterOutput, get_dim_red_model, set_seeds
    from preprocess_utils import get_embeddings
except:
    from .cluster_utils import ClusterGeneration, ClusterOutput, get_dim_red_model, set_seeds
    from .preprocess_utils import get_embeddings

# Clustering and representation
from openai import OpenAI as OpenAIClient
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import OpenAI
from sklearn.feature_extraction.text import CountVectorizer


def main():
    """ If ran as a script, call cluster_data for several models and write
        a summary of results to the output directory
    """
    # Load current configuration
    cfg = config.get_config()
    
    # Logging
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    # formatter = logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Generate results using one model, then save the results
    cluster_metrics = {}
    for embed_model_id in cfg["EMBEDDING_MODEL_ID_MAP"].keys():
        logger.info("Starting with %s" % embed_model_id)
        cluster_output = cluster_data_fn(embed_model_id=embed_model_id)
        cluster_metrics[embed_model_id] = cluster_output.cluster_metrics
        logger.info("Done with %s" % embed_model_id)
    
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
    cfg = config.get_config()
    output_base_dir = os.path.join(cfg["RESULT_DIR"], cfg["RAW_INPUT_FORMAT"])
    set_seeds(cfg["RANDOM_STATE"])  # try to ensure reproducibility
    
    # Generate or load elibibility criterion texts, embeddings, and metadatas
    logger.info("Retrieving elibility criteria embeddings with %s" % embed_model_id)
    embeddings, raw_txts, metadatas = get_embeddings(embed_model_id)
    n_samples = len(raw_txts)
    
    # Create BERTopic model
    topic_model = BERTopic(
        top_n_words=cfg["CLUSTER_REPRESENTATION_TOP_N_WORDS_PER_TOPIC"],
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
    logger.info("Running bertopic algorithm on eligibility criteria embeddings")
    topics, _ = topic_model.fit_transform(raw_txts, embeddings)
    topics = topic_model.reduce_outliers(raw_txts, topics)
    
    # Generate results from the trained model and predictions
    logger.info("Writing clustering results with bertopic titles")
    return ClusterOutput(
        output_base_dir=output_base_dir,
        topic_model=topic_model,
        raw_txts=raw_txts,
        metadatas=metadatas,
        embed_model_id=embed_model_id,
        user_id=user_id,
        project_id=project_id,
    )
    
    
def get_representation_model():
    """ Get a model to represent each cluster-topic with a title
    """
    # Get current configuration
    cfg = config.get_config()
    
    # Prompt chat-gpt with keywords and document content
    if cfg["CLUSTER_REPRESENTATION_MODEL"] == "gpt":
        api_path = os.path.join("data", "api-key.txt")
        try:
            with open(api_path, "r") as f: api_key = f.read()
        except:
            raise FileNotFoundError(" ".join([
                "To use CLUSTER_REPRESENTATION_MODEL = gpt,",
                "you must have an api-key at %s" % api_path,
            ]))
        return OpenAI(
                client=OpenAIClient(api_key=api_key),
                model="gpt-3.5-turbo",
                exponential_backoff=True, chat=True,
                prompt=cfg["CLUSTER_REPRESENTATION_GPT_PROMPT"],
                generator_kwargs={
                    "seed": cfg["RANDOM_STATE"],
                    "temperature": 0,
                },
            )
        
    # BERTopic default, which is a sequence of top-n keywords
    else:
        return None
    
     
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
    