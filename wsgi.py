import logging
from app import app


HOST_ADDRESS = '0.0.0.0'
PORT_NUMBER = 8984


def main():
    set_logger()   
    app.run(debug=False, host=HOST_ADDRESS, port=PORT_NUMBER)


def set_logger(name="CTxAI"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = 0  # prevent logging from propagating to the root logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    

if __name__ == "__main__":
    main()


# *** EXAMPLE JSON REQUEST ***
# {
#     "RAW_DATA_PATH": "data/raw_files/ctxai/intervention_similar_trials.xlsx",
#     "USER_ID": "1234",
#     "PROJECT_ID": "5678",
#     "EMBEDDING_MODEL_ID": "pubmed-bert-sentence",
#     "CLUSTER_DIM_RED_ALGO": "tsne",
#     "CLUSTER_RED_DIM": 2,
#     "PLOT_DIM_RED_ALGO": "tsne",
#     "PLOT_RED_DIM": 2,
#     "CLUSTER_REPRESENTATION_MODEL": null,
#     "CLUSTER_REPRESENTATION_GPT_PROMPT": "I have a topic that contains the following documents: \n[DOCUMENTS]\nThe topic is described by the following keywords: \n[KEYWORDS]\nBased on the information above, extract a short but highly descriptive topic label of at most 5 words.\nMake sure it is in the following format: topic: <topic label>\n"
# }