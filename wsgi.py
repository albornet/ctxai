import sys
import logging
logger = logging.getLogger("cluster")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
from app import app


HOST_ADDRESS = '0.0.0.0'
PORT_NUMBER = 8984


def main():
    app.run(debug=False, host=HOST_ADDRESS, port=PORT_NUMBER)


if __name__ == "__main__":
    main()


# # Typical request
# {
#   "user_id": "1234",
#   "project_id": "5678",
#   "model_id": "pubmed-bert-sentence",
#   "raw_data_path": "data/raw_files/ctxai/intervention_similar_trials.xlsx",
#   "cluster_summarization_params":
#     {
#       "reduced_dim_plot": 2,  # data dimensionality for data visualization; either 2 or 3
#       "reduced_dim_cluster": 2,  # data dimensionality for clustering algorithm; None for no reduction
#       "method": "gpt",  # "closest", "shortest", "gpt"
#       "n_representants": 20,  # cluster titles generated from n_representants samples closest to cluster medoid
#       "gpt_system_prompt": "You are an expert in the fields of clinical trials and eligibility criteria. You express yourself succintly, i.e., less than 5 words per response.",
#       "gpt_user_prompt_intro": "I will show you a list of eligility criteria. They share some level of similarity. Your task is to create a single, short tag that effecively summarizes the list. Importantly, the tag should be very short and concise, i.e., under 5 words. You can use medical abbreviations and you should avoid focusing on details or outliers. Write only your answer, starting with 'Inclusion - ' or 'Exclusion - '. Summarize the criteria list as a whole, choosing either 'Inclusion' or 'Exclusion' for your tag, and not both. Here is the list of criteria (each one is on a new line):\n"
#     }
# }