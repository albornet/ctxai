import json

DATA_DIR = "data_ctgov"
DATA_SUBDIR = "cond-lvl-2_itrv-lvl-1_cluster-tsne-2_plot-tsne-2/results"
COND_IDS = ["C01", "C04", "C14", "C20"]
MODEL_IDS = ["bert", "bert-sentence", "pubmed-bert-token", "pubmed-bert-sentence"]


def main():
    """ Print some cluster information useful for the manuscript
    """
    for cond_id in COND_IDS:
        for model_id in MODEL_IDS:
            print_data(cond_id, model_id)
            

def print_data(cond_id: str, model_id: str) -> None:
    """ Print useful information about clusters using a given condition and model
    """
    path = "%s/ctgov-%s/%s/%s/ec_clustering.json" % (DATA_DIR, cond_id, DATA_SUBDIR, model_id)
    with open(path, 'r') as file:
        data = json.load(file)
        
    clustered_cts = set()
    for cluster in data["cluster_instances"]:
        for ec in cluster["ec_list"]:
            clustered_cts.add(ec["ct_id"])
    
    if model_id == MODEL_IDS[0]:
        print("Condition id %s:" % cond_id)
        print("Number of CTs: %i" % len(clustered_cts))
        print("Number of ECs: %i" % data["cluster_metrics"]["n_samples"])
        
    print("Model id: %s" % model_id)
    print("Number of clusters: %i" % len(data["cluster_instances"]))
    
    if model_id == MODEL_IDS[-1]:
        print()


if __name__ == "__main__":
    main()