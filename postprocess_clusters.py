import os
import csv
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize


DATA_DIR = os.path.join("results", "dict")
CLUSTER_INFO_PATH = os.path.join(DATA_DIR, "cluster_report_stat.csv")
CLUSTER_DATA_PATH = os.path.join(DATA_DIR, "cluster_report_text.csv")
OUTPUT_PATH = CLUSTER_DATA_PATH.replace("_text.csv", "_list.csv")


def main():
    cluster_labels = load_labels(CLUSTER_INFO_PATH)
    raw_data = load_data(CLUSTER_DATA_PATH)
    stop_words = find_stop_words(raw_data)
    cleaned_data = remove_stop_words(raw_data, stop_words)
    cluster_terms = find_cluster_terms(cleaned_data)
    samples = list_samples_by_clusters(raw_data, cleaned_data, cluster_labels, cluster_terms)
    write_to_csv_file(samples)
    
    
def load_labels(path: str) -> dict[int, str]:
    """ Load cluster labels
    """
    df = pd.read_csv(path, header=0)
    labels_dict = df['Cluster representative'].dropna().to_dict()
    return labels_dict


def load_data(path: str) -> dict[int, list[str]]:
    """ Load raw data as a dictionary
    """
    df = pd.read_csv(path, header=0).fillna('PANDAS_NAN')
    df_dict = df.to_dict(orient='list')
    for k, text_list in df_dict.items():
        df_dict[k] = [text for text in text_list if text != 'PANDAS_NAN']
    indexed_df_dict = {i: v for i, v in enumerate(df_dict.values())}
    return indexed_df_dict  # to match load_labels


def find_stop_words(data_dict: dict, top_k: int=50) -> list[str]:
    """ Identify up to top-k words that are very common amongst the whole data
    """
    global_term_counts = Counter()
    for text_list in data_dict.values():
        tokenized = [word_tokenize(text) for text in text_list]
        for token_list in tokenized:
            global_term_counts.update(token_list)
    stop_words = [term for term, _ in global_term_counts.most_common(top_k)]
    return stop_words


def find_cluster_terms(data_dict: dict, top_k: int=10) -> dict[int, list[str]]:
    """ Identify up to top-k terms that are popular for each cluster
    """
    common_words = {}
    for k, v_list in data_dict.items():
        tokenized = [word_tokenize(v) for v in v_list]
        term_counts = Counter()
        for token_list in tokenized:
            term_counts.update(token_list)
        common_words[k] = [t for t, _ in term_counts.most_common(top_k) if len(t) > 3]
    return common_words


def remove_stop_words(data_dict: dict[int, list[str]],
                      stop_words: list[str]
                      ) -> dict[int, list[str]]:
    """ Remove very common words from all texts in the data dictionary
    """
    new_data_dict = dict(data_dict)
    for k, v_list in new_data_dict.items():
        new_data_dict[k] = [remove_stop_words_fn(v, stop_words) for v in v_list]
    return new_data_dict
    
    
def remove_stop_words_fn(text: str, stop_words: list[str]) -> str:
    """ Function to apply to a string to remove very common words
    """
    return ' '.join([w for w in word_tokenize(text) if w not in stop_words])


def list_samples_by_clusters(raw_data: dict[int, list[str]],
                             cleaned_data: dict[int, list[str]],
                             cluster_labels: dict[int, str],
                             cluster_terms: dict[int, list[str]],
                             ) -> list[list[str]]:
    # Go through all sentences of the data
    sample_rows = []
    for cluster_id, text_list in cleaned_data.items():
        for text_id, text in enumerate(text_list):
            
            # Append new sample row with newly identified clusters
            clusters = find_clusters(text, cluster_labels, cluster_terms, cluster_id)
            raw_text = [raw_data[cluster_id][text_id]]
            sample_rows.append([raw_text] + clusters)
    
    # Return final list of csv rows)
    # sample_rows = [r for r in sample_rows if len(r) > 2]  # DEBUG!!!!
    return sample_rows


def find_clusters(text: str,
                  cluster_labels: dict[int, str],
                  cluster_terms: dict[int, list[str]],
                  original_cluster_id: int,
                  ) -> list[str]:
    # Initialize text and attributed cluster list
    text_words = word_tokenize(text)
    attributed_clusters = [cluster_labels[original_cluster_id]]  # original cluster
    other_cluster_ids = [i for i in cluster_labels.keys() if i != original_cluster_id]
    for other_cluster_id in other_cluster_ids:
        
        # Check for common words with any other cluster
        check_words = cluster_terms[other_cluster_id]
        count = sum(1 for word in check_words if word in text_words)
        if count >= 5:
            attributed_clusters.append(cluster_labels[other_cluster_id])
    
    # Return final list of cluster (note: not cluster_ids)
    return attributed_clusters


def write_to_csv_file(sample_rows: list[list[str]]) -> None:
    """ Write sample rows to a file
    """
    # Create headers
    max_columns = max(len(sublist) for sublist in sample_rows)
    headers = ["Sample text", "Original cluster"]
    for i in range(2, max_columns + 1):
        headers.append(f"Added cluster {i-1}")

    # Write to csv file
    with open(OUTPUT_PATH, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for sublist in sample_rows:
            writer.writerow(sublist)
                

if __name__ == "__main__":
    main()
    