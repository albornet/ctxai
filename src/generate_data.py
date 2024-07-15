import os
import csv
import json
import time
import random
import config
logger = config.CTxAILogger("INFO")
import torchdata.datapipes.iter as dpi
import openai
from openai import OpenAI
from rouge_score import rouge_scorer
from parse_data import CustomJsonParser
from preprocess_utils import ClinicalTrialFilter
from cluster_utils import ClusterOutput
from cluster_data import cluster_data_fn
from config import update_config


GENERATOR_EMBEDDING_MODEL_ID = "pubmed-bert-sentence"
GENERATOR_COND_FILTER_LVL = 2
GENERATOR_ITRV_FILTER_LVL = 1
GENERATOR_NUM_EVALUATED_SAMPLES = 250
GENERATOR_RESULT_PATH = "./ec_generation_evaluation_results.csv"  # for now
GENERATOR_LLM_SYSTEM_PROMPT = "You are an assistant helping to generate eligibility criteria sections for clinical trials."
GENERATOR_LLM_USER_PROMPT = """
I have a clinical trial that includes the following information:
[CT_DATA_TEXT]
Based on the information above, generate the eligibility criteria section for this clinical trial.
Make sure it is in the following format:
Inclusion criteria:
<all inclusion criteria>
Exclustion criteria:
<all exclusion criteria>
"""


def main():
    """ Compare clustering to llm methods for generating eligibility criteria
        section in clinical trials
    """
    # Load evaluation dataset
    cfg = config.get_config()
    ct_data, ec_data = get_evaluated_ct_dataset(cfg["FULL_DATA_PATH"])
    
    # Loop through evaluation dataset to score both methods
    for ct_sample, ec_sample in zip(ct_data, ec_data):
        
        # Clustering method for ec-section generation
        update_config_filters(ct_sample)  # /!\ this updates "cfg" /!\
        try:
            cluster_output = cluster_data_fn(GENERATOR_EMBEDDING_MODEL_ID, write_results=False)
            cluster_ec_section = generate_cluster_ec_section(cluster_output)
            cluster_quality = cluster_output.cluster_metrics["label_free"]["Silhouette score"]
        except Exception as e:
            logger.info("Clustering method failed (%s)" % str(e))
            cluster_ec_section = ""  # default cluster ec-section
            cluster_quality = -1.0  # minimum value
        
        # LLM method for ec-section generation
        ct_path = ct_sample["ct_path"]
        llm_ec_section = generate_llm_ec_section(ct_path)
        
        # Compute and add scores to a csv file
        write_scores_to_csv_file(
            ct_path=ct_path,
            reference=ec_sample,
            cluster_prediction=cluster_ec_section,
            llm_prediction=llm_ec_section,
            cluster_quality=cluster_quality,
        )


def write_scores_to_csv_file(
    ct_path: str,
    reference: str,
    cluster_prediction: str,
    llm_prediction: str,
    cluster_quality: float,
) -> None:
    """ Add results about one evaluated sample for eligibility criterion section
        generation, comparing the clustering method to the llm method
        
    Args:
        ct_path (str): the path of the clinical trial document
        reference (str): the reference text to compare against
        cluster_prediction (str): the cluster-generated prediction
        llm_prediction (str): the LLM-generated prediction
        cluster_quality (float): how good clusters are (silhouette score)
    """
    # Compute scores and average scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    cluster_scores = scorer.score(target=reference, prediction=cluster_prediction)
    llm_scores = scorer.score(target=reference, prediction=llm_prediction)
    cluster_avg_r = (cluster_scores["rouge1"].recall + cluster_scores["rouge2"].recall + cluster_scores["rougeL"].recall) / 3
    cluster_avg_p = (cluster_scores["rouge1"].precision + cluster_scores["rouge2"].precision + cluster_scores["rougeL"].precision) / 3
    cluster_avg_f = (cluster_scores["rouge1"].fmeasure + cluster_scores["rouge2"].fmeasure + cluster_scores["rougeL"].fmeasure) / 3
    llm_avg_r = (llm_scores["rouge1"].recall + llm_scores["rouge2"].recall + llm_scores["rougeL"].recall) / 3
    llm_avg_p = (llm_scores["rouge1"].precision + llm_scores["rouge2"].precision + llm_scores["rougeL"].precision) / 3
    llm_avg_f = (llm_scores["rouge1"].fmeasure + llm_scores["rouge2"].fmeasure + llm_scores["rougeL"].fmeasure) / 3
    
    # Initialize result csv file if first line
    column_names = [
        "CT Path", "Best Method", "Cluster Quality",
        "Cluster ROUGE-1-Recall Score", "Cluster ROUGE-1-Precision Score", "Cluster ROUGE-1-F Score",
        "Cluster ROUGE-2-Recall Score", "Cluster ROUGE-2-Precision Score", "Cluster ROUGE-2-F Score",
        "Cluster ROUGE-L-Recall Score", "Cluster ROUGE-L-Precision Score", "Cluster ROUGE-L-F Score",
        "Cluster ROUGE-Average-R Score", "Cluster ROUGE-Average-P Score", "Cluster ROUGE-Average-F Score",
        "LLM ROUGE-1-Recall Score", "LLM ROUGE-1-Precision Score", "LLM ROUGE-1-F Score",
        "LLM ROUGE-2-Recall Score", "LLM ROUGE-2-Precision Score", "LLM ROUGE-2-F Score",
        "LLM ROUGE-L-Recall Score", "LLM ROUGE-L-Precision Score", "LLM ROUGE-L-F Score",
        "LLM ROUGE-Average-R Score", "LLM ROUGE-Average-P Score", "LLM ROUGE-Average-F Score",
        "Cluster EC Section", "LLM EC Section",
    ]
    
    # Prepare the row to append
    best_method = "Cluster" if cluster_avg_f > llm_avg_f else "LLM"
    row = [
        ct_path, best_method, cluster_quality,
        cluster_scores["rouge1"].recall, cluster_scores["rouge1"].precision, cluster_scores["rouge1"].fmeasure,
        cluster_scores["rouge2"].recall, cluster_scores["rouge2"].precision, cluster_scores["rouge2"].fmeasure,
        cluster_scores["rougeL"].recall, cluster_scores["rougeL"].precision, cluster_scores["rougeL"].fmeasure,
        cluster_avg_r, cluster_avg_p, cluster_avg_f,
        llm_scores["rouge1"].recall, llm_scores["rouge1"].precision, llm_scores["rouge1"].fmeasure,
        llm_scores["rouge2"].recall, llm_scores["rouge2"].precision, llm_scores["rouge2"].fmeasure,
        llm_scores["rougeL"].recall, llm_scores["rougeL"].precision, llm_scores["rougeL"].fmeasure,
        llm_avg_r, llm_avg_p, llm_avg_f,
        cluster_prediction, llm_prediction,
    ]
    
    # Write to CSV file
    file_exists = os.path.isfile(GENERATOR_RESULT_PATH)
    with open(GENERATOR_RESULT_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(column_names)
        writer.writerow(row)


def generate_llm_ec_section(ct_path: str) -> str:
    """ Generate the eligibility criterion section by prompting GPT-3.5-turbo with
        clinical trial information

    Args:
        ct_path (str): path to raw clinical trial data file (json)

    Returns:
        str: generated eligibility criterion section
    """
    # Generate a prompt using data from the clinical trial file
    with open(ct_path, "r", encoding="utf-8") as file:
        ct_raw_dict: dict[str, dict|bool] = json.load(file)
    ct_raw_dict["protocolSection"].pop("eligibilityModule")
    ct_raw_dict.pop("resultsSection", None)
    ct_raw_dict.pop("hasResults", None)
    user_prompt = GENERATOR_LLM_USER_PROMPT.replace("[CT_DATA_TEXT]", str(ct_raw_dict))
    
    # Prompt gpt-3.5-turbo
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            client = OpenAI(api_key=get_openai_api_key())
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": GENERATOR_LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            # Extract the generated EC section from the API response
            ec_section: str = response.choices[0].message.content
            
            return ec_section.strip()
        
        # Expected error handling
        except openai.OpenAIError as e:
            logger.error("Openai error occured (%s), retrying" % str(e))
        
        # Exponential back-off
        time.sleep(2 ** attempt)
        
    return "Failed to generate EC section after multiple attempts."


def get_openai_api_key():
    """ Retrieve the OpenAI API key from the configuration file
    
    Returns:
        str: the OpenAI API key
    """
    cfg = config.get_config()
    api_path = os.path.join(cfg["CLUSTER_REPRESENTATION_PATH_TO_OPENAI_API_KEY"])
    try:
        with open(api_path, "r", encoding="utf-8") as f: return f.read()
    except:
        raise FileNotFoundError(" ".join([
            "To use CLUSTER_REPRESENTATION_MODEL = gpt,",
            "you must have an api-key file at the path defined in the",
            "config under CLUSTER_REPRESENTATION_PATH_TO_OPENAI_API_KEY",
        ]))


def generate_cluster_ec_section(cluster_output: ClusterOutput) -> str:
    """ Generate the clinical trial section for elibibility criteria based on the
        output of the clustering pipeline

    Args:
        cluster_output (ClusterOutput): formatted clustering pipeline output

    Returns:
        str: generated eligibility criteria section
    """
    # Extract clusters and make sure larger clusters are processed first
    clusters = cluster_output.cluster_instances
    clusters.sort(key=lambda cluster: cluster.prevalence, reverse=True)
    
    # Loop through each cluster to select ECs based on medoid proximity and cluster prevalence
    selected_ecs = []
    for cluster in clusters:
        if cluster.cluster_id != -1:
            
            # Randomly select one EC from the closest ones
            closest_ec = cluster.ec_list[0]  # ec closest to cluster medoid
            # if len(selected_ecs) < GENERATOR_MAX_GENERATED_ECS:
            if random.random() < cluster.prevalence:
                selected_ec = closest_ec.raw_text.strip(";").strip()
                selected_ecs.append(selected_ec)
        
    return format_ec_section(selected_ecs)


def format_ec_section(ec_list: list[str]) -> str:
    """ Format a list of eligibility criteria into a text with inclusion criteria
        and exclusion criteria subsections
        
    Args:
        raw_ec_section (str): original eligibility criterion section text
        
    Returns:
        str: transformed section with separate inclusion and exclusion criteria
    """
    inclusion_criteria, exclusion_criteria = [], []
    check_str_in = "inclusion criterion - "
    check_str_ex = "exclusion criterion - "    
    for criterion in ec_list:
        if criterion.lower().startswith(check_str_in):
            inclusion_criteria.append(criterion.split(check_str_in)[-1])
        elif criterion.lower().startswith(check_str_ex):
            exclusion_criteria.append(criterion.split(check_str_ex)[-1])
        else:
            inclusion_criteria.append(criterion)
    
    # Format the output text with the specified sections
    ec_section = "Inclusion criteria:\n" + "\n".join(inclusion_criteria)
    ec_section += "\n\nExclusion criteria:\n" + "\n".join(exclusion_criteria)
                         
    return ec_section


def update_config_filters(ct_data: dict) -> None:
    """ Update running configuration with phase, condition and intervention
        filters for later clustering 

    Args:
        ct_data (dict): clinical trial data
    """
    # Identify phase(s), condition(s), and intervention(s) of the clinical trial
    cond_ids = extract_ids_at_lvl(ct_data["condition_ids"], GENERATOR_COND_FILTER_LVL)
    itrv_ids = extract_ids_at_lvl(ct_data["intervention_ids"], GENERATOR_ITRV_FILTER_LVL)
    to_update = {
        "CHOSEN_PHASES": ct_data["phases"],
        "CHOSEN_COND_IDS": cond_ids,
        "CHOSEN_ITRV_IDS": itrv_ids,
    }
    logger.info("Evaluating ec-generation for ct with: %s" % to_update)
    
    # Make sure that evaluated clinical trial is not considered for clustering 
    to_update.update({
        "ADDITIONAL_NEGATIVE_FILTER": {"ct path": ct_data["ct_path"]},
        "MAX_ELIGIBILITY_CRITERIA_SAMPLES": 75_000,
    })
    
    # Update globally shared configuration with current information
    update_config(request_data=to_update)


def extract_ids_at_lvl(ids: str, lvl: int) -> list[str]:
    """ Extract condition or intervention id(s) up to a certain MeSH tree level

    Args:
        ids (str): raw condition or intervention id(s)
        lvl (int): level up to which id(s) are considered

    Returns:
        list[str]: list of unique ids up to the required level
    """
    flat_ids = [id for id_sublist in ids for id in id_sublist]
    extracted_ids = [".".join(id.split(".")[:lvl]) for id in flat_ids]
    return list(set(extracted_ids))


def get_evaluated_ct_dataset(data_path: str) -> list[dict]:
    """ Build a dataset of evaluated clinical trials from raw json files
    
    Args:
        data_path (str): full path to the raw clinical trial data
        
    Returns:
        [
            list[dict]: evaluated clinical trials data
            list[str]: eligibility criteria section references
        ]
    """
    logger.info("Building ec-generation dataset")
    files = dpi.FileLister(data_path, recursive=True, masks="*.json")
    shuffled_files = dpi.Shuffler(files)
    json_streams = dpi.FileOpener(shuffled_files, encoding="utf-8")
    parsed_cts = CustomJsonParser(json_streams)
    filtered_cts = ClinicalTrialFilter(parsed_cts)
    
    input_data, target_data = [], []
    for ct in filtered_cts:
        if len(input_data) >= GENERATOR_NUM_EVALUATED_SAMPLES: break
        input_data.append(ct[0])  # clinical trial metadata
        target_data.append(ct[1])  # eligibility criteria section
    
    return input_data, target_data


if __name__ == "__main__":
    main()