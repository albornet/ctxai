# CTxAI - Eligibility Criteria

This project aims to provide feedback about eligibility criteria of a new study, based on the history of similar clinical trials.

## Description

Using BERTopic as a backbone, the pipeline performs the following steps:
* First, eligibility criteria are parsed from the CT.gov database and split into a set of individual criteria.
* Then, a language model pre-trained on a clinical corpus embeds all criteria coming from studies that are similar to a new clinical trial.
* After dimensionality reduction, these embedded criteria are clustered using HDBScan, and the clusters are used to compute statistics about the selected similar studies.
* Finally, cluster statistics are computed, and criteria clusters are represented by.

## Getting Started

### Install Dependencies

#### Using Docker (Recommended)

1. Ensure Docker and Docker Compose are installed on your system. You can download them from [Docker's official website](https://www.docker.com/get-started).

2. Clone the repository to your local machine:

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

3. To build and start the project using Docker Compose, run:

   ```
   docker-compose up --build
   ```

   This command builds the Docker images and starts the services defined in your `docker-compose.yml` file. It's a simple way to get the environment set up without manually installing dependencies.

#### Installing Environment with Droplet

If you prefer not to use Docker or if you're working in an environment where Docker is not available, you can install the dependencies manually using Conda:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda on your system if you haven't already.

2. Create a new Conda environment using the `environment_droplet.yml` file provided in the `environment` directory:

   ```
   conda env create -f environment/environmen_droplet.yml
   ```

3. Activate the new environment:

   ```
   conda activate <environment-name>
   ```

### Executing Program

#### Using Docker

To execute the program using Docker, send a POST request with the desired configuration. For example:

```bash
curl -X POST -H "Content-Type: application/json" -d @request.json http://localhost:8984/api/execute
```

Where `request.json` contains the configuration for your execution (key-value pairs that modify any field you would like to modify from the default configuration), similar to the following example (note that "RAW_DATA_PATH", "USER_ID", and "PROJECT_ID" are mandatory fields in the request):

```json
{
    "RAW_DATA_PATH": "data/eligibility_criteria.xlsx",
    "USER_ID": "1234",
    "PROJECT_ID": "5678",
    "EMBEDDING_MODEL_ID": "pubmed-bert-sentence",
    "CLUSTER_DIM_RED_ALGO": "umap",
    "CLUSTER_RED_DIM": 10,
    "PLOT_DIM_RED_ALGO": "tsne",
    "PLOT_RED_DIM": 2,
    "CLUSTER_REPRESENTATION_MODEL": "gpt",
    "CLUSTER_REPRESENTATION_GPT_PROMPT": "I have a topic that contains the following documents: \n[DOCUMENTS]\nThe topic is described by the following keywords: \n[KEYWORDS]\nBased on the information above, extract a short but highly descriptive topic label of at most 5 words.\nMake sure it is in the following format: topic: <topic label>\n"
}
```

The container will execute the pipeline and response with an output like this one:

```json 
{
    "cluster_json_path": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/ec_clustering.json",
    "cluster_raw_ec_list_path": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/raw_ec_list.csv",
    "cluster_visualization_paths": {
        "all": {
            "html": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_all.html",
            "png": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_all.png"
        },
        "top_20": {
            "html": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_top_20.html",
            "png": "results/ctxai/user-1234_project/5678_model/pubmed-bert-sentence/cluster_plot_top_20.png"
        }
    }
}
```

The response defines where the results (included formatted cluster output, as well as interactive visualizations) were saved.

#### Direct Execution

If you've set up your environment using the Conda environment or prefer direct execution:

1. Ensure all configurations are correctly set in `config.yaml` according to your project's needs.

2. From the project's root directory, run:

   ```
   python src/cluster_data.py
   ```

   This command executes the clustering process based on your configuration, performing the entire pipeline from data preprocessing to clustering and visualization, and generates results in a new "results" directory.
