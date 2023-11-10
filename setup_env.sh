#!/bin/bash

# Environment specifics
ENV_NAME="ctxai"
PYTHON_VERSION="3.10"
PIP_PACKAGES="torch numpy pandas scipy scikit-learn matplotlib tqdm nltk torchdata torchmetrics transformers openai optuna hdbscan openpyxl"

# Create a new conda environment and activate it
conda create --name $ENV_NAME python=$PYTHON_VERSION -y
source activate $ENV_NAME

# Check if the activation was successful
if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then

    # Install latest version of pip packages
    pip install -U $PIP_PACKAGES

    # Install cuml last with conda
    conda install -n $ENV_NAME cuml -c rapidsai -c conda-forge -c nvidia -y

    # Success message
    echo -e "\nSuccessfully installed all packages in $ENV_NAME.\n\
    Don't forget to activate it by using the following command:\n\
    >>> conda activate $ENV_NAME\n"
else
    # Avoid installing anything if still in the base environment
    echo "Failed to activate the environment. No packages were installed."
fi
