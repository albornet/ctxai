#!/bin/bash

# Environment specifics
ENV_NAME="ctxai"
PYTHON_VERSION="3.10"
PIP_PACKAGES="numpy pandas scipy scikit-learn matplotlib \
tqdm nltk torch torchdata transformers openai optuna hdbscan"

# Create a new conda environment and activate it
conda create --name $ENV_NAME python=$PYTHON_VERSION -y
source activate $ENV_NAME

# Check if the activation was successful
if [ "$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
    # Install latest version of packages using pip
    pip install -U $PIP_PACKAGES
    echo -e "\nSuccessfully installed all packages in $ENV_NAME.\n\
    Don't forget to activate it by using the following command:\n\
    >>> conda activate $ENV_NAME\n"
else
    # Avoid installing anything if not in the correct environment
    echo "Failed to activate the environment. No packages were installed."
fi
