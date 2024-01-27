# Use an nvidia + torch + cuml image as the base
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set a working directory
WORKDIR /app

# Install additional packages (including gunicorn)
RUN pip install \
    torchmetrics==1.2.0 \
    openpyxl==3.1.2 \
    openai==1.9.0 \
    optuna==3.5.0 \
    transformers==4.37.0 \
    nltk==3.8.1 \
    hdbscan==0.8.33 \
    Flask==3.0.1 \
    gunicorn==21.2.0
