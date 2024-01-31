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
    plotly=5.18.0 \
    kaleido==0.2.1 \
    Flask==3.0.1 \
    gunicorn==21.2.0

# Download and cache the pretrained model resources
COPY setup_resources.py /app/setup_resources.py
RUN python /app/setup_resources.py

# Command run in interactive mode, for debug (but not used by docker-compose)
CMD ["/bin/bash"]
