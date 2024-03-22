# Use the a base image with CuML and CUDA
FROM rapidsai/base:24.02-cuda12.0-py3.10

# Set a working directory
WORKDIR /app

# Set root privileges for installs
USER root

# Installing system build dependencies and additional packages in a single step
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir \
        bertopic==0.16.0 \
        torchmetrics==1.2.0 \
        torchdata==0.7.1 \
        openpyxl==3.1.2 \
        openai==1.9.0 \
        optuna==3.5.0 \
        nltk==3.8.1 \
        kaleido==0.2.1 \
        Flask==3.0.1 \
        gunicorn==21.2.0

# Copy and execute setup script in a single layer
COPY setup_resources.py /app/setup_resources.py
RUN python /app/setup_resources.py

# Come back to original user for runtime
USER rapids

# Command run in interactive mode, for debug (but not used by docker-compose)
CMD ["/bin/bash"]
