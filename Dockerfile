# Use an nvidia + torch + cuml image as the base
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set a working directory
WORKDIR /app

# Install additional packages
RUN pip install \
    torchmetrics==1.2.0 \
    openpyxl==3.1.2 \
    openai==1.9.0 \
    optuna==3.5.0 \
    transformers==4.37.0 \
    nltk==3.8.1 \
    hdbscan==0.8.33 \
    Flask==3.0.1

# Install gunicorn
RUN pip install gunicorn

# Copy necessary files and folders
COPY app.py .
COPY wsgi.py .
COPY src ./src
COPY data/raw_files/ctxai ./data/raw_files/ctxai

# Set the environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Expose the port the app runs on
EXPOSE 8984

# # The code to run when container is started (gunicorn wsgi:app -b 0.0.0.0:8000)
# CMD ["gunicorn", "-b", "0.0.0.0:8000", "wsgi:app"]
