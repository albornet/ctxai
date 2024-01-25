# Use a Conda-based image as the base
FROM continuumio/miniconda3

# Set a working directory
WORKDIR /app

# Copy the environment setup script and requirements file into the container
COPY setup_env.sh .
COPY pip_requirements.txt .

# Run the environment setup script
RUN chmod +x ./setup_env.sh && ./setup_env.sh

# Copy the local src directory, drugbank_data, model, pretrained_configs directory, app and wsgi Python files into the container
COPY src/ /app/src/
COPY app.py .
COPY wsgi.py .

# Set the environment variable for Flask to run in production mode and Expose the port the app runs on
ENV FLASK_ENV=production
EXPOSE 8984

# Command to run the application within the Conda environment
CMD ["sh", "-c", "source activate ctxai && gunicorn --worker-class gthread --threads 20 -b 0.0.0.0:8984 wsgi:app"]
