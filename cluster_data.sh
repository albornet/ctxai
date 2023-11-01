#!/bin/bash

#SBATCH --job-name=cluster_data
#SBATCH --partition=private-teodoro-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=41
#SBATCH --mem=48gb
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/job_%j.txt
#SBATCH --error=logs/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=ctxai-image.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=cluster_data.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT} --hpc
