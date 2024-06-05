#!/bin/bash

#SBATCH --job-name=ct_project_run_clustering
#SBATCH --partition=shared-gpu,private-teodoro-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96gb
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/users/b/borneta/CT_project/logs/run_clustering/job_%j.txt
#SBATCH --error=/home/users/b/borneta/CT_project/logs/run_clustering/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=ctxai_cluster.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=src/cluster_data.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}
