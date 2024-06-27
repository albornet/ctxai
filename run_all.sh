#!/bin/bash

#SBATCH --job-name=ct_project_run_all
#SBATCH --partition=shared-gpu,private-teodoro-gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/users/b/borneta/CT_project/logs/run_all/job_%j.txt
#SBATCH --error=/home/users/b/borneta/CT_project/logs/run_all/job_%j.err

REGISTRY=/home/users/b/borneta/sif
SIF=ctxai_cluster.sif
IMAGE=${REGISTRY}/${SIF}
SCRIPT=src/run_all.py

srun apptainer run --nv ${IMAGE} python ${SCRIPT}
