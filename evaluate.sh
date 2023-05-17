#!/bin/bash

# ###############################################################################
#
# Evalutes the performance of the specified ViT model
#
# ###############################################################################


# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes.  Another option is the 'batch' partition.
#SBATCH --partition=dgx
#SBATCH --account=undergrad_research
#SBATCH --mail-type=ALL

# The number of nodes to request
#SBATCH --nodes=1

# The number of GPUs to request
#SBATCH --gpus=1

# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=16

# Prevent out file from being generated
#SBATCH --output=./segm/outputs/slurm-%j.out


# Create logging directory
now=$(date +"%m-%d-%y|%H:%M:%S")
logdir="./segm/outputs/${now}" 

# Path to container

container="/data/containers/msoe-pytorch-20.07-py3.sif"

# Command to run inside container
command="python -m segm.evaluate --model-path T2_ViT_Tiny_New/checkpoint.pth -i /home/nelsonni/laviolette/segmenter/ade20k/ade20k/release_test/testing/ -o T2_ViT_Tiny_New/"

# Define dataset location
location="~/laviolette/segmenter/ade20k"

# Execute singularity container on node.
DATASET=${location} singularity exec --nv -B /data:/data ${container} ${command}