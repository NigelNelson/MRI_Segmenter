#!/bin/bash

# ###############################################################################
#
# Bash script to run training on ROSIE with horovod
# To run on Rosie, run `sbatch ./train.sh` from the project home directory
#
# ###############################################################################


# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes.  Another option is the 'batch' partition.
#SBATCH --partition=teaching
#SBATCH --account=undergrad_research
#SBATCH --mail-type=ALL

# The number of nodes to request
#SBATCH --nodes=1

# The number of GPUs to request
#SBATCH --gpus=0

# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=16

# Prevent out file from being generated
#SBATCH --output=./segm/outputs/slurm-%j.out


# Create logging directory
now=$(date +"%m-%d-%y|%H:%M:%S") 

# Path to container
#container="/data/containers/msoe-tensorflow-20.07-tf2-py3.sif"
container="/data/containers/msoe-pytorch-20.07-py3.sif"

# Command to run inside container
command="python wandb_search_UNet.py"

# Define dataset location
location="~/laviolette/segmenter/ade20k"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} ${command}
