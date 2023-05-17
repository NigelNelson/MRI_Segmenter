#!/bin/bash

# ###############################################################################
#
# Run's the registration script for the ViT
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
#SBATCH --gpus=2

# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=16

# Prevent out file from being generated
#SBATCH --output=./segm/outputs/slurm-%j.out

#SBATCH --nodelist=dh-dgx1-1


# Create logging directory
now=$(date +"%m-%d-%y|%H:%M:%S") 

# Path to container
container="/data/containers/msoe-pytorch-20.07-py3.sif"

# Command to run inside container
command="python -m segm.ViT_Register --log-dir B10_ViT_Registration
 --dataset ade20k --no-resume --backbone vit_tiny_patch16_384 --decoder mask_transformer
  --batch-size 8 --epochs 250 --weight-decay 0.001 -lr 0.001
   --mri_sequence_name mri_slice_double_DWI_b10_reg --wandb_run_name B10"

# mri_slice_double_DWI_b1400_reg
# Define dataset location
location="~/laviolette/segmenter/ade20k"

# Execute singularity container on node.
DATASET=${location} singularity exec --nv -B /data:/data ${container} ${command}