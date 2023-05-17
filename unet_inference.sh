#!/bin/bash

# ###############################################################################
#
# Runs inference using the UNet
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
#container="/data/containers/msoe-tensorflow-20.07-tf2-py3.sif"
container="/data/containers/msoe-pytorch-20.07-py3.sif"

# Command to run inside container
command="python -m segm.unet_inference
 --model-path seg_tiny_mask_retrain/checkpoint.pth
  -i /home/nelsonni/laviolette/segmenter/ade20k/ade20k/release_test/testing/
   -o segm/inference_segs/vit_elastic/
   --n_cls 8"


# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} ${command}