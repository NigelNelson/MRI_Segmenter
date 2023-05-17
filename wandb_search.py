"""
File that is responsible for defining the metrics that Weights & Biases will use
to conduct its sweep. This file is for the ViT sweep
"""

import sys
import os

import wandb

def main():
 

    wandb.login()

    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'val_mean_iou',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'epochs': {
            'value': 300
        },
        'backbone' : {
            'values' : ['vit_tiny_patch16_384',
                        'vit_small_patch32_384',
                        'vit_small_patch16_384',
                        'vit_base_patch8_384',
                        'vit_base_patch16_384',
                        'vit_base_patch32_384']
        },
        'lr' : {
            'values' : [0.01, 0.005, 0.001, 0.0005]
        },
        'random_crop_size' : {
            'values' : [256, 300, 350, 400]
        }
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="ViT-Test_Sweep")
    print(sweep_id)

if __name__ == "__main__":
    main()
