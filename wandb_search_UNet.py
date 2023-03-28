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
        'bilinear' : {
            'values' : [True, False]
        },
        'lr' : {
            'values' : [0.01, 0.005, 0.001, 0.0005]
        },
        'random_crop_size' : {
            'values' : [256, 300, 350, 400]
        },
        'optim' : {
            'values' : ['Adam', 'SGD']
        },
        'weight_decay' : {
            'values' : [0, 1e-1, 1e-2, 1e-3, 1e-4]
        }
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="UNet-Test_Sweep")
    print(sweep_id)

if __name__ == "__main__":
    main()
