import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent.parent) + '/mri_histology_toolkit')
sys.path.append(str(Path.cwd().parent.parent) + '/homologous_point_prediction')

# from homologous_point_prediction.evaluate import evaluate, evaluate_rotation

# from mri_histology_toolkit.data_loader import DataLoader

from segm.data_processing.seg_data_loader import SegDataLoader
from segm.data_processing.transforms import ToColor

from torch.utils.data.distributed import DistributedSampler

import matplotlib.pyplot as plt

import torch

import cv2

import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

import segm.utils.torch as ptu

from segm.data.utils import STATS
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb

from segm.model.factory import load_model
from segm.model.utils import inference

from segm.unet.unet_model import UNet


@click.command()
@click.option("--model-path", type=str)
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
@click.option("--gpu/--cpu", default=True, is_flag=True)
def main(model_path, input_dir, output_dir, gpu):
    ptu.set_gpu_mode(gpu)

    # model_dir = Path(model_path).parent
    model_dir = Path(model_path)
    model, variant = load_model(model_dir)
    model.to(ptu.device)


    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    test_augs = ToColor()

    test_data_config = "/home/nelsonni/laviolette/method_analysis/configs/seg_test_config.json"

    test_loader = SegDataLoader(test_data_config, transform=test_augs)

    test_loader = DataLoader(test_loader, batch_size=1,
                        shuffle=False, sampler=DistributedSampler(test_loader))

    for batch in test_loader:

        im = batch['mri'].to(ptu.device) # Get MRI

        im_meta = dict(flip=False)
        logits = inference(
            model,
            [im],
            [im_meta],
            ori_shape=(512, 512),
            window_size=512,
            window_stride=512,
            batch_size=1,
        )

        
        seg_map = logits.argmax(0, keepdim=True)


        # Saves an MRI image with the prediction mask semi-transparent on top
        plt.imshow(im[0].cpu(), cmap='gray')
        plt.imshow(seg_map[0].cpu(), alpha=0.5)
        file_name = data_dict['patient'] +'_' + data_dict['slide'] + '.jpg'

        plt.savefig(output_dir / file_name)
        


if __name__ == "__main__":
    main()
