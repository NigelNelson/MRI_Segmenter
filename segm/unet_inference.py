import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent) + '/mri_histology_toolkit')
sys.path.append(str(Path.cwd().parent.parent) + '/homologous_point_prediction')

# from homologous_point_prediction.evaluate import evaluate, evaluate_rotation

from mri_histology_toolkit.data_loader import DataLoader

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
@click.option("--n_cls", type=str)
def main(model_path, input_dir, output_dir, gpu, n_cls):
    ptu.set_gpu_mode(gpu)

    # model_dir = Path(model_path).parent
    model_dir = Path(model_path)
    model = UNet(n_channels=1, n_classes=n_cls, bilinear=True)
    data = torch.load(model_dir, map_location=ptu.device)
    model.load_state_dict(data)
    model.to(ptu.device)

    #########################################################################
    # Below can be used if you want a ~fancier output PIL Image

    # cat_names = ['background',
    #             '1',
    #             '2',
    #             '3',
    #             '4',
    #             '5',
    #             '6',
    #             '7'
    #             ]
    # cat_colors = {
    #     0: torch.tensor([0.0, 0.0, 0.0]).float(), 
    #     1: torch.tensor([255.0, 51.0, 51.0]).float() / 255.0, # red
    #     2: torch.tensor([255.0, 128.0, 0.0]).float() / 255.0, # orange
    #     3: torch.tensor([255.0, 255.0, 0.0]).float() / 255.0, # yellow
    #     4: torch.tensor([0.0, 255.0, 0.0]).float() / 255.0, # green
    #     5: torch.tensor([0.0, 255.0, 255.0]).float() / 255.0, # cyan
    #     6: torch.tensor([0.0, 0.0, 255.0]).float() / 255.0, # blue
    #     7: torch.tensor([255.0, 0.0, 255.0]).float() / 255.0, # pink
    # }

    # input_dir = Path(input_dir)
    # output_dir = Path(output_dir)
    # output_dir.mkdir(exist_ok=True)

    # list_dir = list(input_dir.iterdir())
    # for filename in tqdm(list_dir, ncols=80):
    #     pil_im = Image.open(filename).copy()
    #     im = F.pil_to_tensor(pil_im).float() / 255
    #     im = F.normalize(im, normalization["mean"], normalization["std"])
    #     im = im.to(ptu.device).unsqueeze(0)

    #     im_meta = dict(flip=False)
    #     logits = inference(
    #         model,
    #         [im],
    #         [im_meta],
    #         ori_shape=im.shape[2:4],
    #         window_size=variant["inference_kwargs"]["window_size"],
    #         window_stride=variant["inference_kwargs"]["window_stride"],
    #         batch_size=2,
    #     )
    #     seg_map = logits.argmax(0, keepdim=True)
    #     seg_rgb = seg_to_rgb(seg_map, cat_colors)
    #     seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
    #     pil_seg = Image.fromarray(seg_rgb[0])

    #     pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
    #     pil_blend.save(output_dir / filename.name)
        
    #     save_name = str(output_dir) + "/" + "original_" + str(filename.name)

    #     pil_im.save(save_name)

    #########################################################################


    data_loader = DataLoader(config_path="/home/nelsonni/laviolette/method_analysis/configs/seg_test_config.json")
    for data_dict in data_loader:

        unmasked_mri = data_dict["unmasked_mri"]

        # im = torch.from_numpy(unmasked_mri).unsqueeze(0)
        
        unmasked_mri = cv2.cvtColor(unmasked_mri, cv2.COLOR_GRAY2RGB)

        im = torch.from_numpy(unmasked_mri)
        im = F.normalize(im, 0.5, 0.5)
        im = im.permute(2, 0, 1)
    

        im = im.to(ptu.device)

        im_meta = dict(flip=False)
        logits = inference(
            model,
            [im],
            [im_meta],
            ori_shape=im.shape[1:3],
            window_size=512,
            window_stride=512,
            batch_size=1,
        )

        
        seg_map = logits.argmax(0, keepdim=True)


        plt.imshow(im[0].cpu(), cmap='gray')
        plt.imshow(seg_map[0].cpu(), alpha=0.5)
        file_name = data_dict['patient'] +'_' + data_dict['slide'] + '.jpg'

        plt.savefig(output_dir / file_name)

        # seg_rgb = seg_to_rgb(seg_map, cat_colors)
        # seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        # pil_seg = Image.fromarray(seg_rgb[0])

        # file_name = data_dict['patient'] +'_' + data_dict['slide'] + '.jpg'
        # pil_im = Image.fromarray(np.uint8(norm_image))

        # pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
        # pil_blend.save(output_dir / file_name)
        


if __name__ == "__main__":
    main()
