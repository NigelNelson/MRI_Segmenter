import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent) + '/mri_histology_toolkit')
sys.path.append(str(Path.cwd().parent.parent) + '/homologous_point_prediction')

# from homologous_point_prediction.evaluate import evaluate, evaluate_rotation

from mri_histology_toolkit.data_loader import DataLoader

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


@click.command()
@click.option("--model-path", type=str)
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
@click.option("--gpu/--cpu", default=True, is_flag=True)
def main(model_path, input_dir, output_dir, gpu):
    ptu.set_gpu_mode(gpu)

    model_dir = Path(model_path).parent
    model, variant = load_model(model_path)
    model.to(ptu.device)

    normalization_name = variant["prostates_dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

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


    data_loader = DataLoader(config_path="/home/nelsonni/laviolette/method_analysis/configs/test_config.json")
    for data_dict in data_loader:
        if True:
            print(data_dict["patient"], data_dict["slide"])
            print(len(data_dict["hist_points"]))
            print(len(data_dict["mri_points"]))

        unmasked_mri = data_dict["unmasked_mri"]

        unmasked_mri = cv2.cvtColor(unmasked_mri, cv2.COLOR_GRAY2RGB)

        im = torch.from_numpy(unmasked_mri)
        # im = im.unsqueeze(0)

        im = F.normalize(im, normalization["mean"], normalization["std"])

        im = im.permute(2, 0, 1)

        print('IMs SHAPE:', im.shape)

        im = im.to(ptu.device)

        im_meta = dict(flip=False)
        logits = inference(
            model,
            [im],
            [im_meta],
            ori_shape=im.shape[1:3],
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=variant["inference_kwargs"]["window_stride"],
            batch_size=2,
        )
        seg_map = logits.argmax(0, keepdim=True)
        seg_rgb = seg_to_rgb(seg_map, cat_colors)
        seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        pil_seg = Image.fromarray(seg_rgb[0])

        file_name = data_dict['patient'] +'_' + data_dict['slide'] + '.jpg'
        pil_im = im = Image.fromarray(np.uint8(unmasked_mri))

        pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
        pil_blend.save(output_dir / file_name)


if __name__ == "__main__":
    main()
