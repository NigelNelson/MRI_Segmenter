import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent) + '/mri_histology_toolkit')
sys.path.append(str(Path.cwd().parent.parent) + '/homologous_point_prediction')
sys.path.append(str(Path.cwd().parent))
# from homologous_point_prediction.evaluate import evaluate, evaluate_rotation

from torch.utils.data import DataLoader

from segm.data_processing.seg_data_loader import SegDataLoader
from segm.data_processing.transforms import RandomCrop, RandomFlip, ElasticTransform, ToColor, ToGray


import matplotlib.pyplot as plt
from segm.metrics import gather_data, compute_metrics

import torch

import cv2

import yaml

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

import torch

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
# You can comment out this line if you are passing tensors of equal shape
# But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(0)  # BATCH x 1 x H x W => BATCH x H x W

    outputs = outputs.long()
    labels = labels.long()

    intersection = (outputs & labels).float().sum((0, 1))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((0, 1))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.7), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


@click.command()
@click.option("--model-path", type=str)
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
@click.option("--gpu/--cpu", default=True, is_flag=True)
def main(model_path, input_dir, output_dir, gpu):
    ptu.set_gpu_mode(gpu)

    # model_dir = Path(model_path).parent
    model_dir = Path(model_path)
    # model = UNet(n_channels=1, n_classes=8, bilinear=True)
    # data = torch.load(model_dir, map_location=ptu.device)
    # model.load_state_dict(data)
    print(model_dir)
    model, variant = load_model(model_dir)

    model.to(ptu.device)

    amp_autocast = torch.cuda.amp.autocast

    # normalization_name = variant["dataset_kwargs"]["normalization"]
    # normalization = STATS[normalization_name]
    # cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)
    cat_names = ['background',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7'
                ]
    cat_colors = {
        0: torch.tensor([0.0, 0.0, 0.0]).float(), 
        1: torch.tensor([255.0, 51.0, 51.0]).float() / 255.0, # red
        2: torch.tensor([255.0, 128.0, 0.0]).float() / 255.0, # orange
        3: torch.tensor([255.0, 255.0, 0.0]).float() / 255.0, # yellow
        4: torch.tensor([0.0, 255.0, 0.0]).float() / 255.0, # green
        5: torch.tensor([0.0, 255.0, 255.0]).float() / 255.0, # cyan
        6: torch.tensor([0.0, 0.0, 255.0]).float() / 255.0, # blue
        7: torch.tensor([255.0, 0.0, 255.0]).float() / 255.0, # pink
    }

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
        
    #     save_name = str(output_dir) + "/" + "original_" + str(filename.name)

    #     pil_im.save(save_name)


    test_augs = ToColor()

    test_data_config = "/home/nelsonni/laviolette/method_analysis/configs/seg_test_config.json"

    test_loader = SegDataLoader(test_data_config, transform=test_augs)

    test_loader = DataLoader(test_loader, batch_size=1,
                        shuffle=False)

    test_seg_gt = {}
    for batch in test_loader:
        test_seg_gt[batch['patient'][0]] = batch['seg'][0]

    test_seg_pred = {}
    for batch in test_loader:
        im = batch['mri'].to(ptu.device) # Get MRI
        ori_shape = (512, 512)
        im_size = (512, 512)
        filename = batch['patient']

        with amp_autocast():
            seg_pred = inference(
                model,
                im,
                im,
                im_size,
                512,
                1,
                batch_size=1,
            )
#             seg_pred = model.forward(im)
            seg_pred = seg_pred.argmax(0)
        seg_pred = seg_pred.cpu().numpy()
        test_seg_pred[filename[0]] = seg_pred

        print(f'{iou_pytorch(torch.from_numpy(seg_pred), test_seg_gt[filename[0]])},')


        # plt.imshow(im[0].cpu(), cmap='gray')
        # plt.imshow(seg_map[0].cpu(), alpha=0.5)
        # file_name = data_dict['patient'] +'_' + data_dict['slide'] + '.jpg'

        # plt.savefig(output_dir / file_name)

        # seg_rgb = seg_to_rgb(seg_map, cat_colors)
        # seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        # pil_seg = Image.fromarray(seg_rgb[0])

        # file_name = data_dict['patient'] +'_' + data_dict['slide'] + '.jpg'
        # pil_im = Image.fromarray(np.uint8(norm_image))

        # pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
        # pil_blend.save(output_dir / file_name)
        
    scores = compute_metrics(
        test_seg_pred,
        test_seg_gt,
        8,
        ret_cat_iou=True,
    )

    if ptu.dist_rank == 0:
        scores["inference"] = "single_scale" if not False else "multi_scale"
        suffix = "ss" if not False else "ms"
        scores["cat_iou"] = np.round(100 * scores["cat_iou"], 2).tolist()
        for k, v in scores.items():
            if k != "cat_iou" and k != "inference":
                scores[k] = v.item()
            if k != "cat_iou":
                print(f"{k}: {scores[k]}")
        scores_str = yaml.dump(scores)
        with open(model_dir.parent / f"scores_{suffix}.yml", "w") as f:
            f.write(scores_str)


if __name__ == "__main__":
    main()
