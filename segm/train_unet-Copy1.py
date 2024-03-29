# from homologous_point_prediction.evaluate import evaluate, evaluate_rotation

# +
import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from segm.data_processing.seg_data_loader import SegDataLoader
from segm.data_processing.transforms import RandomCrop, RandomFlip, ElasticTransform, ToColor
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision import transforms

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
import segm.utils.torch as ptu
from segm.model import utils
from segm.data.factory import create_dataset
from segm.utils.logger import MetricLogger
from segm.model.utils import num_params
from segm.metrics import gather_data, compute_metrics

from segm.unet.unet_model import UNet

from timm.utils import NativeScaler
from contextlib import suppress

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate, unet_evaluate, train_one_epoch_unet

from sklearn.utils.class_weight import compute_class_weight


# +
log_dir = '../unet_200_noCrop'
im_size = (512, 512)

optimizer='sgd'
scheduler='polynomial'
weight_decay=0.0
drop_path=0.1
batch_size=2
epochs=200
learning_rate=0.001
normalization=None
eval_freq=None
amp=False
resume=False

# start distributed mode
ptu.set_gpu_mode(True)
distributed.init_process()


# model config
if not im_size:
    im_size = dataset_cfg["im_size"]

if batch_size:
    world_batch_size = batch_size
if epochs:
    num_epochs = epochs
if learning_rate:
    lr = learning_rate
if eval_freq is None:
    eval_freq = 1

# experiment config
batch_size = world_batch_size // ptu.world_size


# +
log_dir = Path(log_dir)
log_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = log_dir / "checkpoint.pth"

image_augs = transforms.Compose([ElasticTransform(), RandomCrop(400), RandomFlip(), ToColor()])

training_data_config = "/home/nelsonni/laviolette/method_analysis/configs/seg_train_config.json"
validation_data_config = "/home/nelsonni/laviolette/method_analysis/configs/seg_val_config.json"

train_loader = SegDataLoader(training_data_config, transform=image_augs)
pixel_count_loader = SegDataLoader(training_data_config, transform=None)

val_loader = SegDataLoader(validation_data_config, transform=None)

train_loader = dataloader = DataLoader(train_loader, batch_size=batch_size,
                    shuffle=True, num_workers=batch_size)
val_loader = dataloader = DataLoader(val_loader, batch_size=1,
                    shuffle=True, num_workers=0)


n_cls = 8 #TODO fix sloppy code


model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=8, init_features=32, pretrained=False)

# model =  UNet(n_channels=3, n_classes=8)
model = model.to(ptu.device)

# +
# optimizer
optimizer_kwargs = {
  'clip_grad': None,
  'epochs': epochs,
  'iter_max': 7600,
  'iter_warmup': 0.0,
  'lr': lr,
  'min_lr': 1.0e-05,
  'momentum': 0.9,
  'opt': 'sgd',
  'poly_power': 0.9,
  'poly_step_size': 1,
  'sched': 'polynomial',
  'weight_decay': 0.0
}
optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
optimizer_kwargs["iter_warmup"] = 0.0
opt_args = argparse.Namespace()
opt_vars = vars(opt_args)
for k, v in optimizer_kwargs.items():
    opt_vars[k] = v

# optimizer = create_optimizer(opt_args, model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = create_scheduler(opt_args, optimizer)
num_iterations = 0
amp_autocast = suppress
loss_scaler = None
if amp:
    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()

sync_model(log_dir, model)

if ptu.distributed:
    model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)


start_epoch = 0 # TODO remove slop
num_epochs = epochs # TODO remove slop

model_without_ddp = model
if hasattr(model, "module"):
    model_without_ddp = model.module

#val_seg_gt = val_loader.dataset.get_gt_seg_maps()

val_seg_gt = {}
for batch in val_loader:
    val_seg_gt[batch['patient'][0]] = batch['seg'][0]

print(f"Train dataset length: {len(train_loader) * batch_size}")
print(f"Val dataset length: {len(val_loader)}")

# +
pixel_values = torch.tensor([])
for batch in train_loader:
    if len(batch['seg']) > 1:
        for i in range(len(batch['seg'])):
            pixel_values = torch.cat((pixel_values, batch['seg'][i].flatten())) 
    else:
        pixel_values = torch.cat((pixel_values, batch['seg'][0].flatten()))

counts = pixel_values.unique(return_counts=True)
class_weights = []

for count in counts[1]:
    class_weights.append(sum(counts[1]) / (n_cls * count))
class_weights = torch.tensor(class_weights)

# +

for epoch in range(start_epoch, num_epochs):
    #train
    train_logger = MetricLogger(delimiter="  ")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.float().to(ptu.device))
    num_updates = epoch * len(train_loader)
    model.train()
    for batch in train_loader:
        im = batch['mri'].to(ptu.device) # Get MRI
        seg_gt = batch['seg'].long().to(ptu.device) # Get Seg
        optimizer.zero_grad()
        with amp_autocast():
#             before = model.encoder1.enc1conv1.weight.clone()
            seg_pred = model.forward(im)
#             after = model.encoder1.enc1conv1.weight.clone()
            loss = criterion(seg_pred, seg_gt)
        loss.backward()
#         print(torch.allclose(before, after))
        
        optimizer.step()
        
#         before = after
#         after = model.encoder1.enc1conv1.weight.clone()
#         print(torch.allclose(before, after))
#         print()
        num_updates += 1
#         lr_scheduler.step_update(num_updates=num_updates)
        torch.cuda.synchronize()

        train_logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        

    # save checkpoint
    if ptu.dist_rank == 0:
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_path)
        
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    # Evaluate
    model.eval()
    val_seg_pred = {}
    eval_logger = MetricLogger(delimiter="  ")
    
    for patient, seg_gt_tmp in val_seg_gt.items():
        print(f'GT values: {torch.unique(seg_gt_tmp)}')
    for batch in val_loader:
        im = batch['mri'].to(ptu.device) # Get MRI
        ori_shape = (512, 512)
        filename = batch['patient']

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                im,
                im,
                (512,512),
                512,
                1,
                batch_size=1,
            )
            seg_pred_2 = model_without_ddp.forward(im)
            print('seg_pred_2 shape:', seg_pred_2.shape)
            print(f'pred values2 unique 1: {torch.unique(seg_pred_2[0].argmax(0))}')
            seg_pred_2 = F.softmax(seg_pred_2[0], 0).argmax(0)
#             seg_pred = model.forward(im)
            seg_pred = seg_pred.argmax(0)
        print(f'pred values: {torch.unique(seg_pred)}')
        print(f'pred values2 unique 2: {torch.unique(seg_pred_2)}')
        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename[0]] = seg_pred
        
   
    val_seg_pred = gather_data(val_seg_pred, tmp_dir='.')
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        8, #TODO remove brutal hard coded values
        #ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        eval_logger.update(**{f"{k}": v, "n": 1})

    # log stats
    eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
    if ptu.dist_rank == 0:
        train_stats = {
            k: meter.global_avg for k, meter in train_logger.meters.items()
        }
        val_stats = {}
        if eval_epoch:
            val_stats = {
                k: meter.global_avg for k, meter in eval_logger.meters.items()
            }

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "num_updates": (epoch + 1) * len(train_loader),
        }

        print(log_stats)
#         with open(log_dir / "log.txt", "a") as f:
#             f.write(json.dumps(log_stats) + "\n")

distributed.barrier()
distributed.destroy_process()
sys.exit(1)
