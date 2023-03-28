import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
from PIL import Image
import numpy as np

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    class_weights
):
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.float().to(ptu.device))
    # criterion = torch.nn.CrossEntropyLoss()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    #data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in data_loader: #logger.log_every(data_loader, print_freq, header):
        im = batch['mri'].to(ptu.device) # Get MRI
        seg_gt = batch['seg'].long().to(ptu.device) # Get Seg

        with amp_autocast():
            seg_pred = model.forward(im)
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    return logger


def train_one_epoch_unet(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    class_weights
):
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights.float().to(ptu.device))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    #data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in data_loader: #logger.log_every(data_loader, print_freq, header):
        im = batch['mri'].to(ptu.device) # Get MRI
        seg_gt = batch['seg'].long().to(ptu.device) # Get Seg

        with amp_autocast():
            seg_pred = model.forward(im)
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

    return logger

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    num = 0
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch['mri']]
        #ims_metas = batch["im_metas"]
        #ori_shape = ims_metas[0]["ori_shape"] #TODO fix brutal hardcoding
        ori_shape = (512, 512)
        filename = batch['patient']

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)


        print(f'pred values: {torch.unique(seg_pred)}')


        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename[0]] = seg_pred


        # pil_im = Image.fromarray(np.uint8(seg_pred), 'L')
        # name = str(num) + '.jpg'
        # pil_im.save('./segm/outputs' + '/' + name)

        num += 1

        

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        8, #TODO remove brutal hard coded values
        #ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger


@torch.no_grad()
def unet_evaluate(
    model,
    data_loader,
    val_seg_gt,
    amp_autocast,
):
    model_without_ddp = model
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50


    for patient, seg_gt_tmp in val_seg_gt.items():
        print(f'GT values: {torch.unique(seg_gt_tmp)}')

    val_seg_pred = {}
    model.eval()
    num = 0
    for batch in data_loader: #logger.log_every(data_loader, print_freq, header):
        # ims = [im.to(ptu.device) for im in batch[0]]
        im = batch['mri'].to(ptu.device) # Get MRI
        #ims_metas = batch["im_metas"]
        #ori_shape = ims_metas[0]["ori_shape"] #TODO fix brutal hardcoding
        ori_shape = (512, 512)
        filename = batch['patient']

        with amp_autocast():
            seg_pred = model.forward(im)
            # seg_pred = utils.unet_inference(
            #     model_without_ddp,
            #     ims,
            #     ori_shape,
            # )
            seg_pred = seg_pred.argmax(0)

        print(f'pred values: {torch.unique(seg_pred)}')


        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename[0]] = seg_pred


        # pil_im = Image.fromarray(np.uint8(seg_pred), 'L')
        # name = str(num) + '.jpg'
        # pil_im.save('./segm/outputs' + '/' + name)

        num += 1

        
    val_seg_pred = gather_data(val_seg_pred, tmp_dir='.')
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        8, #TODO remove brutal hard coded values
        #ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
