#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent) + '/mri_histology_toolkit')
sys.path.append(str(Path.cwd().parent.parent) + '/homologous_point_prediction')
from mri_histology_toolkit.data_loader import get_child_dirs, get_leaf_dirs
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
import random
import json
import math
import os

import scipy.io
import nibabel as nib
import cv2
import torch
import warnings

nib.imageglobals.logger.setLevel(40)

def load_histology(filepath):
    """Loads the histology as a grayscale 2D float32 array of range [0, 1]"""
    hist = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if hist.shape[0] != 512 or hist.shape[1] != 512:
        hist = cv2.resize(hist, (512, 512), interpolation=cv2.INTER_AREA)
    if len(hist.shape) == 3:
        hist =  cv2.cvtColor(hist, cv2.COLOR_BGR2GRAY) # 2D uint8
    return hist.astype(np.float32)

def load_mri(filepath):
    """Loads MRI nii file returns a 2D float 32 array of range [0, 1]"""
    warnings.filterwarnings('ignore')
    mri = nib.load(filepath).get_fdata()
    temp = mri - np.min(mri)
    return (temp / (1 if np.max(temp) == 0 else np.max(temp))).astype(np.float32)

def extract_normal(slide_dir, other_mri_name):
    # Remeber that these generated files are augmented, so they don't line up with the mat files

    seg_file_name = 'mri_seg_mask.tiff'
    t2_name = 'mri_slice_double_T2.nii'
    # mri_file_name = 'mri_slice_double_DWI_clin_reg.nii'
    adc_name = 'mri_slice_double_ADC_realigned.nii'
    mri_1 = 'mri_slice_double_b10_realigned.nii'
    mri_2 = 'mri_slice_double_b100_realigned.nii'
    mri_3 = 'mri_slice_double_b1000_realigned.nii'
    mri_4 = 'mri_slice_double_b1400_realigned.nii'
    mask_name  = 'mri_slice_double_prostate_mask.nii'

    seg = load_histology(os.path.join(slide_dir, seg_file_name))
    # seg = np.clip(seg.astype(int) - 1, 0, None)
    
    # Logic below is used to convert 9 class into just 3 class, containing
    # background, prostate, and cancer classes
    seg[seg == 7] = 10
    seg[seg == 6] = 10
    seg[seg == 5] = 10
    seg[seg == 4] = 10

    seg[seg == 3] = 8
    seg[seg == 2] = 8
    seg[seg == 1] = 8

    seg[seg == 10] = 2
    seg[seg == 8] = 1

    if len(np.unique(seg, return_counts=True)[1]) != 2:
        raise Exception("Incorrect Mask")


    mri_names = [t2_name]

    t2_mris = []


    for mri_name in mri_names:
        mri = load_mri(os.path.join(slide_dir, (mri_name)))
        # Zscore normalization
        mri = (mri - mri.mean()) / mri.std()
        mri = torch.from_numpy(mri)
        t2_mris.append(mri)

    mri_names = [other_mri_name]

    other_mris = []


    for mri_name in mri_names:
        mri = load_mri(os.path.join(slide_dir, (mri_name)))
        # Zscore normalization
        mri = (mri - mri.mean()) / mri.std()
        mri = torch.from_numpy(mri)
        other_mris.append(mri)


    t2_mri = torch.stack(t2_mris, dim=0)
    other_mri = torch.stack(other_mris, dim=0)

    return (t2_mri, other_mri, slide_dir.split('Prostates/')[1])

def extract_slide(slide_dir, other_mri_name):
    (t2_mri, other_mri, filename) = extract_normal(slide_dir, other_mri_name)

    return (t2_mri, other_mri, filename)


class RegDataLoader(Dataset):

    """Used to load multiple MRI images for use in the Registration Pipeline"""

    def __init__(self, config_path, transform=None, other_mri_name=''):
        self._parse_config(config_path)
        self.slide_dirs = []
        self.transform = transform
        self.other_mri_name = other_mri_name

        patients = get_child_dirs(self.parent_dir)
        for include_patient in self.include_patients:
            if include_patient not in patients:
                print('Warning: patient {0} not found...ignoring'.format(include_patient))
            else:
                self.slide_dirs += get_child_dirs(os.path.join(self.parent_dir, include_patient), full_path=True)
        print(f'Removing the following slides: {self.incomplete_patients}')
        self.slide_dirs = [i for i in self.slide_dirs if i not in self.incomplete_patients]
        self.slide_dirs = self.get_complete_dirs(self.slide_dirs)
        self.indices = np.arange(len(self.slide_dirs))
        self.slide_dirs = np.array(self.slide_dirs)
        np.random.shuffle(self.indices)

    def _parse_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        self.parent_dir = config['parent_dir']
        self.include_patients = config['include_patients']
        self.augment_rotation = config['augment_rotation_range']
        self.use_masked = config['use_masked']
        self.use_warped_pairs = config['use_warped_pairs']
        self.incomplete_patients = config['incomplete_patients']
        assert len(self.parent_dir) > 0
        assert len(self.include_patients) > 0
        assert len(self.augment_rotation) == 2

    def __len__(self):
        return int((len(self.slide_dirs)))

    def __getitem__(self, batch_index):
        batch_indices = self.indices[batch_index : batch_index + 1]
        batch_sample_dir = self.slide_dirs[batch_indices]
        sample = self.extract_data(batch_sample_dir[0])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_complete_dirs(self, dirs):
        tmp_dirs = dirs.copy()
        for sample_dir in dirs:
            try:
                extract_slide(sample_dir, self.other_mri_name)
            except Exception as e:
                pass
                tmp_dirs.remove(sample_dir)
        return tmp_dirs

    def extract_data(self, dir):
        data_dict = {}
        try:
            (t2_mri, other_mri, patient) = extract_slide(dir, self.other_mri_name)
            data_dict['t2_mri'] = t2_mri
            data_dict['other_mri'] = other_mri
            data_dict['patient'] = patient
        except Exception as e:
            pass
        return data_dict

    def on_epoch_end(self):
        np.random.shuffle(self.indices)




