import os
import glob
import numpy as np
from tqdm import tqdm

from .utils import load_nifti_volume, process_segmentation_data, CLASS_MAP


def process_mosmed_data(ct_dir, mask_dir, output_dir):
    """Process slices for COVID-19 segmentation studies from MosMedData"""
    filenames = []
    classes = []
    seg_files = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz')))
    for seg_file in tqdm(seg_files):
        ct_file = os.path.join(ct_dir, os.path.basename(seg_file).replace('_mask', ''))
        volume = load_nifti_volume(ct_file)
        seg = load_nifti_volume(seg_file)
        volume = np.rot90(volume, axes=(1, 2))  # rotate to natural orientation
        seg = np.rot90(seg, axes=(1, 2))  # rotate to natural orientation
        fnames = process_segmentation_data(
            volume, seg, os.path.basename(ct_file).split('.')[0], output_dir)
        filenames.extend(fnames)
        classes.extend([CLASS_MAP['COVID-19']]*len(fnames))
    return filenames, classes
