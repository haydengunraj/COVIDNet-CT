import os
import glob
import numpy as np
from tqdm import tqdm

from .utils import load_nifti_volume, process_segmentation_data, CLASS_MAP


def process_radiopaedia_and_coronacases_seg_data(ct_dir, mask_dir, output_dir):
    """Process slices for COVID-19 segmentation studies from coronacases and radiopaedia"""
    filenames = []
    classes = []
    ct_files = sorted(glob.glob(os.path.join(ct_dir, '*.nii.gz')))
    for ct_file in tqdm(ct_files):
        seg_file = os.path.join(mask_dir, os.path.basename(ct_file))
        volume = load_nifti_volume(ct_file)
        seg = load_nifti_volume(seg_file)
        if 'radiopaedia' in ct_file:
            volume = np.swapaxes(volume, 1, 2)  # transpose to natural orientation
            seg = np.swapaxes(seg, 1, 2)  # transpose to natural orientation
        else:
            volume = np.rot90(volume, axes=(1, 2))  # rotate to natural orientation
            seg = np.rot90(seg, axes=(1, 2))  # rotate to natural orientation
        fnames = process_segmentation_data(
            volume, seg, os.path.basename(ct_file).split('.')[0], output_dir)
        filenames.extend(fnames)
        classes.extend([CLASS_MAP['COVID-19']]*len(fnames))
    return filenames, classes
