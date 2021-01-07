import os
import cv2
import csv
import glob
import numpy as np
from tqdm import tqdm

from .utils import load_nifti_volume, process_segmentation_data, ranges_to_indices, CLASS_MAP


def process_mosmed_seg_data(ct_dir, mask_dir, output_dir):
    """Process slices for segmented COVID-19 studies from MosMedData"""
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


def process_mosmed_unseg_data(mosmed_meta_csv, ct_dir, output_dir, class_map=CLASS_MAP):
    """Process slices for unsegmented COVID-19 studies from MosMedData"""
    filenames = []
    classes = []
    with open(mosmed_meta_csv, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in tqdm(reader):
            # Load volume
            pid = row['pid']
            ct_file = glob.glob(os.path.join(ct_dir, '*', pid + '*.nii.gz'))[0]
            volume = load_nifti_volume(ct_file)
            volume = np.rot90(volume, axes=(1, 2))  # rotate to natural orientation

            # Save slices
            cls = class_map[row['finding']]
            slice_indices = ranges_to_indices(row['slice indices'])
            for idx in slice_indices:
                filenames.append(pid + '-{:04d}.png'.format(idx))
                classes.append(cls)

                out_file = os.path.join(output_dir, filenames[-1])
                if not os.path.exists(out_file):
                    cv2.imwrite(out_file, volume[idx])
    return filenames, classes
