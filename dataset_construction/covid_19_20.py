import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

from .utils import load_nifti_volume, CLASS_MAP, LESION_THRESHOLD

_CT_FNAME_GLOB = 'volume-covid19-A-*_ct.nii.gz'
_SEG_FNAME_GLOB = 'volume-covid19-A-*_seg.nii.gz'


def process_covid_19_20_data(root_dir, output_dir):
    """Process slices for COVID-19-20 challenge studies"""
    filenames = []
    classes = []
    ct_files = sorted(glob.glob(os.path.join(root_dir, _CT_FNAME_GLOB)))
    seg_files = sorted(glob.glob(os.path.join(root_dir, _SEG_FNAME_GLOB)))
    for ct_file, seg_file in tqdm(zip(ct_files, seg_files), total=len(ct_files)):
        volume = load_nifti_volume(ct_file)
        volume = np.swapaxes(volume, 1, 2)  # transpose to natural orientation
        seg = load_nifti_volume(seg_file)
        seg = np.swapaxes(seg, 1, 2)  # transpose to natural orientation
        lesion_frac = np.sum(seg, axis=(1, 2))/(volume.shape[1]*volume.shape[2])
        lesion_slices = np.where(lesion_frac > LESION_THRESHOLD)[0]
        for i in lesion_slices:
            fname = os.path.basename(ct_file).split('.')[0] + '-{:04d}.png'.format(i)
            out_file = os.path.join(output_dir, fname)
            filenames.append(fname)
            classes.append(CLASS_MAP['COVID-19'])
            if not os.path.exists(out_file):
                slc = volume[i]
                cv2.imwrite(out_file, slc)
    return filenames, classes
