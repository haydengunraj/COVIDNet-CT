import os
import csv
import cv2
import numpy as np
from tqdm import tqdm

from .utils import load_nifti_volume, ranges_to_indices, CLASS_MAP

_TCIA_FNAME_GLOB = 'volume-covid19-A-{}.nii.gz'


def process_tcia_covid_data(root_dir, meta_csv, output_dir, class_map=CLASS_MAP):
    """Process slices for all TCIA COVID-19 studies in the given CSV file"""
    filenames = []
    classes = []
    with open(meta_csv, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in tqdm(reader):
            if row['slice indices']:
                slice_indices = ranges_to_indices(row['slice indices'])
            else:
                slice_indices = None
            fnames = _process_tcia_covid_stack(
                root_dir, row['pid'], slice_indices, output_dir)
            cls = class_map[row['finding']]
            filenames.extend(fnames)
            classes.extend([cls for _ in range(len(fnames))])
    return filenames, classes


def _process_tcia_covid_stack(tcia_dir, pid, slice_indices, output_dir):
    """Processes slices from a particular TCIA CT stack"""
    # Load NiFTI stack
    fname = _TCIA_FNAME_GLOB.format(pid)
    nifti_fname = os.path.join(tcia_dir, fname)
    volume = load_nifti_volume(nifti_fname)
    volume = np.swapaxes(volume, 1, 2)  # transpose to natural orientation
    if slice_indices is None:
        slice_indices = range(volume.shape[0])

    # Write slice images
    filenames = []
    prefix = fname.split('.')[0]
    for idx in slice_indices:
        slc = volume[idx]
        filenames.append(prefix + '-{:04d}.png'.format(idx))
        out_file = os.path.join(output_dir, filenames[-1])
        if not os.path.exists(out_file):
            cv2.imwrite(out_file, slc)
    return filenames
