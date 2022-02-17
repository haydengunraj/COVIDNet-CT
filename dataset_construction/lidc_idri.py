import os
import csv
import cv2
import numpy as np
import pylidc as pl
import pydicom.errors
from tqdm import tqdm

from data_utils import ensure_uint8
from .utils import ranges_to_indices, CLASS_MAP

_NODULE_PAD = 3


def process_lidc_idri_data(meta_csv, output_dir, pad=_NODULE_PAD):
    """Process slices for all LIDC-IDRI studies in the given CSV file"""
    filenames = []
    classes = []
    with open(meta_csv, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in tqdm(reader):
            if row['lung range']:
                slice_indices = ranges_to_indices(row['lung range'])
            else:
                slice_indices = None
            fnames = _process_lidc_idri_stack(
                row['patient id'], row['SeriesInstanceUID'], slice_indices, output_dir, pad=pad)
            cls = CLASS_MAP[row['finding']]
            filenames.extend(fnames)
            classes.extend([cls]*len(fnames))
    return filenames, classes


def _process_lidc_idri_stack(patient_id, series_instance_uid, slice_indices, output_dir, pad=3):
    """Processes slices from a particular LIDC-IDRI stack"""
    # Load scan volume
    scan = pl.query(pl.Scan).filter(
        pl.Scan.patient_id == patient_id,
        pl.Scan.series_instance_uid == series_instance_uid).first()
    try:
        vol = scan.to_volume(verbose=False)
        vol = ensure_uint8(vol)
    except pydicom.errors.InvalidDicomError:
        raise pydicom.errors.InvalidDicomError('Could not load volume for ', patient_id)

    if slice_indices is None:
        slice_indices = range(vol.shape[2])

    # Make nodule mask
    full_mask = np.zeros(vol.shape, dtype=np.uint8)
    for ann in scan.annotations:
        mask = ann.boolean_mask(pad=pad)
        bbox = ann.bbox(pad=pad)
        full_mask[bbox][mask] = 1

    # Write slice images
    filenames = []
    prefix = patient_id + '-' + series_instance_uid
    for idx in slice_indices:
        has_nodule = np.any(full_mask[:, :, idx])
        if not has_nodule:
            filenames.append(prefix + '-{:04d}.png'.format(idx))
            out_fname = os.path.join(output_dir, filenames[-1])
            if not os.path.isfile(out_fname):
                cv2.imwrite(out_fname, vol[:, :, idx])
    return filenames
