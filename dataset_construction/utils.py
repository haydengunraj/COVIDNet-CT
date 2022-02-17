import os
import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pydicom import dcmread

from data_utils import ensure_uint8

CLASS_MAP = {'Normal': 0, 'Pneumonia': 1, 'COVID-19': 2}
LESION_THRESHOLD = 0.001  # at least 0.1% of the image area must be lesions


def load_nifti_volume(nifti_file):
    """Loads a volume from a NIfTI file, ensuring uint8 dtype"""
    volume = nib.load(nifti_file).get_fdata()
    volume = ensure_uint8(volume)
    volume = np.rollaxis(volume, 2, 0)  # HWN to NHW
    return volume


def load_dicom(dcm_file):
    """Loads a slice from a DICOM file, ensuring uint8 dtype"""
    ds = dcmread(dcm_file)
    data = np.float32(ds.pixel_array)*float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    image = ensure_uint8(data)
    return image


def load_mha_volume(mha_file):
    """Loads a volume from an MHA file, ensuring uint8 dtype"""
    reader = sitk.ImageFileReader()
    reader.SetFileName(mha_file)
    reader.SetImageIO('MetaImageIO')
    volume = reader.Execute()
    volume = sitk.GetArrayFromImage(volume)
    volume = ensure_uint8(volume)
    return volume


def ranges_to_indices(range_string):
    """Converts a string of ranges to a list of indices"""
    indices = []
    for span in range_string.split('/'):
        if ':' in span:
            start_idx, stop_idx = [int(idx) for idx in span.split(':')]
            stop_idx += 1  # add 1 since end index is excluded in range()
            indices.extend(list(range(start_idx, stop_idx)))
        else:
            indices.append(int(span))
    return indices


def process_segmentation_data(ct_volume, seg_volume, file_prefix, output_dir, lesion_threshold=LESION_THRESHOLD):
    """Extracts slices from the given ct volume where significant lesions are present.
    Volumes are assumed to have shape num_slices x height x width"""
    filenames = []
    lesion_frac = np.sum(seg_volume, axis=(1, 2)) / (ct_volume.shape[1]*ct_volume.shape[2])
    lesion_slices = np.where(lesion_frac > lesion_threshold)[0]
    for i in lesion_slices:
        slc = ct_volume[i]
        fname = file_prefix + '-{:04d}.png'.format(i)
        out_file = os.path.join(output_dir, fname)
        filenames.append(fname)
        if not os.path.exists(out_file):
            cv2.imwrite(out_file, slc)
    return filenames
