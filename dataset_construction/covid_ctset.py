import os
import cv2
import csv
import glob
import numpy as np
from tqdm import tqdm

from .utils import CLASS_MAP
from data_utils import ensure_uint8

_RESCALE_SLOPE = 1
_RESCALE_INTERCEPT = -1024
_CLASS_MAP = {'Positive': CLASS_MAP['COVID-19'], 'Negative': CLASS_MAP['Normal']}
_FNAME_FMT_GLOB = '*_{}_*.tif'


def _uint16_hu_to_uint8(data):
    data = data.astype(np.float)*_RESCALE_SLOPE + _RESCALE_INTERCEPT
    return ensure_uint8(data)


def process_covid_ctset_data(meta_csv, root_dir, output_dir):
    """Processes slices for all patients in the given COVID-CTSet CSV file"""
    filenames = []
    classes = []
    with open(meta_csv, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in tqdm(reader):
            fnames = _process_covid_ctset_patient(row['Patient ID'], root_dir, output_dir)
            cls = _CLASS_MAP[row['COVID-19 Infection']]
            filenames.extend(fnames)
            classes.extend([cls]*len(fnames))
    return filenames, classes


def _process_covid_ctset_patient(pid, img_dir, output_dir):
    """Processes images for a COVID-CTSet patient"""
    filenames = []
    image_files = glob.glob(os.path.join(img_dir, _FNAME_FMT_GLOB.format(pid)))
    for imf in image_files:
        fname = os.path.basename(imf).replace('.tif', '.png')
        filenames.append(fname)
        out_file = os.path.join(output_dir, fname)
        if not os.path.exists(out_file):
            image = cv2.imread(imf, cv2.IMREAD_UNCHANGED)
            image = _uint16_hu_to_uint8(image)
            cv2.imwrite(out_file, image)
    return filenames

