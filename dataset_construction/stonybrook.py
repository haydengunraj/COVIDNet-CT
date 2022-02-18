import os
import cv2
import csv
import glob
from tqdm import tqdm

from .utils import load_dicom, ranges_to_indices, CLASS_MAP


def process_stonybrook_data(root_dir, meta_csv, output_dir, class_map=CLASS_MAP):
    """Process slices from Stony Brook University's dataset"""
    filenames, classes = [], []
    with open(meta_csv, 'r') as f:
        reader = list(csv.DictReader(f))
        for row in tqdm(reader):
            pid = row['pid']
            study_desc = row['study description']
            series_num = row['series number']
            indices = ranges_to_indices(row['slice indices'])

            # Find series folder
            series_glob = os.path.join(root_dir, pid, study_desc, series_num + '.*')
            series_dir = glob.glob(series_glob)
            if len(series_dir) != 1:
                raise ValueError('Multiple matching series for {}'.format(series_glob))
            series_dir = series_dir[0]

            # Save slices
            dcm_files = sorted(glob.glob(os.path.join(series_dir, '*.dcm')))
            for i in indices:
                fname = '{}-{}-{}-{:04d}.png'.format(pid, study_desc.replace(' ', '-'), series_num, i)
                out_file = os.path.join(output_dir, fname)
                filenames.append(fname)
                classes.append(class_map['COVID-19'])
                if not os.path.exists(out_file):
                    slc = load_dicom(dcm_files[i])
                    cv2.imwrite(out_file, slc)
    return filenames, classes
