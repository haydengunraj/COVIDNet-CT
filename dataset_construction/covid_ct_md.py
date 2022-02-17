import os
import cv2
import csv
import glob
import numpy as np
from tqdm import tqdm

from .utils import load_dicom, CLASS_MAP


def process_covid_ct_md_data(root_dir, index_csv, label_file, output_dir, class_map=CLASS_MAP):
    """Process slices from COVID-CT-MD dataset"""
    filenames, classes = [], []

    # Process COVID-19 and pneumonia cases
    labels = np.load(label_file)
    with open(index_csv, 'r') as f:
        # Get list of normal case dirs and CSV lines to set progress bar length
        normal_dirs = glob.glob(os.path.join(root_dir, 'Normal Cases', 'normal*'))
        reader = list(csv.DictReader(f))
        pbar = tqdm(total=(len(normal_dirs) + len(reader)))
        for row in reader:
            label_idx = int(row['Label Index'])
            if label_idx > 42:
                label_idx -= 1  # indexing jumps from 41 to 43, so need to correct > 42
            cls = row['Diagnosis']
            pid = row['Folder/ID']
            path = os.path.join(root_dir, row['Relative Path'].rstrip('/').rstrip('.') + ' Cases', pid)
            dcm_files = sorted(glob.glob(os.path.join(path, '*.dcm')))
            indices = np.where(labels[label_idx, :len(dcm_files)])[0]
            for i in indices:
                fname = 'COVIDCTMD-{}-{}.png'.format(pid, os.path.basename(dcm_files[i]).split('.')[0])
                out_file = os.path.join(output_dir, fname)
                filenames.append(fname)
                classes.append(class_map['Pneumonia' if cls == 'CAP' else 'COVID-19'])
                if not os.path.exists(out_file):
                    slc = load_dicom(dcm_files[i])
                    cv2.imwrite(out_file, slc)
            pbar.update(1)

    # Process normal cases
    for normal_dir in normal_dirs:
        pid = os.path.basename(normal_dir)
        dcm_files = sorted(glob.glob(os.path.join(normal_dir, '*.dcm')))
        for i, dcm_file in enumerate(dcm_files):
            fname = 'COVIDCTMD-{}-{}.png'.format(pid, os.path.basename(dcm_file).split('.')[0])
            out_file = os.path.join(output_dir, fname)
            filenames.append(fname)
            classes.append(class_map['Normal'])
            if not os.path.exists(out_file):
                slc = load_dicom(dcm_file)
                cv2.imwrite(out_file, slc)
        pbar.update(1)
    pbar.close()

    return filenames, classes
