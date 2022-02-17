import os
import csv
import cv2
import glob
from tqdm import tqdm

from .utils import ranges_to_indices, CLASS_MAP


def process_ictcf_data(root_dir, meta_csv, output_dir, class_map=CLASS_MAP):
    """Process slices for COVID-19 studies from iCTCF"""
    filenames = []
    classes = []
    with open(meta_csv, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in tqdm(reader):
            slice_indices = ranges_to_indices(row['slice indices'])
            image_files = _get_patient_files(root_dir, row['pid'])
            cls = class_map[row['finding']]
            for i in slice_indices:
                out_fname = 'HUST-{}-{:04d}.png'.format(row['pid'].replace(' ', ''), i)
                out_file = os.path.join(output_dir, out_fname)
                filenames.append(out_fname)
                classes.append(cls)
                if not os.path.exists(out_file):
                    # Write to grayscale PNGs to save space
                    image = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(out_file, image)
    return filenames, classes


def _get_patient_files(ictcf_dir, pid):
    """Gets file paths for a particular iCTCF patient"""
    patient_dir = os.path.join(ictcf_dir, pid)
    return sorted(glob.glob(os.path.join(patient_dir, os.listdir(patient_dir)[0], '*.jpg')))


if __name__ == '__main__':
    ictcf_meta_csv = 'metadata/ictcf_metadata.csv'
    ictcf_dir = 'D:\\Datasets\\HUST-19\\cases'
    out_dir = 'D:\\Datasets\\HUST-19\\test_constructor'
    os.makedirs(out_dir, exist_ok=True)

    import numpy as np
    fnames, classes = process_ictcf_data(ictcf_meta_csv, ictcf_dir, out_dir)
    print(len(fnames), len(classes), np.unique(classes))
