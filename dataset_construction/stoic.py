import os
import cv2
import csv
from tqdm import tqdm

from .utils import load_mha_volume, ranges_to_indices, CLASS_MAP


def process_stoic_data(root_dir, meta_csv, output_dir, class_map=CLASS_MAP):
    """Process slices from the STOIC challenge dataset"""
    filenames, classes = [], []
    with open(meta_csv, 'r') as f:
        reader = list(csv.DictReader(f))
        for row in tqdm(reader):
            pid = row['pid']
            indices = ranges_to_indices(row['slice indices'])
            mha_file = os.path.join(root_dir, '{}.mha'.format(pid))
            volume = load_mha_volume(mha_file)[::-1]  # reverse to match indices
            for i in indices:
                fname = 'STOIC-{}-{:04d}.png'.format(pid, i)
                out_file = os.path.join(output_dir, fname)
                filenames.append(fname)
                classes.append(class_map['COVID-19'])
                if not os.path.exists(out_file):
                    cv2.imwrite(out_file, volume[i])
    return filenames, classes
