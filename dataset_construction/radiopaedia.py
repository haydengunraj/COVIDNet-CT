import os
import cv2
import csv
import shutil
import requests
import numpy as np
from tqdm import tqdm

from .utils import ranges_to_indices, CLASS_MAP

_RADIOPAEDIA_URL_FMT = 'https://radiopaedia.org/studies/{}/stacks'


def process_radiopaedia_data(meta_csv, exclude_file, output_dir, class_map=CLASS_MAP):
    """Downloads slices for all studies in the given radiopaedia CSV file"""
    # Create exclude list
    exclude_list = set()
    with open(exclude_file, 'r') as f:
        for line in f.readlines():
            exclude_list.add(line.strip('\n'))

    filenames = []
    classes = []
    with open(meta_csv, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in tqdm(reader):
            if row['studyID'] in exclude_list:
                continue
            stack_index = int(row['stack index'])
            if row['slice indices']:
                slice_indices = ranges_to_indices(row['slice indices'])
            else:
                slice_indices = None
            fnames = _download_radiopaedia_stack(
                row['rID'], row['studyID'], stack_index, output_dir, slice_indices=slice_indices)
            cls = class_map[row['finding']]
            filenames.extend(fnames)
            classes.extend([cls for _ in range(len(fnames))])
    return filenames, classes


def _download_radiopaedia_stack(r_id, study_id, stack_index, output_dir, slice_indices=None):
    """Downloads slices from a particular radiopaedia stack"""
    # Get JSON metadata from radiopaedia
    rad_url = _RADIOPAEDIA_URL_FMT.format(study_id)
    r = requests.get(rad_url)
    if r.status_code != 200:
        raise ValueError('Could not retrieve stack metadata for studyID: ' + study_id)
    data = r.json()

    # Get filenames and sort by position
    urls = []
    positions = []
    for img in data[stack_index]['images']:
        urls.append(img['fullscreen_filename'])
        positions.append(img['position'])
    urls = np.asarray(urls)[np.argsort(positions)]
    if slice_indices is None:
        slice_indices = range(len(urls))

    # Download images
    case_str = 'radiopaedia-{}-{}-{}'.format(r_id, study_id, stack_index)
    filenames = []
    for idx in slice_indices:
        jpg_fname = os.path.join(output_dir, '{}-{:04d}.jpg'.format(case_str, idx))
        png_fname = jpg_fname.replace('.jpg', '.png')
        filenames.append(png_fname)

        if not os.path.isfile(png_fname):
            if not os.path.isfile(jpg_fname):
                # Download JPG image
                url = urls[idx]
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(jpg_fname, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                else:
                    print('Could not retrieve image from ' + url)

            # Convert to grayscale PNG
            image = cv2.imread(jpg_fname, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(png_fname, image)
            os.remove(jpg_fname)

    return filenames
