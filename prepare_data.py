import os
import cv2
import csv
import argparse

import data_utils

CLASS_MAP = {'Normal': 0, 'CP': 1, 'NCP': 2}


def get_lesion_files(lesion_file, exclude_list, root_dir):
    """Reads the lesion file to identify relevant
    slices and returns their paths"""
    files = []
    with open(lesion_file, 'r') as f:
        f.readline()
        for line in f.readlines():
            cls, pid = line.split('/')[:2]
            if pid not in exclude_list:
                files.append(os.path.join(root_dir, line.strip('\n')))
    return files


def get_files(lesion_file, unzip_file, exclude_file, root_dir):
    """Gets image file paths according to given lists"""
    excluded_pids = get_excluded_pids(exclude_file)
    files = get_lesion_files(lesion_file, excluded_pids['CP'] + excluded_pids['NCP'], root_dir)
    with open(unzip_file, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in reader:
            if row['label'] == 'Normal':
                pid = row['patient_id']
                sid = row['scan_id']
                if pid not in excluded_pids['Normal']:
                    files += get_source_paths(root_dir, 'Normal', pid, sid)
    return files


def get_source_paths(root_dir, cls, pid, sid):
    """Helper function to construct source paths"""
    exam_dir = os.path.join(root_dir, cls, pid, sid)
    return list(data_utils.multi_ext_file_iter(exam_dir, data_utils.IMG_EXTENSIONS))


def make_output_path(output_dir, source_path):
    """Helper function to construct output paths"""
    source_path = source_path.replace('\\', '/')
    parts = source_path.split('/')[-4:]
    output_path = os.path.join(output_dir, '_'.join(parts))
    output_path = os.path.splitext(output_path)[0] + '.png'
    return output_path


def imread_gray(path):
    """Reads images in grayscale"""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.ndim > 2:
        image = image[:, :, 0]
    return image


def get_excluded_pids(exclude_file):
    """Reads the exclusion list and returns a
    dict of lists of excluded patients"""
    exclude_pids = {
        'NCP': [],
        'CP': [],
        'Normal': []
    }
    with open(exclude_file, 'r') as f:
        for line in f.readlines():
            cls, pid = line.strip('\n').split()
            exclude_pids[cls].append(pid)
    return exclude_pids


parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str,
                    help='Root directory with NCP, CP, and Normal directories, as well as metadata files')
parser.add_argument('-o', '--output_dir', type=str, default='data/COVIDx-CT', help='Directory to construct dataset in')
parser.add_argument('-l', '--lesion_file', type=str, default='lesions_slices.csv',
                    help='CSV file indicating slices with lesions')
parser.add_argument('-u', '--unzip_file', type=str, default='unzip_filenames.csv',
                    help='CSV file indicating unzipped filenames')
parser.add_argument('-e', '--exclude_file', type=str, default='exclude_list.txt',
                    help='Text file indicating patient IDs to skip')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Get filenames
lesion_file = os.path.join(args.root_dir, args.lesion_file)
unzip_file = os.path.join(args.root_dir, args.unzip_file)
exclude_file = os.path.join(args.root_dir, args.exclude_file)
image_files = get_files(lesion_file, unzip_file, exclude_file, args.root_dir)

# Write to new files as PNGs
for imf in image_files:
    image = imread_gray(imf)
    output_path = make_output_path(args.output_dir, imf)
    cv2.imwrite(output_path, image)
