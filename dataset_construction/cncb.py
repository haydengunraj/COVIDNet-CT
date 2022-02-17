import os
import cv2
import csv
from tqdm import tqdm

from data_utils import multi_ext_file_iter, IMG_EXTENSIONS
from .utils import CLASS_MAP

_CLASS_MAP = {'Normal': CLASS_MAP['Normal'], 'CP': CLASS_MAP['Pneumonia'], 'NCP': CLASS_MAP['COVID-19']}
_LESION_FILE = 'lesions_slices.csv'
_UNZIP_FILE = 'unzip_filenames.csv'
_EXCLUDE_FILE = 'exclude_list.txt'

# Cases accidentally included that are removed in v2+
_PATCH_CASES = ['NCP_328_1805', 'CP_1781_3567', 'CP_1769_3516', 'NCP_1058_2635',
                'NCP_868_2395', 'NCP_868_2396', 'NCP_869_2397', 'NCP_911_2453']


def process_cncb_data(root_dir, exclude_file, output_dir, extra_lesion_files=None):
    """Process slices for all included CNCB studies"""
    # Get file paths
    lesion_files = [os.path.join(root_dir, _LESION_FILE)]
    if extra_lesion_files is not None:
        lesion_files += extra_lesion_files
    unzip_file = os.path.join(root_dir, _UNZIP_FILE)
    # exclude_file = os.path.join(cncb_dir, _EXCLUDE_FILE)
    image_files, classes = _get_files(lesion_files, unzip_file, exclude_file, root_dir)
    filenames = [os.path.basename(f) for f in image_files]

    # Write to new files as PNGs
    for imf in tqdm(image_files):
        output_path = _make_output_path(output_dir, imf)
        if not os.path.exists(output_path):
            image = cv2.imread(imf, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(output_path, image)

    return filenames, classes


def _get_lesion_files(lesion_files, exclude_list, root_dir):
    """Reads the lesion files to identify relevant
    slices and returns their paths"""
    files, classes = [], []
    for lesion_file in lesion_files:
        with open(lesion_file, 'r') as f:
            f.readline()
            for line in f.readlines():
                cls, pid, sid = line.split('/')[:3]
                if pid not in exclude_list:
                    # Patch to remove a few erroneous files
                    case_str = '{}_{}_{}'.format(cls, pid, sid)
                    if case_str in _PATCH_CASES:
                        continue

                    files.append(os.path.join(root_dir, line.strip('\n')))
                    classes.append(_CLASS_MAP[cls])
    return files, classes


def _get_files(lesion_files, unzip_file, exclude_file, root_dir):
    """Gets image file paths according to given lists"""
    excluded_pids = _get_excluded_pids(exclude_file)
    files, classes = _get_lesion_files(lesion_files, excluded_pids['CP'] + excluded_pids['NCP'], root_dir)
    with open(unzip_file, 'r') as f:
        reader = list(csv.DictReader(f, delimiter=',', quotechar='|'))
        for row in reader:
            if row['label'] == 'Normal':
                pid = row['patient_id']
                sid = row['scan_id']
                if pid not in excluded_pids['Normal']:
                    new_paths = _get_source_paths(root_dir, 'Normal', pid, sid)
                    files += new_paths
                    classes += [_CLASS_MAP['Normal'] for _ in range(len(new_paths))]
    return files, classes


def _get_source_paths(root_dir, cls, pid, sid):
    """Helper function to construct source paths"""
    exam_dir = os.path.join(root_dir, cls, pid, sid)
    return list(multi_ext_file_iter(exam_dir, IMG_EXTENSIONS))


def _make_output_path(output_dir, source_path):
    """Helper function to construct output paths"""
    source_path = source_path.replace('\\', '/')
    parts = source_path.split('/')[-4:]
    output_path = os.path.join(output_dir, '_'.join(parts))
    output_path = os.path.splitext(output_path)[0] + '.png'
    return output_path


def _get_excluded_pids(exclude_file):
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
