import os
import cv2
import glob
import itertools
import numpy as np


def hu_to_uint8(hu_images, window_width, window_center):
    """Converts HU images to uint8 images"""
    images = (hu_images.astype(np.float) - window_center + window_width/2)/window_width
    uint8_images = np.uint8(255.0*np.clip(images, 0.0, 1.0))
    return uint8_images


def ensure_uint8(data, window_width=1500, window_center=-600):
    """Converts non-uint8 data to uint8 and applies window level to HU data"""
    if data.dtype != np.uint8:
        if data.ptp() > 255:
            # Assume HU
            data = hu_to_uint8(data, window_width, window_center)
        data = data.astype(np.uint8)
    return data


def find_contours(binary_image):
    """Helper function for finding contours"""
    return cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]


def body_contour(binary_image):
    """Helper function to get body contour"""
    contours = find_contours(binary_image)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    body_idx = np.argmax(areas)
    return contours[body_idx]


def auto_body_crop(image, scale=1.0):
    """Roughly crop an image to the body region"""
    # Create initial binary image
    filt_image = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(filt_image[filt_image > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bin_image = np.uint8(filt_image > thresh)
    erode_kernel = np.ones((7, 7), dtype=np.uint8)
    bin_image = cv2.erode(bin_image, erode_kernel)

    # Find body contour
    body_cont = body_contour(bin_image).squeeze()

    # Get bbox
    xmin = body_cont[:, 0].min()
    xmax = body_cont[:, 0].max() + 1
    ymin = body_cont[:, 1].min()
    ymax = body_cont[:, 1].max() + 1

    # Scale to final bbox
    if scale > 0 and scale != 1.0:
        center = ((xmax + xmin)/2, (ymin + ymax)/2)
        width = scale*(xmax - xmin + 1)
        height = scale*(ymax - ymin + 1)
        xmin = int(center[0] - width/2)
        xmax = int(center[0] + width/2)
        ymin = int(center[1] - height/2)
        ymax = int(center[1] + height/2)

    return image[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax)


def multi_ext_file_iter(directory, extensions):
    """Creates a multi-extension file iterator"""
    patterns = ['*.' + ext.lower() for ext in extensions]
    return itertools.chain.from_iterable(
        glob.iglob(os.path.join(directory, pat)) for pat in patterns)
