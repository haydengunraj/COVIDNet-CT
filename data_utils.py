import os
import cv2
import glob
import itertools
import numpy as np

IMG_EXTENSIONS = ('png', 'jpg', 'jpeg', 'tif', 'bmp')
HU_WINDOW_WIDTH = 1500
HU_WINDOW_CENTER = -600


def hu_to_uint8(hu_images, window_width, window_center):
    """Converts HU images to uint8 images"""
    images = (hu_images.astype(np.float32) - window_center + window_width/2)/window_width
    uint8_images = np.uint8(255.0*np.clip(images, 0.0, 1.0))
    return uint8_images


def ensure_uint8(data, window_width=HU_WINDOW_WIDTH, window_center=HU_WINDOW_CENTER):
    """Converts non-uint8 data to uint8 and applies window level to HU data"""
    if data.dtype != np.uint8:
        if float(data.max()) - float(data.min()) > 255:  # Note: ptp() not used due to overflow issues
            # Assume HU
            data = hu_to_uint8(data, window_width, window_center)
        else:
            # Assume uint8 range with incorrect dtype
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
    extensions = set(ext.lower() for ext in extensions)
    patterns = ['*.' + ext for ext in extensions]
    if os.name != 'nt':  # on non-Windows OS, file extensions are case sensitive
        patterns += ['*.' + ext.upper() for ext in extensions]
    return itertools.chain.from_iterable(
        glob.iglob(os.path.join(directory, pat)) for pat in patterns)


def exterior_exclusion(image):
    """Removes visual features exterior to the patient's body"""
    # Create initial binary image
    filt_image = cv2.GaussianBlur(image, (5, 5), 0)
    filt_image.shape = image.shape  # ensure channel dimension is preserved if present
    thresh = cv2.threshold(filt_image[filt_image > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bin_image = filt_image > thresh

    # Find body contour
    body_cont = body_contour(bin_image.astype(np.uint8))

    # Exclude external regions by replacing with bg mean
    body_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(body_mask, [body_cont], 0, 1, -1)
    body_mask = body_mask.astype(bool)
    bg_mask = (~body_mask) & (image > 0)
    bg_dark = bg_mask & (~bin_image)  # exclude bright regions from mean
    bg_mean = np.mean(image[bg_dark])
    image[bg_mask] = bg_mean
    return image
