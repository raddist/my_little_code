import cv2
import sys
import os

import numpy as np

from shutil import copyfile
from shutil import rmtree
from skimage.segmentation import slic
from copy import copy, deepcopy

from descriptorsGenerator import generate_descriptors


def get_img_cutters(image):
    height, width, channels = image.shape

    n_h = height // 100
    h2 =  height % 100
    if h2 > 50:
        n_h += 1

    n_w = width // 100
    w2 =  width % 100
    if w2 > 50:
        n_w += 1

    crop_w = width // n_w
    crop_h = height // n_h
    #print('crop_w ',crop_w ,', crop_h', crop_h ,', n_w', n_w ,', n_h', n_h )

    cutters = []
    for row in range(1, n_h):
        for col in range(1, n_w):
            cutters.append([(col-1)*crop_w, (row-1)*crop_h, crop_w, crop_h])

    for row in range(1, n_h):
        cutters.append([(n_w-1)*crop_w, (row-1)*crop_h, width-(n_w-1)*crop_w, crop_h])

    for col in range(1, n_w):
        cutters.append([(col-1)*crop_w, (n_h-1)*crop_h, crop_w, height - (n_h-1)*crop_h])

    cutters.append([(n_w-1)*crop_w, (n_h-1)*crop_h, width-(n_w-1)*crop_w, height - (n_h-1)*crop_h])
    #print("\n",str(cutters))

    return cutters

def crop(img, cutter):
    x, y, w, h = cutter
    return img[y:y + h, x:x + w]

def cut_image( image, cutter_fn ):
    parts = []
    cuts = cutter_fn(image)
    for cutter in cuts:
        parts.append( crop(image, cutter) )

    return parts


def get_segments_masks(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("  Start segmentstion:")
    segments = slic(img, n_segments = 10, compactness=0.1, sigma = 5)
    print("  end segmentstion!")


    min_mask = np.min(segments)
    max_mask = np.max(segments)

    print("  Start copy:")
    masks = []
    for i in range(min_mask, max_mask+1):
        print(str(i))
        mask = copy(segments)
        mask = np.uint8(mask)
        print(str(i))
        h,w = mask.shape
        for row in range(0,h):
            for col in range(0,w):
                if mask[row][col] == i:
                    mask[row][col] = 1
                else:
                    mask[row][col] = 0
        masks.append(mask)
    print("  end copy!")


    return masks


data_locaton = '../Data'


# m1
generate_descriptors(data_locaton)
# m2
generate_descriptors(data_locaton,  cutter_fn = get_img_cutters )
# m3
generate_descriptors(data_locaton,  cutter_fn = None, segments_maker = get_segments_masks)