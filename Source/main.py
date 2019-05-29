import cv2
import sys
import os

import numpy as np

from shutil import copyfile
from shutil import rmtree
from skimage.segmentation import slic
from copy import copy, deepcopy

from rankDataForQuery import rankDataForQuery
from getRelevants import RelevatInfoReader
from qualityAnalyzers import q_anayzer

# объект - ground trouth
# объект - разбиватель
# объект - разделитель на сегменты
# объект - создатель векторов

# загрузка и преоразование данных

# вызов rankDataForQuery с нужными параметрами

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

    # show the output of SLIC
    #fig = plt.figure("Superpixels -- %d segments" % (20))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(img, segments))
    #plt.axis("off")
     
    # show the plots
    #plt.show()

    #cv2.imshow('name', img)
    #cv2.waitKey(0)


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


data_locaton = '../Big_data'
relevant_info_location = '../big_rel.txt'
marked_data_location = '../marked_big_data.txt'

# read relevant results for query
info_extractor = RelevatInfoReader(relevant_info_location)
analyzer = q_anayzer(info_extractor)

# m1
#rankDataForQuery(data_locaton, res_analyzer = analyzer)
analyzer.Reset()
# m2
#rankDataForQuery(data_locaton, marked_data_location, cutter_fn = get_img_cutters,  res_analyzer = analyzer)
analyzer.Reset()
# m3
rankDataForQuery(data_locaton, marked_data_location, cutter_fn = None, segments_maker = get_segments_masks,  res_analyzer = analyzer)
analyzer.Reset()