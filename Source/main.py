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
from getRoiArea import ROIAreaReader

# объект - ground trouth
# объект - разбиватель
# объект - разделитель на сегменты
# объект - создатель векторов

# загрузка и преоразование данных

# вызов rankDataForQuery с нужными параметрами

def get_area_vector(img, mask = None):
    mean, std = cv2.meanStdDev( img, mask = mask  )
    r_mean, g_mean, b_mean = mean
    r_std, g_std, b_std = std
    vector = np.array([ r_mean[0], g_mean[0], b_mean[0], r_std[0], g_std[0], b_std[0] ])

    return vector

def get_img_vectors(img, masks = None):

    if masks is None:
        return [get_area_vector(img)]

    vectors = []
    if masks is not None:
        for mask in masks:
            vectors.append( get_area_vector(img, mask) )

    return vectors

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
    segments = slic(img, n_segments = 10, compactness=0.1, sigma = 5)

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

    masks = []
    for i in range(min_mask, max_mask+1):
        mask = copy(segments)
        mask = np.uint8(mask)
        for row in mask:
            for j, elem in enumerate(row):
                if elem == i:
                    row[j] = True
                else:
                    row[j] = False
        masks.append(mask)

    return masks


# read data location
data_location_name = sys.argv[1]


# get data
data = []
for filename in os.listdir(data_location_name):
    name = os.path.join(data_location_name, filename)

    img = cv2.imread(name, 1)
    data.append( dict(image = img, imgname = filename) )

# get queries
queries = []
for filename in os.listdir(data_location_name):
    name = os.path.join(data_location_name, filename)

    img = cv2.imread(name, 1)
    queries.append( dict(image = img, imgname = filename) )

marked_queries = []
roiReader = ROIAreaReader('../marked_data.txt')
for filename in os.listdir(data_location_name):
    name = os.path.join(data_location_name, filename)

    img = cv2.imread(name, 1)
    roi_cutter = roiReader.get_cutter(filename)

    marked_queries.append( dict(image = crop(img, roi_cutter), imgname = filename) )

# read relevant results for query
info_extractor = RelevatInfoReader('../rel.txt')
analyzer = q_anayzer(info_extractor)

rankDataForQuery(marked_queries, data, get_img_vectors, cutter_fn = get_img_cutters, segments_maker = get_segments_masks,  res_analyzer = analyzer)