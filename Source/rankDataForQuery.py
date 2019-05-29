import cv2
import sys
import os
import numpy as np

from skimage.segmentation import slic
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries

from copy import copy, deepcopy

from doRankingData import doRankingData
from getRoiArea import ROIAreaReader
from descriptorsGenerator import load_descriptors
from getAreaDescriptor import get_img_vectors


def crop(img, cutter):
    x, y, w, h = cutter
    return img[y:y + h, x:x + w]


def cut_image( image, cutter_fn ):
    parts = []
    cuts = cutter_fn(image)
    for cutter in cuts:
        parts.append( crop(image, cutter) )

    return parts


def prepareData(data_location_name, indexed_data, marked_data_file_name = None):
    roiReader = None
    if marked_data_file_name is not None:
        roiReader = ROIAreaReader(marked_data_file_name)
    # get data
    data_vecs = load_descriptors(indexed_data)
    query_vecs = []
    for filename in os.listdir(data_location_name):
        name = os.path.join(data_location_name, filename)
        img = cv2.imread(name, 1)

        if marked_data_file_name is not None:
            roi_cutter = roiReader.get_cutter(filename)
            img = crop(img, roi_cutter)

        img_vectors = get_img_vectors(img)
        for vec in img_vectors:
            query_vecs.append( [vec, filename])

    desc_len = 0
    # collect descriptor len
    if len(query_vecs) > 0:
        desc_len = len(query_vecs[0][0])

    return query_vecs, data_vecs, desc_len


def rankDataForQuery(queries, data, res_analyzer = None, multiplier = None, count_equal = False):

    for query in queries:
        print(query)
        sorted_names = doRankingData(query, data, multiplier, count_equal = count_equal)

        if res_analyzer is not None:
            res_analyzer.add_query_results( query[1], sorted_names, count_equal )
        #else:
        #    print("\n New results for ",query[1] ,": \n")
        #    for name in sorted_names:
        #        print(name)
    
    res_analyzer.ShowResults()
    return res_analyzer.GetNdcg()