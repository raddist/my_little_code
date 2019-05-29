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

# разбиение
# сегментирование
# превращение в вектор
# 
# перебор запросов
#     получение резудбтатов для запроса
#     составление метрик


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


def crop(img, cutter):
    x, y, w, h = cutter
    return img[y:y + h, x:x + w]


def cut_image( image, cutter_fn ):
    parts = []
    cuts = cutter_fn(image)
    for cutter in cuts:
        parts.append( crop(image, cutter) )

    return parts


#def rankDataForQuery(queries, data, vector_maker, cutter_fn = None, segments_maker = None, res_analyzer = None):
def rankDataForQuery(data_location_name, marked_data_file_name = None, cutter_fn = None, segments_maker = None, res_analyzer = None):

    roiReader = None
    if marked_data_file_name is not None:
        roiReader = ROIAreaReader(marked_data_file_name)
    # get data
    data_vecs = []
    query_vecs = []
    for filename in os.listdir(data_location_name):

        print("Process ",filename, "file\n")
        name = os.path.join(data_location_name, filename)

        img = cv2.imread(name, 1)
        sub_images = [img] if cutter_fn is None else cut_image(img, cutter_fn)
        for sub_image in sub_images:
            img_vectors = []
            if segments_maker is None:
                img_vectors = get_img_vectors(sub_image)
            else:
                img_masks = segments_maker(sub_image)
                img_vectors = get_img_vectors(sub_image, masks = img_masks)
            
            for vector in img_vectors:
                data_vecs.append( [vector, filename] )

        if marked_data_file_name is not None:
            roi_cutter = roiReader.get_cutter(filename)
            img = crop(img, roi_cutter)

        query_vecs.append( [get_img_vectors(img), filename])



    for query in query_vecs:
        sorted_names = doRankingData(query, data_vecs)

        if res_analyzer is not None:
            res_analyzer.add_query_results( query[1], sorted_names )
        #else:
        #    print("\n New results for ",query[1] ,": \n")
        #    for name in sorted_names:
        #        print(name)
    
    res_analyzer.ShowResults()