import cv2
import numpy as np

from skimage.segmentation import slic
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries

from copy import copy, deepcopy

from doRankingData import doRankingData

# разбиение
# сегментирование
# превращение в вектор
# 
# перебор запросов
#     получение резудбтатов для запроса
#     составление метрик


def crop(img, cutter):
    x, y, w, h = cutter
    return img[y:y + h, x:x + w]


def cut_image( image, cutter_fn ):
    parts = []
    cuts = cutter_fn(image)
    for cutter in cuts:
        parts.append( crop(image, cutter) )

    return parts


def rankDataForQuery(queries, data, vector_maker, cutter_fn = None, segments_maker = None, res_analyzer = None):

    data_vecs = []
    for elem in data:
        sub_images = [elem['image']] if cutter_fn is None else cut_image(elem['image'], cutter_fn)
        for sub_image in sub_images:
            img_vectors = []
            if segments_maker is None:
                img_vectors = vector_maker(sub_image)
            else:
                img_masks = segments_maker(sub_image)
                img_vectors = vector_maker(sub_image, masks = img_masks)
            
            for vector in img_vectors:
                data_vecs.append( [vector, elem['imgname']] )

    query_vecs = []
    for query in queries:
        query_vecs.append( [vector_maker(query['image']), query['imgname'] ])

    for query in query_vecs:
        sorted_names = doRankingData(query, data_vecs)

        if res_analyzer is not None:
            res_analyzer.add_query_results( query[1], sorted_names )
        else:
            print("\n New results for ",query[1] ,": \n")
            for name in sorted_names:
                print(name)
    