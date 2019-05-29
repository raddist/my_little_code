import cv2
import numpy as np

from skimage.segmentation import slic
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from math import *

from copy import copy, deepcopy



def rank_vectors(query, data_vectors, multiplier = None, count_equal = False):

    mult = 1 
    if multiplier is not None:
        mult = multiplier

    distances = {}
    for (i, (vector, name)) in enumerate(data_vectors):
        if not count_equal and name == query[1]:
            continue

        q_vector = query[0]
        #print(q_vector, vector)
        if name in distances:
            distances[name] = min( np.linalg.norm( (q_vector - vector) * mult ), distances[name])
        else:
            distances[name] = np.linalg.norm( (q_vector - vector) * mult )

    # sort the results
    distances = sorted([(v, k) for (k, v) in distances.items()])

    return distances


def doRankingData(query, data, multiplier = None, count_equal = False):

    #print(len(data_vecs))
    distances = rank_vectors(query, data, multiplier, count_equal)

    #metadata = None
    #if info:
    #    metadata = distances
    #print(distances)


    sorted_names = []
    #print("results:/n")
    for distance in distances:
        sorted_names.append(distance[1])
        #print(distance[1])

    return sorted_names#, metadata

    