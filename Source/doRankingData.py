import cv2
import numpy as np

from skimage.segmentation import slic
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries

from copy import copy, deepcopy



def rank_vectors(q_vector, data_vectors):

    distances = {}
    for (i, (vector, name)) in enumerate(data_vectors):
        #print(q_vector, vector)
        if name in distances:
            distances[name] = min( np.linalg.norm( q_vector - vector ), distances[name])
        else:
            distances[name] = np.linalg.norm( q_vector - vector )

    # sort the results
    distances = sorted([(v, k) for (k, v) in distances.items()])

    return distances


def doRankingData(query, data):

    #print(len(data_vecs))
    distances = rank_vectors(query[0], data)

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

    