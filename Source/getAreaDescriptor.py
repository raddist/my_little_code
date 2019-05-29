import cv2
import numpy as np
import math

from skimage.measure import shannon_entropy
from skimage import io, color, feature


def calc_hom_ent(hist, mask = None):
    homogeneity = 0.0
    entropy = 0.0
    for _bin in hist:
        homogeneity += _bin**2
        if not _bin == 0:
            entropy += _bin * math.log(1/_bin, 2)

    return homogeneity, entropy



def get_area_vector(img, mask = None):
    #bw_img = color.rgb2gray(img)

    mean, std = cv2.meanStdDev( img, mask = mask  )
    r_mean, g_mean, b_mean = mean
    r_std, g_std, b_std = std

    query_hist = cv2.calcHist([img], [0], mask, [16], [0, 256])
    query_hist = cv2.normalize(query_hist, query_hist).flatten()
    hom, ent = calc_hom_ent(query_hist, mask)

    vector = np.array([ r_mean[0], g_mean[0], b_mean[0], r_std[0], g_std[0], b_std[0], hom, ent ])

    return vector

def get_img_vectors(img, masks = None):
    if masks is None:
        return [get_area_vector(img)]

    vectors = []
    if masks is not None:
        for mask in masks:
            vectors.append( get_area_vector(img, mask) )

    return vectors