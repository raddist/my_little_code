import cv2
import sys
import os
import re
import numpy as np

from skimage.segmentation import slic
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries

from copy import copy, deepcopy

from doRankingData import doRankingData
from getRoiArea import ROIAreaReader
from getAreaDescriptor import get_area_vector




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


def write_descriptors(data_vecs, write_file_name = None):

    if write_file_name is None:
        return

    with open(write_file_name, 'w') as fw:
        for desc_and_name in data_vecs:
            cached_str = desc_and_name[1] + ":" + str(desc_and_name[0]).replace("\n", "") + "\n"
            fw.write(cached_str)

def read_descriptors(read_file_name = None):

    descriptors = []
    with open(read_file_name, 'r') as fw:
            file = fw.readlines()

            for line in file:
                m1 = re.search(":", line)
                m2 = re.search("\[", line)
                m3 = re.search("\]", line)

                if m1 is not None and m2 is not None and m3 is not None:
                    ex_name_ind         = m1.start()
                    ex_desc_start_ind   = m2.start()
                    ex_desc_end_ind     = m3.start()

                    img_name = line[0:ex_name_ind].rstrip()
                    elem_str_arr = line[ex_desc_start_ind+1:ex_desc_end_ind].split(' ')
                    descriptor = np.array( [float(x) for x in elem_str_arr if x] )

                    descriptors.append( [descriptor, img_name] )

    return descriptors



def generate_descriptors(data_location_name, cutter_fn = None, segments_maker = None):

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

    file_name = data_location_name.replace("../", "_")
    if cutter_fn is not None:
        file_name += "_cutted"
    if segments_maker is not None:
        file_name += "_segmented"
    file_name += ".txt"

    write_descriptors(data_vecs, file_name)

def load_descriptors(read_file_name):
    return read_descriptors(read_file_name)
