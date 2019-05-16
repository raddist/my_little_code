import cv2
import numpy as np

from skimage.segmentation import slic
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries

from copy import copy, deepcopy


def get_area_vector(img, mask = None):
    mean, std = cv2.meanStdDev( img, mask = mask  )
    r_mean, g_mean, b_mean = mean
    r_std, g_std, b_std = std
    vector = np.array([ r_mean[0], g_mean[0], b_mean[0], r_std[0], g_std[0], b_std[0] ])

    return vector

def crop(img, cutter):
    x, y, w, h = cutter
    return img[y:y + h, x:x + w]

def get_img_vectors(img, masks = None):

    if masks is None:
        return [get_area_vector(img)]

    vectors = []
    if masks is not None:
        for mask in masks:
            vectors.append( get_area_vector(img, mask) )

    return vectors

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


def get_segments_masks(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segments = slic(img, n_segments = 10, compactness=0.1, sigma = 5)

    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (20))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img, segments))
    plt.axis("off")
     
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


def visualize_vecs(key_vectors, img_vectors):
    for vec_and_name in img_vectors:
        vector = vec_and_name[0]
        plt.plot( vector[0], vector[1],'ko' )

    for vec in key_vectors:
        plt.plot( vec[0], vec[1],'ro' )

    plt.show()


def cut_image( image, cutter_fn ):
    parts = []
    cuts = cutter_fn(image)
    for cutter in cuts:
        parts.append( crop(image, cutter) )

    return parts
    
    
def rank_images_by_vectors(query_img_data, data,
                           cutter_fn = None,
                           do_segmentation = False,
                           info_collector = None,
                           info = False):

    q_vector = get_img_vectors(query_img_data['image'])
    #print('Query vector: ', q_vector)

    data_vecs = []
    for elem in data:
        sub_images = [elem['image']] if cutter_fn is None else cut_image(elem['image'], cutter_fn)
        for sub_image in sub_images:
            img_vectors = []
            if not do_segmentation:
                img_vectors = get_img_vectors(sub_image)
            else:
                img_masks = get_segments_masks(sub_image)
                img_vectors = get_img_vectors(sub_image, masks = img_masks)
            
            for vector in img_vectors:
                data_vecs.append( [vector, elem['imgname']] )

    #   for (i, (vector, name)) in enumerate(data_vecs):
    #       print('Vector = ', vector , ', name = ', name)

    print(len(data_vecs))
    distances = rank_vectors(q_vector[0], data_vecs)

    metadata = None
    if info:
        metadata = distances
    #print(distances)


    sorted_names = []
    print("results:/n")
    for distance in distances:
        sorted_names.append(distance[1])
        print(distance[1])

    return sorted_names, metadata