import cv2
import numpy as np

from skimage.segmentation import slic
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries

from copy import copy, deepcopy


def get_segments_masks(img):
	segments = slic(img, n_segments = 10, compactness=0.1, sigma = 5)

	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (20))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(img, segments))
	plt.axis("off")
	 
	# show the plots
	plt.show()


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

def get_img_vectors(img, masks = None):
	vectors = []

	for mask in masks:
		mean, std = cv2.meanStdDev( img, mask = mask  )
		#print(mean[0][0], *std)
		vectors.append([ mean[0][0], std[0][0] ])

	return vectors

def visualize_vecs(key_vectors, img_vectors):
	for vec_and_name in img_vectors:
		vector = vec_and_name[0]
		plt.plot( vector[0], vector[1],'ko' )

	for vec_and_name in key_vectors:
		vector = vec_and_name[0]
		plt.plot( vector[0], vector[1],'ro' )

	plt.show()



def ranking_by_segment( key_img, data, query_segmentation = True ):

	height, width = key_img.shape
	key_masks = get_segments_masks(key_img) if query_segmentation else np.array([ np.ones((height, width), np.uint8) ])
	print(len(key_masks))
	print("mask:", key_masks.shape, " img:", key_img.shape )

	key_vectors = get_img_vectors(key_img, key_masks)
	key_vecs = []
	for vector in key_vectors:
			key_vecs.append( [vector, 'query'] )
	

	data_vecs = []
	for elem in data:
		img_masks = get_segments_masks(elem['image'])
		img_vectors = get_img_vectors(elem['image'], img_masks)
		for vector in img_vectors:
			data_vecs.append( [vector, elem['imgname']] )

	print(len(data_vecs))
	visualize_vecs(key_vecs, data_vecs)

	sorted_names = []
	for elem in data:
		sorted_names.append(elem['imgname'])

	return sorted_names