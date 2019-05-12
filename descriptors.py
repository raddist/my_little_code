import cv2
import numpy as np

def make_descriptor( img, name ):
	mean, std = cv2.meanStdDev( img )
	arr = np.array([mean, std])
	#print(arr, name)
	return (arr, name, img)	#debug


def intersection(rect_a, rect_b):
    x = max(rect_a[0], rect_b[0])
    y = max(rect_a[1], rect_b[1])
    w = min(rect_a[0]+rect_a[2], rect_b[0]+rect_b[2]) - x
    h = min(rect_a[1]+rect_a[3], rect_b[1]+rect_b[3]) - y

    if w < 0 or h < 0:
        return None

    return (x, y, w, h)

def divide_image( img, rect = None ):
	parts = []
	height, width = img.shape

	#print(height, width)

	Q_HEIGHT = 100
	Q_WIDTH = 100

	row = 0
	while (row + Q_HEIGHT) < height:

		col = 0
		while (col + Q_WIDTH) < width:

			if (rect is None) or (intersection(rect, [col, row, col + Q_WIDTH, row + Q_HEIGHT])):
				parts.append( img[row:row + Q_HEIGHT, col:col + Q_WIDTH] )

			#cv2.imshow("filename", img[row:row + Q_HEIGHT, col:col + Q_WIDTH])
			#cv2.waitKey(0)
			col += Q_WIDTH

		row += Q_HEIGHT

	return parts


def make_descriptors(img, name):
	parts = []
	parts = divide_image( img )

	descs = []
	for part in parts:
		descs.append( make_descriptor(part, name) )

	return descs


def make_key_descriptors(img, rect):
	parts = []
	parts = divide_image( img, rect )

	descs = []
	for part in parts:
		descs.append( make_descriptor(part, "key") )

	return descs

def rank_vectors(key_dsc, dsc):
	sorted_names = []
	#print(key_dsc)
	for  key_d in key_dsc:
		min_dist = None
		min_name = None
		min_img = None
		for descriptor in dsc:
			dist = np.linalg.norm( descriptor[0] - key_d[0] )
			name = descriptor[1]

			if min_dist is None or min_dist > dist:
				min_dist = dist
				min_name = name
				min_img = descriptor[2]	#debug

		sorted_names.append( (min_name, key_d[2], min_img) )	#debug

	return sorted_names

	

