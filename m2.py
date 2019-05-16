import cv2
import sys
import os

from shutil import copyfile
from shutil import rmtree

import descriptors as dsc
import pairwise as pws

import practice_tools as pt


def get_img_cutters(image):
	height, width, channels = image.shape

	n_h = height // 100
	h2 =  height % 100
	if h2 > 50:
		n_h += 1

	n_w = width // 100
	w2 =  width % 100
	if w2 > 50:
		n_w += 1

	crop_w = width // n_w
	crop_h = height // n_h
	#print('crop_w ',crop_w ,', crop_h', crop_h ,', n_w', n_w ,', n_h', n_h )

	cutters = []
	for row in range(1, n_h):
		for col in range(1, n_w):
			cutters.append([(col-1)*crop_w, (row-1)*crop_h, crop_w, crop_h])

	for row in range(1, n_h):
		cutters.append([(n_w-1)*crop_w, (row-1)*crop_h, width-(n_w-1)*crop_w, crop_h])

	for col in range(1, n_w):
		cutters.append([(col-1)*crop_w, (n_h-1)*crop_h, crop_w, height - (n_h-1)*crop_h])

	cutters.append([(n_w-1)*crop_w, (n_h-1)*crop_h, width-(n_w-1)*crop_w, height - (n_h-1)*crop_h])

	return cutters


# read query name
query_img_name = sys.argv[1]

# read data location
data_location_name = sys.argv[2]

# get query
query_img = cv2.imread(query_img_name, 1)
# Select interesting region
Rect2d = cv2.selectROI(query_img, False)
cv2.waitKey(0)
x, y, w, h = Rect2d
print("ROI coordinates: [", x, y, w, h, "]")
query_img = query_img[y:y + h, x:x + w]

# get data
data = []
for filename in os.listdir(data_location_name):
	name = os.path.join(data_location_name, filename)

	img = cv2.imread(name, 1)
	data.append( dict(image = img, imgname = name) )

#sorted_names = pt.ranking_by_img_areas_vectors( dict(image = query_img, imgname = 'query'), data)
sorted_names = pt.rank_images_by_vectors( dict(image = query_img, imgname = 'query'), data, cutter_fn = get_img_cutters )

#show results
result_dir_name = 'Practice_results'
if not os.path.exists(result_dir_name):
	os.mkdir( result_dir_name )
else:
	rmtree( result_dir_name )
	os.mkdir( result_dir_name )

for (i, res_name) in enumerate(sorted_names):
	copyfile( res_name, os.path.join('./',result_dir_name,str(str(i) + '_' + os.path.basename(res_name))) )