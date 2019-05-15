import cv2
import sys
import os

from shutil import copyfile
from shutil import rmtree

import descriptors as dsc
import pairwise as pws

import practice_tools as pt


# read query name
query_img_name = sys.argv[1]

# read data location
data_location_name = sys.argv[2]

# get query
query_img = cv2.imread(query_img_name, 1)

# get data
data = []
for filename in os.listdir(data_location_name):
	name = os.path.join(data_location_name, filename)

	img = cv2.imread(name, 1)
	data.append( dict(image = img, imgname = name) )

#sorted_names = pt.ranking_by_img_vectors( dict(image = query_img, imgname = 'query'), data)
sorted_names = pt.rank_images_by_vectors( dict(image = query_img, imgname = 'query'), data, cutter_fn = )

#show results
result_dir_name = 'Practice_results'
if not os.path.exists(result_dir_name):
	os.mkdir( result_dir_name )
else:
	rmtree( result_dir_name )
	os.mkdir( result_dir_name )

for (i, res_name) in enumerate(sorted_names):
	copyfile( res_name, os.path.join('./',result_dir_name,str(str(i) + '_' + os.path.basename(res_name))) )