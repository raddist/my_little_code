import cv2
import sys
import os

from shutil import copyfile
from shutil import rmtree

import descriptors as dsc
import pairwise as pws

from segment import ranking_by_segment
#
#descriptors = []
#
##download data and create vector model
#directory = "./Data"
#for filename in os.listdir(directory):
#	name = os.path.join(directory,filename)
#	img = cv2.imread(name, 1)
#	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#	cv2.imshow(name, img)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()
#
#	#create vectors
#	descriptors.extend( dsc.make_descriptors(img, name) )
#
##download key image
#keyName = "./key.bmp"
#img = cv2.imread(keyName, 1)
##cv2.imshow(keyName, img)
#Rect2d = cv2.selectROI(img, False)
#cv2.waitKey(0)
#
##create vector for key image
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#key_desc = dsc.make_key_descriptors(img, Rect2d)
#
#for kd in key_desc:
#	print(kd[0],"\n")
#print("dsfsdfsdf")
#for kd in descriptors:
#	print(kd[0],"\n")
#
#
##compare vector with model
#sorted_names = dsc.rank_vectors(key_desc, descriptors)
#
#print("Results")
##show results
#for filedesc in sorted_names:
#	#print(filedesc)
#	filename = filedesc[0]
#	print(filename)
#	img = cv2.imread(filename, 1)
#	cv2.imshow(filename, img)
#	cv2.imshow("area", filedesc[1])	#debug
#	cv2.imshow("key_area", filedesc[2])	#debug
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()

# read query
query_img_name = sys.argv[1]
query_img = cv2.imread(query_img_name, 0)

print(query_img_name)


# Select interesting region
Rect2d = cv2.selectROI(query_img, False)
cv2.waitKey(0)
x, y, w, h = Rect2d
print(x, y, w, h)

segment_query = True
if (w > 0 and h > 0):
	segment_query = False
	query_img = query_img[y:y + h, x:x + w]


# read data location
data_location_name = sys.argv[2]

data = []
for filename in os.listdir(data_location_name):
	name = os.path.join(data_location_name, filename)
	img = cv2.imread(name, 0)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#cv2.imshow(name, img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#create vectors

	data.append( dict(image = img, imgname = name) )

sorted_names = pws.ranking_by_color_hist(query_img, data)
#sorted_names = ranking_by_segment(query_img, data, query_segmentation = segment_query)

#show results
result_dir_name = 'Results'
if not os.path.exists(result_dir_name):
	os.mkdir( result_dir_name )
else:
	rmtree( result_dir_name )
	os.mkdir( result_dir_name )

for (i, res_name) in enumerate(sorted_names):
	copyfile( res_name, os.path.join('./',result_dir_name,str(str(i) + '_' + os.path.basename(res_name))) )