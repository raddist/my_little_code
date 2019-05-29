import cv2
import sys
import os

import re

# read data location
data_location_name = sys.argv[1]

# read destination file
write_file = sys.argv[2]


# get data
img_names = []
for filename in os.listdir(data_location_name):
    ex_index = -1
    m = re.search("\d", filename)
    if m is None :
        m = re.search("\.", filename)
    
    ex_index = m.start()

    #print(m,filename )
    img_names.append( dict(key=filename[0:ex_index], val= filename) )

#print(img_names)

with open(write_file, 'w') as fw:
    for elem in img_names:
        file_str = str(elem['val']) + ":" 
        for res in img_names:
            if res['key'] == elem['key']:
                file_str += ("," + res['val'])

        fw.write(file_str + "\r\n")