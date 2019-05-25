import cv2
import sys
import os

# read data location
data_location_name = sys.argv[1]

# read destination file
write_file = sys.argv[2]


# get data
data = []
for filename in os.listdir(data_location_name):
    name = os.path.join(data_location_name, filename)

    img = cv2.imread(name, 1)

    # Select interesting region
    Rect2d = cv2.selectROI(img, False)
    cv2.waitKey(0)
    x, y, w, h = Rect2d
    print("ROI coordinates for ", name,": [", x, y, w, h, "]\n")

    data.append([filename, (x, y, w, h)] )

with open(write_file, 'w') as fw:
    for elem in data:
        file_str = elem[0] + "\t - " + str(elem[1]) + "\r\n"
        fw.write(file_str)