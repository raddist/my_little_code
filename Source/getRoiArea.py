import cv2
import sys
import os

import re

class ROIAreaReader:

    def __init__(self, text_file_name):
        self.areas = {}
        with open(text_file_name, 'r') as fw:
            file = fw.readlines()

            for line in file:
                m1 = re.search(" ", line)
                m2 = re.search("\(", line)
                m3 = re.search("\)", line)

                if m1 is not None and m2 is not None and m3 is not None:
                    ex_name_ind         = m1.start()
                    ex_roi_start_ind    = m2.start()
                    ex_roi_end_ind      = m3.start()

                    query_name = line[0:ex_name_ind].rstrip()
                    results = line[ex_roi_start_ind+1:ex_roi_end_ind].split(', ')

                    self.areas[query_name] = [int(x.rstrip()) for x in results if x]

        #print(self.areas)

    def get_cutter(self, query_name):
        return self.areas[query_name]