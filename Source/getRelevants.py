import cv2
import sys
import os

import re


class RelevatInfoReader:

    def __init__(self, text_file_name):

        self.info = {}
        with open(text_file_name, 'r') as fw:
            file = fw.readlines()

            for line in file:
                m = re.search("\:", line)

                if m is not None :
                    ex_ind = m.start()
                    query_name = line[0:ex_ind]
                    results = line[ex_ind+1:].split(',')

                    self.info[query_name] = [x.rstrip() for x in results if x]

        #print(self.info)

    def get_by_query(self, query_name):
        #print(self.info)
        return self.info[query_name]