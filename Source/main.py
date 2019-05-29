import numpy as np

from rankDataForQuery import prepareData
from rankDataForQuery import rankDataForQuery

from getRelevants import RelevatInfoReader
from qualityAnalyzers import q_anayzer



#data_locaton = '../Data'
#relevant_info_location = '../rel.txt'
#marked_data_location = '../marked_data.txt'

data_locaton = '../Big_data'
relevant_info_location = '../big_rel.txt'
marked_data_location = '../marked_big_data.txt'

# read relevant results for query
info_extractor = RelevatInfoReader(relevant_info_location)
analyzer = q_anayzer(info_extractor)

# m1
query_vecs, data_vecs, desc_len =  prepareData(data_locaton, './_Big_data.txt')
rankDataForQuery(query_vecs, data_vecs, res_analyzer = analyzer, multiplier =  np.ones(desc_len))
analyzer.Reset()
# m2
query_vecs, data_vecs, desc_len =  prepareData(data_locaton, './_Big_data_cutted.txt', marked_data_file_name = marked_data_location)
rankDataForQuery(query_vecs, data_vecs, res_analyzer = analyzer)
analyzer.Reset()
# m3
query_vecs, data_vecs, desc_len =  prepareData(data_locaton, './_Big_data_segmented.txt', marked_data_file_name = marked_data_location)
rankDataForQuery(query_vecs, data_vecs, res_analyzer = analyzer)
analyzer.Reset()