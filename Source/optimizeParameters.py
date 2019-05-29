import numpy as np
import scipy.optimize as optimize

from rankDataForQuery import prepareData
from rankDataForQuery import rankDataForQuery

from getRelevants import RelevatInfoReader
from qualityAnalyzers import q_anayzer



data_locaton = '../Data'
relevant_info_location = '../rel.txt'
marked_data_location = '../marked_data.txt'

#data_locaton = '../Big_data'
#relevant_info_location = '../big_rel.txt'
#marked_data_location = '../marked_big_data.txt'

# read relevant results for query
info_extractor = RelevatInfoReader(relevant_info_location)
analyzer = q_anayzer(info_extractor)


query_vecs, data_vecs, desc_len =  prepareData(data_locaton, './_Data_cutted.txt', marked_data_file_name = marked_data_location)

#arr = [[1, 1, 1, 1, 1, 1],
#[2, 2, 2, 1, 1, 1],
#[3, 3, 3, 1, 1, 1],
#[1, 1, 1, 3, 3, 3],
#[1, 1, 1, 2, 2, 2]
#]
#for line in arr:
#    rankDataForQuery(query_vecs, data_vecs, res_analyzer = analyzer, multiplier =  line )
#    analyzer.Reset()

def func( mult ):
    print( mult )
    res = rankDataForQuery(query_vecs, data_vecs, res_analyzer = analyzer, multiplier =  mult )
    analyzer.Reset()
    return -res


xopt = optimize.fmin(func, np.ones(desc_len), disp=True)


