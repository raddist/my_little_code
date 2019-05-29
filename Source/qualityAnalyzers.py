import os
import math

log_file_name = './log.txt'


def fullname2short( full_name ):
    return os.path.basename(full_name)

class q_anayzer:

    def __init__(self, info_extractor):
        self.relevant_info = info_extractor
        self.first_n = 10

        self.Reset()

        with open(log_file_name, 'w') as fw:
            fw.write("")

    def add_query_results(self, query_full_name, sorted_names, count_equal):
        query_name = fullname2short(query_full_name)
        relevant_files = self.relevant_info.get_by_query(query_name)
        len_rel = len(relevant_files)
        if not count_equal:
            len_rel -= 1
        #print("\nRelevant files for",query_name, "/n",relevant_files)

        dcg = 0.0
        der = 0.0
        num_of_relevants = 0
        for i in range(0, self.first_n ):
            if i < len_rel:
                der += 1/math.log(i+2, 2)

            if sorted_names[i] in relevant_files:
                num_of_relevants += 1
                dcg += 1/math.log(i+2, 2)

        self.ndcg       += (dcg / der)
        self.positives  += num_of_relevants
        self.summ       += min( self.first_n, len_rel )
        self.quries_num += 1


        #print("\nIn first",str(self.first_n), " results contains:")
        #print(str(num_of_relevants), " from ", str(len(relevant_files))," for ", query_name, " file.")
        log_str = "For " + query_name + " " + str(num_of_relevants) + "/" + str(len_rel) + "\n"
        with open(log_file_name, 'a') as fw:
            fw.write(log_str)

    def ShowResults(self):
        print("\n Results: ", str(self.positives), " from ", str(self.summ), " elements. NDCG =", str(self.ndcg / self.quries_num))

    def GetNdcg(self):
        return self.ndcg / self.quries_num

    def Reset(self):
        self.positives = 0
        self.summ = 0

        self.ndcg = 0
        self.quries_num = 0