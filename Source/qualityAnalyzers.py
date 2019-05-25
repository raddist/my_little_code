import os


def fullname2short( full_name ):
    return os.path.basename(full_name)

class q_anayzer:

    def __init__(self, info_extractor):
        self.relevant_info = info_extractor
        self.first_n = 5

    def add_query_results(self, query_full_name, sorted_names):
        query_name = fullname2short(query_full_name)
        relevant_files = self.relevant_info.get_by_query(query_name)
        #print("\nRelevant files for",query_name, "/n",relevant_files)

        num_of_relevants = 0
        for i in range(0, self.first_n ):
            if sorted_names[i] in relevant_files:
                num_of_relevants += 1

        print("\nIn first",str(self.first_n), " results contains:")
        print(str(num_of_relevants), " from ", str(len(relevant_files))," for ", query_name, " file.")
