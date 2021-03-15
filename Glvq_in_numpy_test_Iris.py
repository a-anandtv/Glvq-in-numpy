# 
# A Numpy implementation for Generalized LVQ
#
#
# 
########################################

import numpy as np
from sklearn.datasets import load_iris

###
from Glvq import Glvq



#
# 
########################################
if __name__ == '__main__':

    prototype_per_class = 2
    input_data = load_iris().data
    data_labels = load_iris().target

    glvq_model = Glvq(prototypes_per_class=prototype_per_class)
    glvq_model.load_data(input_data, data_labels)
    glvq_model.initialize_prototypes()
    glvq_model.fit()

