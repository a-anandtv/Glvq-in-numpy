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
from glvq_utilities import plot2d



#
# 
########################################
if __name__ == '__main__':

    prototype_per_class = 7
    epochs = 100
    input_data = load_iris().data
    data_labels = load_iris().target

    glvq_model = Glvq(prototypes_per_class=prototype_per_class, epochs=epochs)
    glvq_model.load_data(input_data, data_labels)
    glvq_model.initialize_prototypes()
    glvq_model.setVisualizeOn(dimensions=(3, 2))
    glvq_model.fit()

    # plot2d (plt, "Iris Data", glvq_model.input_data, glvq_model.input_data_labels,
        # glvq_model.prototypes, glvq_model.prototype_labels, dimensions=(1,3))

