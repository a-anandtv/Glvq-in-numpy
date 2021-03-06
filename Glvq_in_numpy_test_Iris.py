# 
# A Numpy implementation for Generalized LVQ
# Testing with iris dataset
#
# By: Akash Anand
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

    prototype_per_class = 6
    epochs = 50
    iris = load_iris()
    input_data = iris.data
    data_labels = iris.target

    glvq_model = Glvq(prototypes_per_class=prototype_per_class, epochs=epochs)
    glvq_model.load_data(input_data, data_labels)
    glvq_model.normalize_data()
    glvq_model.initialize_prototypes("class-mean")
    # print (glvq_model.input_data)
    glvq_model.setVisualizeOn(dimensions=(0, 2), showAccuracy=True)
    glvq_model.fit()

    # plot2d (plt, "Iris Data", glvq_model.input_data, glvq_model.input_data_labels,
        # glvq_model.prototypes, glvq_model.prototype_labels, dimensions=(1,3))

