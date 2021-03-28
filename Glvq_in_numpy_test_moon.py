# 
# A Numpy implementation for Generalized LVQ
# Testing with iris dataset
#
# By: Akash Anand
########################################

import numpy as np
from sklearn.datasets import make_moons

###
from Glvq import Glvq
from glvq_utilities import plot2d



#
# 
########################################
if __name__ == '__main__':

    prototype_per_class = 9
    epochs = 30
    input_data, data_labels = make_moons(n_samples=1000, noise=0.1993)

    glvq_model = Glvq(prototypes_per_class=prototype_per_class, epochs=epochs)
    glvq_model.load_data(input_data, data_labels)
    # glvq_model.normalize_data()
    glvq_model.initialize_prototypes("class-mean")
    # print (glvq_model.input_data)
    glvq_model.setVisualizeOn(dimensions=(0, 1), showAccuracy=True)
    glvq_model.fit()

    # plot2d (plt, "Iris Data", glvq_model.input_data, glvq_model.input_data_labels,
        # glvq_model.prototypes, glvq_model.prototype_labels, dimensions=(1,3))

