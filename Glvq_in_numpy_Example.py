# 
# A Numpy implementation for Generalized LVQ
#
#
# 
########################################

import numpy as np
from Glvq import Glvq



#
# 
########################################
if __name__ == '__main__':
    # Test sample
    x = np.array ([[0, 0], [1, 1], [2, 2], [0, 1], [3, 3], [2, 4], [3, 1], [4, 4], [5, 5], [1, 4]])
    x_labels = np.array ([0, 1, 0, 2, 1, 0, 1, 0, 1, 2])
    w = np.array ([[0, 0], [1, 1], [0, 1], [1, 0], [1, 3]])
    w_labels = np.array ([0, 1, 0, 1, 2])

    glvq_model = Glvq()
    glvq_model.load_data(x, x_labels)
    # glvq_model._initialize_prototypes()
    glvq_model.initialize_prototypes("initialized", w, w_labels)
    glvq_model.fit()

    # print ("x labels ", x_labels.shape)
    # print ("w labels ", w_labels.shape)
    # dst = distances (x, w)
    # print ("fn distances ", dst)
    # print ("fn distance shape ", dst.shape)

    # w_plus = w[w_labels == x_labels[1]]
    # print ("w plus ", w_plus)
    # print ("w plus shape ", w_plus.shape)

    # print ("w indices", np.where (x_labels[1] == w_labels, x, -1))

    # w_minus = w[w_labels != x_labels[1]]
    # print ("w minus ", w_minus)
    # print ("w minus shape ", w_minus.shape)

    # dsts_to_wplus = distances (x, w_plus)
    # dsts_to_wminus = distances (x, w_minus)

    # print ("distance to w+ ", dsts_to_wplus)
    # print ("distance to w- ", dsts_to_wminus)

    # print ("argmin to w+ ", np.argmin (dsts_to_wplus, axis=1))
    # print ("argmin to w- ", np.argmin (dsts_to_wminus, axis=1))

    # print ("w+ ", w_plus[np.argmin (dsts_to_wplus, axis=1)])

