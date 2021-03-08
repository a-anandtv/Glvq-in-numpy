import numpy as np
from glvq_utilities import distances, sigmoid, plot2d


#
# 
########################################
class Glvq:
    """
        Class for creating and training a Generalized Learning Vector Quantization network 
            and use this network to predict for new data.

        Attributes:
            input_data: 
                (n,m) matrix of data with n data points and m features.
            input_data_labels:
                (n,) element array of data labels. Labels representing each data point 
                    respectively.
            prototypes_per_class:
                k number of prototypes per class > 1.
                Default: 1
            prorotype_init:
                Initialization type for the prototype.
                Possible values: 
                    "class-mean": Prototypes are initialized to the mean of datapoints belonging to each class.
                    "random-input": Prototypes are initialized to a random data point in each class.
                    "random": Prototypes are mapped to random points in the data space.
                Default: 
                    Default is set to "class-mean".
            learning_rate:
                Learning rate for the model.
                Default: 0.01
            epochs:
                Total number of cycles the training has to be run for.
                Default: 30
            validation_set_percentage:
                Ratio of input data that has to be used as validation set.
                Default: 0.1 (10%)
            stop_by:
                Number of epochs to continue after the accuracy has stabilized 
                    (accuracy changes between epochs is < 10E-4).
                Default: 5
            prototypes:
                (k * number of distinct input data labels, m) matrix of prototypes
            prototype_labels:
                (k * number of distinct input data labels,) element array of labels for the prototypes.
    """


    #
    # 
    ########################################
    def __init__ (self, input_data, input_data_labels, prototypes_per_class=1, prototype_init="class-mean",
        learning_rate=0.01, epochs=30, validation_set_percentage=0.1, stop_by=5):
        """
            Function initializes model with given input data and attributes or using the default
                values for the attributes.
        """

        # Identifier to check if model has initialized correctly
        self.valid_model = False

        # Saving input data
        self.input_data = input_data

        if (len(self.input_data) != len(input_data_labels)):
            # Labels do not match to the passed input
            raise ValueError(f"Error! Invalid input label size. Input size of {len(self.input_data)}, Input label size is {len(input_data_labels)}.")
        
        self.input_data_labels = input_data_labels
        self.prototypes_per_class = prototypes_per_class
        self.prototype_init = prototype_init
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_set_percentage = validation_set_percentage
        self.stop_by = stop_by

        # Count the number of unique classes
        self.no_of_classes = len (np.unique(input_data_labels))

        if (prototype_init == "class-mean"):
            self.__generateClassMeanPrototypes__()
        elif (prototype_init == "random-input"):
            self.__generateRandomInputPrototypes__()
        elif (prototype_init == "random"):
            self.__generateRandomPrototypes__()
        else:
            raise ValueError(f"Unknown value passed for attribute prototype_init. Passed value: {prototype_init}")

        plot2d (self.input_data, "Data plot")


    #
    # 
    ########################################
    def __generateClassMeanPrototypes__(self):
        """
            Generates a set of prototypes initialized to the specific class mean
        """


    #
    # 
    ########################################
    def __generateRandomInputPrototypes__(self):
        """
            Generates a set of prototypes initialized to a random input data point
        """


    #
    # 
    ########################################
    def __generateRandomPrototypes__(self):
        """
            Generates a set of prototypes initialized randomly in the data space
        """
