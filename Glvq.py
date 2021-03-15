# 
# A Numpy implementation for Generalized LVQ
#
#
# 
########################################

import numpy as np
import matplotlib.pyplot as plt
from glvq_utilities import squared_euclidean, sigmoid, plot2d


#
# 
########################################
class Glvq:
    """
        Class for creating and training a Generalized Learning Vector Quantization network 
            and use this network to predict for new data.

        Parameters:
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
    def __init__(self, prototypes_per_class=1, learning_rate=0.01, epochs=30, 
        validation_set_percentage=0.1, stop_by=5):
        """
            Function initializes model with given attributes or using the default
                values for the attributes.
        """

        # Identifier to check if model has initialized correctly
        self.data_loaded = False
        self.prototypes_init = False

        if (prototypes_per_class < 1):
            # Prototypes per class has to be >= 1
            raise ValueError("At least 1 prototype per class expected.")

        # Model attributes
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_set_percentage = validation_set_percentage
        self.stop_by = stop_by

        # Data collected
        self.input_data = np.array([])
        self.input_data_labels = np.array([])
        self.prototypes = np.array([])
        self.prototype_labels = np.array([])
        self.distances = np.array([])
        self.dplus = np.array([])
        self.dminus = np.array([])
        self.costs = np.array([])
        self.accuracies = np.array([])

        # For Visualization
        self.visualize_data = False
        self.view_dimensions = (0,1)
        self.showError = False
        self.showAccuracy = False


    #
    # 
    ########################################
    def load_data(self, input_data, input_data_labels):
        """
            Funtion to load data into the model.
        """
        # Saving input data
        self.input_data = input_data

        if (len(self.input_data) != len(input_data_labels)):
            # Labels do not match to the passed input
            raise ValueError(f"Error! Invalid input label size. Input size of {len(self.input_data)}, Input label size is {len(input_data_labels)}.")
        
        self.input_data_labels = input_data_labels

        # Count the number of unique classes
        self.no_of_classes = len (np.unique(self.input_data_labels))

        self.data_loaded = True

        print ("GLVQ Model: Data loaded.")


    #
    # 
    ########################################
    def initialize_prototypes(self, initialize_by="class-mean", pass_w=[], pass_w_labels=[]):
        """
            Function to initialize prototypes in the model.
        """
        self.initialize_by = initialize_by

        # Initialize the prototypes
        if (initialize_by == "initialized"):
            if (len(pass_w) != 0) and (len(pass_w_labels) != 0):
                if (self.input_data.shape[1] == pass_w.shape[1]) and (len(pass_w_labels) == len(pass_w)):
                    self.prototypes = pass_w
                    self.prototype_labels = pass_w_labels
                else:
                    raise ValueError("Attribute shapes do not match. ")
            else:
                raise RuntimeError("Missing arguments. Initial weights and weight labels expected")  
        elif (initialize_by == "class-mean"):
            if self.data_loaded:
                self.prototypes, self.prototype_labels = self._generateClassMeanPrototypes(
                    self.input_data, 
                    self.input_data_labels, 
                    self.prototypes_per_class, 
                    self.no_of_classes)
            else:
                raise RuntimeError("Data not loaded into the model. Model requires data to process for class means.")
        elif (initialize_by == "random-input"):
            if self.data_loaded:
                self.prototypes, self.prototype_labels = self._generateRandomInputPrototypes()
            else:
                raise RuntimeError("Data not loaded into the model. Model requires data to process for class means.")
        elif (initialize_by == "random"):
            self.prototypes, self.prototype_labels = self._generateRandomPrototypes()
        else:
            raise ValueError(f"Unknown value passed for attribute initialize_by. Passed value: \"{initialize_by}\"")

        self.prototypes_init = True

        print ("GLVQ Model: Prototypes generated and initialized.")



    #
    # 
    ########################################
    def _generateClassMeanPrototypes(self, xData, xLabels, k, C):
        """
            Generates a set of prototypes initialized to the specific class mean.
            
            Parameters:
                xData: Input dataset. (n,m) dimension.
                xLabels: Data labels for the input dataset. (n,) dimension.
                k: Number of prototypes per class.
                C: Number of classes.
            
            Returns:
                prototypes: Array of initialized prototypes. (k*C,m) dimension.
                prototype_labels: Array of labels for the prototypes. (k*C,) dimension.
        """

        unique_labels = np.unique(xLabels)  # (C,)
        unique_label_mask = np.equal(xLabels, np.expand_dims(unique_labels, axis=1))   # (C,n)

        # Use this mask to get matching xData
        class_data = np.where(np.expand_dims(unique_label_mask, axis=2), xData, 0)  # (C,n,m)

        # Count number of elements per class
        elmnts_per_class = np.sum(unique_label_mask, axis=1)    # (C,)

        # Initial location for prototypes (class means)
        class_means = np.sum(class_data, axis=1) / np.expand_dims(elmnts_per_class, axis=1) # (C,m)

        prototype_labels = list(unique_labels) * k
        prototypes = class_means[prototype_labels]

        return prototypes, prototype_labels



    #
    # 
    ########################################
    def _generateRandomInputPrototypes(self):
        """
            Generates a set of prototypes initialized to a random input data point
        """
        prototype_labels = list([])
        prototypes = np.array([])
        
        return prototypes, prototype_labels


    #
    # 
    ########################################
    def _generateRandomPrototypes(self):
        """
            Generates a set of prototypes initialized randomly in the data space
        """
        prototype_labels = list([])
        prototypes = np.array([])

        return prototypes, prototype_labels


    #
    # 
    ########################################
    def setVisualizeOn(self, dimensions=(0, 1), showError=False, showAccuracy=False):
        """
            Sels and initializes the model to generate visualizations for the training

            Parameters:
                dimensions: A 2D array of the dimensions to be used to plot. Defaults to dimensions 0 and 1.
        """

        if (len(dimensions) != 2):
            # Dimensions passed is more than 2. Raise exception
            raise ValueError("Only 2 dimensions are allowed.")

        if (self.data_loaded):
            for dms in dimensions:
                if (dms > self.input_data.size):
                    # Invalid dimension passed
                    raise ValueError(f"Dimension value {dms} overflows the size for the given dataset.")
        else:
            raise RuntimeError("Input data not loaded into model for validating the given view dimensions")

        self.visualize_data = True
        self.view_dimensions = dimensions
        self.showAccuracy = showAccuracy
        self.showError = showError

        # Forming mesh for drawing the decision boundaries
        # grid_step_size = 0.01
        # x_min = self.input_data[:, self.view_dimensions[0]].min() - 1
        # x_max = self.input_data[:, self.view_dimensions[0]].max() + 1
        # y_min = self.input_data[:, self.view_dimensions[1]].min() - 1
        # y_max = self.input_data[:, self.view_dimensions[1]].max() + 1

        # self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, grid_step_size), np.arange(y_min, y_max, grid_step_size))

        print ("GLVQ Model: Model visualization set to ON.")


    #
    # 
    ########################################
    def _plot2d(self, chart):
        """
            Plots a 2 dimensional scatter plot into the sublot pointed to be the variable chart.
            Plots the input data and the prototypes into the same chart.
            Also plots the decision boundaries formed by the prototypes.

            Parameters:
                chart: A subplot object where the scatter plot has to be plotted
        """
        if (not(self.visualize_data)):
            raise RuntimeError("Model visualization not initialized.")

        # print ("distance: ", squared_euclidean(self.input_data[:, (self.view_dimensions)],
        #                          np.c_[self.xx.ravel(), self.yy.ravel()]).shape)

        # contour_heights = self.prototype_labels[np.argmin(squared_euclidean(self.input_data[:, (self.view_dimensions)],
        #                          np.c_[self.xx.ravel(), self.yy.ravel()]), axis=1)]
        
        chart.scatter(self.input_data[:, self.view_dimensions[0]], self.input_data[:, self.view_dimensions[1]]
            , c=self.input_data_labels, cmap='viridis')
        chart.scatter(self.prototypes[:, self.view_dimensions[0]], self.prototypes[:, self.view_dimensions[1]]
            , c=self.prototype_labels, marker='D', edgecolor="black")
        # chart.contourf(xx, yy, contour_heights, cmap='viridis', alpha=0.6)


    #
    # 
    ########################################
    def _plotLine(self, chart, values):
        """
            Plots a simple line plot for the provided values

            Parameters:
                chart: A subplot object where the scatter plot has to be plotted
                values: The list of values that has to be plotted
        """
        if (not(self.visualize_data)):
            raise RuntimeError("Model visualization not initialized.")

        chart.plot(np.arange(values.size), values, marker="d")



    #
    # 
    ########################################
    def _mu(self, dplus, dminus):
        """
            Calculates the value for mu(x) = (d_plus - d_minus) / (dplus + d_minus)

            Parameters:
                dplus: A 1D array of d_plus values. Size is equal to that of the data points
                dminus: A 1D array of d_minus values. Size is equal to that of the data points
            
            Returns:
                A 1D array of the result of mu(x)
        """
        return (dplus - dminus) / (dminus + dplus)


    #
    # 
    ########################################
    def _mu_dash_dplus(self, dplus, dminus):
        """
            Returns the first order derivative of the classifier function mu(x) with respect to dplus

            Parameters:
                dplus: A 1D array of d_plus values. Size is equal to that of the data points
                dminus: A 1D array of d_minus values. Size is equal to that of the data points
            
            Returns:
                A 1D array of the result of derivative of mu(x) with respect to dplus
        """
        return (2) * (dminus) / np.square(dplus + dminus)


    #
    # 
    ########################################
    def _mu_dash_dminus(self, dplus, dminus):
        """
            Returns the first order derivative of the classifier function mu(x) with respect to dplus

            Parameters:
                dplus: A 1D array of d_plus values. Size is equal to that of the data points
                dminus: A 1D array of d_minus values. Size is equal to that of the data points
            
            Returns:
                A 1D array of the result of derivative of mu(x) with respect to dplus
        """
        return (-2) * (dplus) / np.square(dplus + dminus)


    #
    # 
    ########################################
    def fit(self):
        """
            Run the model with initialized values. Function optimizes the prototypes minimizing the
                GLVQ classification error
        """
        distances = self.distances
        dplus = self.dplus
        dminus = self.dminus

        # Check if model has data loaded and prototypes initialized
        if self.data_loaded and self.prototypes_init:
            # Model is valid

            # Number of datapoints
            n_x = len(self.input_data)

            # Use the data labels and prototype labels to define a mask
            # Placing it here since, this mask does not change with epochs
            dist_mask = np.equal(np.expand_dims(self.input_data_labels, axis=1), self.prototype_labels)

            fig = plt.figure("GLVQ Model Training!", figsize=(10, 10))

            # Check if visualization is set and initialize plot object
            if (self.visualize_data):
                chartCount = 1

                if (self.showAccuracy):
                    chartCount += 1
                if (self.showError):
                    chartCount += 1
                
                gs = fig.add_gridspec(chartCount, 2)
                # plt.ion()
                # plt.show()

            for i in range(self.epochs):

                if (self.visualize_data):
                    plt.clf()
                
                # Calculate element wise distances
                distances = squared_euclidean(self.input_data, self.prototypes)

                # Use this mask to identify and mark distances to correctly matched prototypes
                # And then find the argument for the closest prototype
                nearest_matching_prototypes = np.argmin(np.where(dist_mask, distances, np.inf), axis=1)
                nearest_mismatched_prototypes = np.argmin(np.where(np.logical_not(dist_mask), distances, np.inf), axis=1)

                # dplus and dminus seperated from the distances matrix
                dplus = distances[np.arange(n_x), nearest_matching_prototypes]
                dminus = distances[np.arange(n_x), nearest_mismatched_prototypes]

                # Initial cost for validation
                initial_cost = np.sum(self._mu(dplus, dminus))

                if (i == 0):
                    self.costs = np.append(self.costs, initial_cost)

                    accuracy = round((np.sum(dplus < dminus) / n_x) * 100, 3)
                    self.accuracies = np.append(self.accuracies, accuracy)

                # Components of the gradient of the cost function _mu(x)
                # x - w split for wplus and wminus respectively
                x_minus_wplus = self.input_data - self.prototypes[nearest_matching_prototypes]
                x_minus_wminus = self.input_data - self.prototypes[nearest_mismatched_prototypes]

                # d _mu(x) / d distance fn : split with respect to dplus and dminus
                change_in_dplus = self._mu_dash_dplus(dplus, dminus) * sigmoid(self._mu(dplus, dminus)) \
                    * (1 - sigmoid(self._mu(dplus, dminus)))
                change_in_dminus = self._mu_dash_dminus(dplus, dminus) * sigmoid(self._mu(dplus, dminus)) \
                    * (1 - sigmoid(self._mu(dplus, dminus)))
                
                # Combining the components to assemble the gradient of the cost function _mu(x)
                dell_mu_wplus = (-2) * x_minus_wplus * np.expand_dims(change_in_dplus, axis=1)
                dell_mu_wminus = (-2) * x_minus_wminus * np.expand_dims(change_in_dminus, axis=1)

                # Generating an update matrix with the calculated gradient
                update_for_prototypes = np.zeros((distances.shape[0], distances.shape[1], self.input_data.shape[1]))
                update_for_prototypes[np.arange(n_x), nearest_matching_prototypes] += dell_mu_wplus
                update_for_prototypes[np.arange(n_x), nearest_mismatched_prototypes] += dell_mu_wminus
                update_for_prototypes = np.sum(update_for_prototypes, axis=0)

                # Update the prototypes
                new_prototypes = self.prototypes - (self.learning_rate) * update_for_prototypes
                self.prototypes = new_prototypes

                # Visualize
                if (self.visualize_data):
                    pos = 0
                    axes1 = fig.add_subplot(gs[pos, :], title="Data plot")
                    self._plot2d(axes1)

                    if (self.showError):
                        pos += 1
                        axes2 = fig.add_subplot(gs[pos, :], title="Error trend")
                        self._plotLine(axes2, self.costs)

                    if (self.showAccuracy):
                        pos += 1
                        axes3 = fig.add_subplot(gs[pos, :], title="Accuracy trend (in %)")
                        self._plotLine(axes3, self.accuracies)

                    plt.draw()
                    plt.pause(0.001)

                # For Validation
                distances = squared_euclidean(self.input_data, new_prototypes)
                nearest_matching_prototypes = np.argmin(np.where(dist_mask, distances, np.inf), axis=1)
                nearest_mismatched_prototypes = np.argmin(np.where(np.logical_not(dist_mask), distances, np.inf), axis=1)
                dplus = distances[np.arange(n_x), nearest_matching_prototypes]
                dminus = distances[np.arange(n_x), nearest_mismatched_prototypes]

                updated_cost = np.sum(self._mu(dplus, dminus))

                # Store calculated costs for the training epoch
                self.costs = np.append(self.costs, updated_cost)

                # Calculate and store accuracy
                accuracy = round((np.sum(dplus < dminus) / n_x) * 100, 3)
                self.accuracies = np.append(self.accuracies, accuracy)

                print ("Epoch: ", i+1, " Cost: ", self.costs[i], " Accuracy: ", self.accuracies[i], "%")
            
            if (self.visualize_data):
                plt.show()
            # plot2d (self.input_data, self.input_data_labels, self.prototypes, self.prototype_labels, "Data plot")
        else:
            # Model is not valid
            raise RuntimeError("Model not initialized properly to run _fit().")


    #
    # 
    ########################################
    def predict(self, predictData):
        """
            Generate a prediction for a provided data point or data matrix

            Parameter:
                predicData: A (n,m) matrix of n datapoints with m features each
            
            Returns:
                A (n,) array of labels for the provided dataset
        """
        if (predictData.shape[1] != self.prototypes.shape[1]):
            raise ValueError("Dimension of the data to be predicted does not match with the model prototypes")

        closest_prototypes = np.argmin(squared_euclidean(predictData, self.prototypes), axis=1)

        return self.prototype_labels[closest_prototypes]
