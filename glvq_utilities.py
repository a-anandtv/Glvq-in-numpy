# 
# A utility class to support the Glvq class
#
#
# By: Akash Anand
########################################

import numpy as np


#
# 
########################################
def distances (xData, wPrototypes, metric = "squared-euclidean"):
    """
        Calculates the distances between two vectors, xData and wPrototype. Returns array of distances for 
            each data point with each of the prototypes

        Paramters:
            xData: Input data points. (n,m) matrix. n is number of data points. m is the number of features.
            wPrototypes: Prototypes to which distance has to be calculated. (k,m) matrix. k is number of 
                prototypes. m is the number of features (similar to x).
            metric: Distance calculation metric. {"squared-euclidean", "euclidean", "manhattan"}
        
        Returns:
            Distance matrix. (n,k) matrix. Distance between each n data points to k prototypes.
        
        Raises:
    """

    # Checking if dimensions match
    if (xData.shape[1] != wPrototypes.shape[1]):
        # Invalid dimensions exception
        raise ValueError("Invalid inputs. Shapes for the passed arguments do not match.")
    
    # Checking for metric
    if (metric == "squared-euclidean"):
        # Caclulate Euclidean distance
        # Using Numpy
        expanded_data = np.expand_dims (xData, axis=1)
        distances = np.sum (np.power (expanded_data - wPrototypes, 2), axis=2)
        return distances.astype(float)
    else:
        # Code for other distances here
        return False


#
# 
########################################
def squared_euclidean (xData, wPrototypes):
    """
        Calculate the squared euclidean distance between xData and wPrototypes
            = (xData - wPrototypes) ^ 2
        
        Parameters:
            xData: Input data points. (n,m) matrix. n is number of data points. m is the number of features.
            wPrototypes: Prototypes to which distance has to be calculated. (k,m) matrix. k is number of 
                prototypes. m is the number of features (similar to x).
        
        Returns:
            Distance matrix. (n,k) matrix. Distance between each n data points to k prototypes.
    """

    # Checking if dimensions match
    if (xData.shape[1] != wPrototypes.shape[1]):
        # Invalid dimensions exception
        raise ValueError("Invalid inputs. Shapes for the passed arguments do not match.")

    # Caclulate Euclidean distance
    # Using Numpy
    expanded_data = np.expand_dims (xData, axis=1)
    distances = np.sum (np.power (expanded_data - wPrototypes, 2), axis=2)

    return distances.astype(float)


#
# 
########################################
def sigmoid (xData, theta = 1):
    """
        Calculates the sigmoid of the provided data vector.
        Sigmoid (x) = 1 / (1 + e^(-theta * x))

        Parameter:
            xData: Input data vector. (n,m) matrix. n is number of data points and m is the number of features
            theta: Theta parameter in the calculation of sigmoid. Default set to 1
        
        Returns:
            Resultant (n,m) matrix
        
        Raises:
    """

    return (1 / (1 + (np.exp(-1 * theta * xData))))



#
# 
########################################
def plot2d (plotObject, figure, xData, xLabels, wData, wLabels, dimensions=(0, 1)):
    """
        Plot a 2D slice of an N dimensional dataset. Sliced using the dimensions proivided else 1st and 2nd 
            dimensions chosen by default
        
        Parameters:
            plotObject: Matplotlib.pyplot object to be used to plot in
            figure: Name or Id for the figure.
            xData: Data to be plotted. (n, m) matrix. n is number of data points and m is the number of features.
            xLabels: Labels for the dataset.
            wData: Prototype data to be plotted. (k, m) matrix.
            wLabels: Labels for the prototype.
            dimensions: A 2D array of the dimesnions to be used to plot. Defaults to dimensions 0 and 1.
    """

    if (len(dimensions) != 2):
        # Dimensions passed is more than 2. Raise exception
        raise ValueError("Only 2 dimensions are allowed.")

    for dms in dimensions:
        if (dms > xData.size):
            # Invalid dimension passed
            raise ValueError(f"Dimension value {dms} overflows the size for the given dataset.")
    
    fig = plotObject.figure(figure)
    chart = fig.add_subplot(1, 2, 1)

    chart.scatter(xData[:, dimensions[0]], xData[:, dimensions[1]], c=xLabels, cmap='viridis')
    chart.scatter(wData[:, dimensions[0]], wData[:, dimensions[1]], c=wLabels, marker='D')
    
    plotObject.show()





