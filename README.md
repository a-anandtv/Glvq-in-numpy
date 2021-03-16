# Generalized Learning Vector Quantization

An implementation for GLVQ in Numpy. The class Glvq can be used to generate a model for GLVQ initialised with a built in default values for its parameters or pass these parameters to the model while initializing the object. Data to train the model on will have to be loaded using the *model.load_data()* function. Prototypes will then have to be initialized using the *model.initialize_prototypes()* function. Then the model can be trained using the *model.fit()* function and the optimized prototypes can be viewed by accessing *model.prototypes* and their respective classes by accessing *model.prototype_labels*.

## A simple usage for the class Glvq

```python
.
.
.
from Glvq import Glvq

# Model initialized with all default values for parameters
model = Glvq()

# Load model with input data and data labels
model.load_data(input_data, input_data_labels)

# Initialize parameters with default settings (prototypes initialized to mean of each class)
model.initialize_prototypes()

# Other forms
# model.initialize_prototypes(initialize_by="random-input") 	# prototypes are initialized to a random input from the class
# model.initialize_prototypes(initialize_by="random")		# prototypes are initialized to a random data point in the data space
# model.initialize_prototypes(initialize_by="class-mean")	# prototypes are initialized to the class mean (set by default)
# model.initialize_prototypes(initialize_by="initialized", pass_w=<prototypes>, pass_w_labels=<prototype labels>)
								# prototypes are initialized with a pre decided set "prototypes" with labels "prototype_labels"

# Model does not show any visualization while fitting.
# Enable visualization to show a 2D representation for the data space and the trend for Accuracy and Error while fitting

# Visualization is set to On and displays a 2D representation of the data space. Default dimensions used are (0, 1)
model.setVisualizeOn()

# Other forms
# model.setVisualizeOn(dimensions=(0, 2))			# define dimensions to use to visualize data
# model.setVisualizeOn(dimensions=(0, 2), showAccuracy=True)	# define dimensions and activate the accuracy trend graph
# model.setVisualizeOn(dimensions=(0, 2), showError=True)	# define dimensions and activate the error trend graph

# Fit the model to the provided data inputs
model.fit()

.
.
.
# Predict using the fit model
predicted_class = model.predict(predict_data)

```



## A brief description for GLVQ

GLVQ is a form of vector quantization which uses the nearest prototypes with maching labels and mismatched labels to optimize the prototypes. The model has a generalized cost function introduced that is continuous and mathematically differentiable. This will help with optimizing the model by taking a derivative of the cost function and use it to optimize the position for the prototypes. These optimized prototypes can then be used as quantised representation for the data points and use these for future data classification.

The cost function for the model,

![Cost function GLVQ](Resources/costfn.png?raw=true)

Here **d** can be any form of a distance measure and **Squared Euclidean distance** is used here in this implementation. **X** is the set of data points and **W** is the prototype set. The function **f** is implemented here as the **Sigmoid Function**. 

![Sigmoid function](Resources/sigmoidfn.png?raw=true)

The prototypes are then optimized using the following steps,

![Derivative of the cost function](Resources/derivativecostfn.png?raw=true)

Since we have 2 set of prototype, W^+^ , prototypes with matching labels and W^-^, prototypes with mismatched labels, the derivatives will have to be taken separately.

The derivative turns out to be:

![Gradient of the cost function with respect to W](Resources/gradientprototypes.png?raw=true)

![Xi function](Resources/xi.png?raw=true)

And the prototypes are updated by,

![Update to prototypes](Resources/prototypeupdate.png?raw=true)

Where **$\alpha$** is the learning rate.
