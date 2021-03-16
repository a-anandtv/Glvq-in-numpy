# Generalized Learning Vector Quantization

An implementation for GLVQ in Numpy. The class Glvq can be used to generate a model for GLVQ initialised with a built in default values for its parameters or pass these parameters to the model while initializing the object. Data to train the model on will have to be loaded using the *model.load_data()* function. Prototypes will then have to be initialized using the *model.initialize_prototypes()* function. Then the model can be trained using the *model.fit()* function and the optimized prototypes can be viewed by accessing *model.prototypes* and their respective classes by accessing *model.prototype_labels*.

## A simple usage for the class Glvq

```python
from Glvq import Glvq

# Model initialized with all default values for parameters
model = Glvq()
model.load_data(input_data, input_data_labels)
model.initialize_prototypes()
model.fit()

```



## A brief description for GLVQ

GLVQ is a form of vector quantization which uses the nearest prototypes with maching labels and mismatched labels to optimize the prototypes. The model has a generalized cost function introduced that is continuous and mathematically differentiable. This will help with optimizing the model by taking a derivative of the cost function and use it to optimize the position for the prototypes. These optimized prototypes can then be used as quantised representation for the data points and use these for future data classification.

The cost function for the model,
$$
\begin{equation}
	Cost = \mu(x,w) = f\bigg( \frac{d^{+} - d^{-}}{d^{+} + d^{-}}\bigg)
\end{equation}
$$
Here **d** can be any form of a distance measure and **Squared Euclidean distance** is used here in this implementation. **X** is the set of data points and **W** is the prototype set. The function **f** is implemented here as the **Sigmoid Function**. 
$$
\begin{equation}
	Sigmoid: f(x) = \frac{1}{(1 + e^{-\theta.x})}
\end{equation}
$$
The prototypes are then optimized using the following steps,
$$
\begin{equation}
	\frac{d(Cost fn.)}{dW^{\pm}}
\end{equation}
$$
Since we have 2 set of prototype, W^+^ , prototypes with matching labels and W^-^, prototypes with mismatched labels, the derivatives will have to be taken separately.

The derivative turns out to be:
$$
\begin{equation}
	\triangle W: \frac{d(Cost fn.)}{dW^{\pm}} = (-2)(x - w)(\frac{\xi(\mu(x))}{d(d^{\pm})})(sigmoid(x))(1 + sigmoid(x))
\end{equation}
$$

$$
\begin{equation}
  \frac{\xi(\mu(x))}{d(d^{\pm})} = (\mp 2)\bigg(\frac{d^{\mp}}{(d^{+}+d^{-})^{2}}\bigg)
\end{equation}
$$

And the prototypes are updated by,
$$
\begin{equation}
	W_{new} = W_{old} - (\alpha).(\triangle W)
\end{equation}
$$
Where **$\alpha$** is the learning rate.