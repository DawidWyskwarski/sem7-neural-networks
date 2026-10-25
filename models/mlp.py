import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from models.activation_functions import ActivationFunction, Sigmoid

class Neuron:

    def __init__(
        self,
        activation_function: ActivationFunction,
        input_dim: int,
        weight_deviation: float,
        learning_rate: float
    ):
        self.activation_function = activation_function

        self.learning_rate = learning_rate

        self.weights_ = np.random.normal(0, weight_deviation, input_dim)
        self.bias = 0

        self.cache_input = None
        self.cache_transformed = None

    def forward(self, input):
        self.cache_input = input
        self.cache_transformed = (input @ self.weights_) + self.bias

        return self.activation_function.apply( self.cache_transformed )

    def backward(self, derivative):
        base_derivative = derivative * self.activation_function.derivative(self.cache_transformed) 
        
        new_derivative = base_derivative * self.weights_ 

        dw = base_derivative * self.cache_input
        db = base_derivative

        self.weights_ -= self.learning_rate * dw.T
        self.bias -= self.learning_rate * db

        return new_derivative

class Layer:

    def __init__(
        self,
        size: int,
        input_dim: int,
        weight_deviation: float,
        learning_rate: float,
        activation_function: ActivationFunction,
    ):
        self.neurons = [
            Neuron(
                activation_function=activation_function,
                input_dim=input_dim,
                weight_deviation=weight_deviation,
                learning_rate=learning_rate,
            )
            for _ in range(size)
        ]


    def forward(self, X):
        return np.array([neuron.forward(X) for neuron in self.neurons])


    def backward(self, derivative):
        grad_input = np.zeros_like(self.neurons[0].weights_)
        
        for neuron, d in zip(self.neurons, derivative):
            grad_input += neuron.backward(d)
        
        return grad_input


class MultiLayerPerceptron(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        hidden_layer_sizes: list[int],
        activation_function: ActivationFunction,
        loss_derivative,
        weight_deviation: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 1000
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        self.loss_derivative = loss_derivative
        self.weight_deviation = weight_deviation
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.layers: list[Layer] = []


    def forward(self, input):
        if self.layers == []:
            raise RuntimeError("Layers not initialized")
            
        tmp = input

        for layer in self.layers:
            tmp = layer.forward(tmp)

        return tmp


    def backward(self, y, predicted):
        if self.layers == []:
            raise RuntimeError("Layers not initialized")
        
        gradient = self.loss_derivative(y, predicted)

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
    
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
    
        self._initialize_layers(n_features)
    
        for _ in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
    
            for i in indices:
                y_predicted = self.forward(X[i])
                self.backward(y[i], y_predicted)
    
        return self
    
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([int(self.forward(x)[0] >= 0.5) for x in X])


    def _initialize_layers(self, input_size):
        self.layers = []
        
        previous_size = input_size
        
        for layer_size in self.hidden_layer_sizes:
            self.layers.append(
                Layer(
                    size=layer_size,
                    input_dim=previous_size,
                    weight_deviation=self.weight_deviation,
                    learning_rate=self.learning_rate,
                    activation_function=self.activation_function
                )
            )

            previous_size = layer_size

        self.layers.append(
            Layer(
                size=1,
                input_dim=previous_size,
                weight_deviation=self.weight_deviation,
                learning_rate=self.learning_rate,
                activation_function=Sigmoid()
            )
        )
