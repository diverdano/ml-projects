#!usr/bin/python

# === load libraries ===
# key libraries
import numpy as np

# data prep
#from sklearn.cross_validation import train_test_split # deprecated
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# data sets
from sklearn.datasets import load_digits


# models
#from NeuralNetwork import NeuralNetwork

# metrics
from sklearn.metrics import confusion_matrix, classification_report

# plot


# === helper functions ===

def logistic(x):            return 1/(1 + np.exp(-x))
def logistic_derivative(x): return logistic(x)*(1-logistic(x))
def tanh(x):                return np.tanh(x)
def tanh_deriv(x):          return 1.0 - np.tanh(x)**2

# === objects ===

class Perceptron(object):
    """This class models an artificial neuron with step activation function."""
    def __init__(self, weights = np.array([1]), threshold = 0):
        """Initialize weights and threshold based on input arguments. Note that no type-checking is being performed here for simplicity."""
        self.weights = weights#.astype(float)
        self.threshold = threshold
        self.last_input = 0 # strength of last input
        self.delta      = 0 # error signal
    def __repr__(self):
        return str({'weights':self.weights, 'threshold':self.threshold})
    def activate(self,inputs):
        """Takes in @param inputs, a list of numbers equal to length of weights. @return the output of a threshold perceptron with given inputs based on perceptron weights and threshold.""" 
        strength = np.dot(inputs,self.weights)
        self.last_input = strength
        return int(strength > self.threshold)       # perceptron activates with a boolean function
    def update(self, values, train, eta=.1):
        """Takes in a 2D array @param values consisting of a LIST of inputs and a 1D array @param train, consisting of a corresponding list of expected outputs.
        Updates internal weights according to the perceptron training rule using these values and an optional learning rate, @param eta."""
        for i, iteration in enumerate(train):       # remember inputs is a list, output is scalar
            prediction = self.activate(values[i])   # obtain prediction
            error = train[i] - prediction
            for j, weight in enumerate(self.weights):
                self.weights[j] += eta * error * values[i][j]

class Sigmoid(Perceptron):
    """This class models an artificial neuron with sigmoid activation and update functions."""
    def activate(self, values):
        """Takes in @param values, a list of numbers equal to length of weights. @return the output of a sigmoid unit with given inputs based on unit weights."""
        strength = np.dot(values, self.weights)
        self.last_input = strength
        return logistic(strength)                 # sigmoid activates with a continuous function
    def update(self, values, train, eta=.1):
        """Takes in a 2D array @param values consisting of a LIST of inputs and a 1D array @param train, consisting of a corresponding list of expected outputs.
        Updates internal weights according to gradient descent using these values and an optional learning rate, @param eta."""
        for i, iteration in enumerate(train):   # remember inputs is a list, output is scalar
            prediction = self.activate(values[i])        # obtain prediction
            error = train[i] - prediction
            for j, weight in enumerate(self.weights):
                self.weights[j] += eta * error * values[i][j] * prediction * (1 - prediction)         # suggestion of forum poster, dropping the minus sign

class NeuralNetwork(object):
    def __init__(self, layers, activation='tanh'):
        """ layers:     A list containing the number of units in each layer. Should be at least two values
            activation: The activation function to be used. Can be 'logistic' or 'tanh'."""
        if activation == 'logistic':
            self.activation         = logistic
            self.activation_deriv   = logistic_derivative
        elif activation == 'tanh':
            self.activation         = tanh
            self.activation_deriv   = tanh_deriv
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)    
#     def logistic(x):            return 1/(1 + np.exp(-x))
#     def logistic_derivative(x): return logistic(x)*(1-logistic(x))
#     def tanh(x):                return np.tanh(x)
#     def tanh_deriv(x):          return 1.0 - np.tanh(x)**2
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X               = np.atleast_2d(X)
        temp            = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1]   = X  # adding the bias unit to the input layer
        X               = temp
        y               = np.array(y)
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            for l in range(len(self.weights)):
#                 print(a[l])
#                 print(self.weights[l])
#                 print(np.dot(a[l], self.weights[l]))
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

# === data ===

# Part 2: Define a procedure to compute the output of the network, given inputs
def EvalNetwork(inputValues, Network):
    """Takes in @param inputValues, a list of input values, and @param Network that specifies a perceptron network. @return the output of the Network for the given set of inputs."""
    inputValues = np.array(list(inputValues))
    for layer in Network:
        results = [node.activate(inputValues) for node in layer]
        inputValues = results
    return results[0]

# === test functions ===
def test_sigmoid():
    """A few tests to make sure that the perceptron class performs as expected. Nothing should show up in the output if all the assertions pass."""
    def sum_almost_equal(array1, array2, tol = 1e-5):
        return sum(abs(array1 - array2)) < tol
    u1 = Sigmoid(weights=[3,-2,1])
    assert abs(u1.activate(np.array([1,2,3])) - 0.880797) < 1e-5
    u1.update(np.array([[1,2,3]]),np.array([0]))
    print(u1.weights)
    assert sum_almost_equal(u1.weights, np.array([2.990752, -2.018496, 0.972257]))
    u2 = Sigmoid(weights=[0,3,-1])
    u2.update(np.array([[-3,-1,2],[2,1,2]]),np.array([1,0]))
    print(u2.weights)
    assert sum_almost_equal(u2.weights, np.array([-0.030739, 2.984961, -1.027437]))
def test_nn():
    nn = NeuralNetwork([2,2,1], 'tanh')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
        print(i,nn.predict(i))

def test_nn_digits():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X -= X.min() # normalize the values to bring them into the range 0-1
    X /= X.max()

    nn = NeuralNetwork([64,100,10],'tanh')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)

    nn.fit(X_train,labels_train,epochs=30000)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i] )
        predictions.append(np.argmax(o))
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

def test_eval():
    """A few tests to make sure that the perceptron class performs as expected."""
    Network = [[ Perceptron(np.array([1,1]),0), Perceptron(np.array([1,1]),1) ], [ Perceptron(np.array([1,-2]),0) ]] # [input perceptrons OR & AND],[output perceptron OR -2xAND]
    print("0 XOR 0 = 0?:", EvalNetwork(np.array([0,0]), Network))
    print("0 XOR 1 = 1?:", EvalNetwork(np.array([0,1]), Network))
    print("1 XOR 0 = 1?:", EvalNetwork(np.array([1,0]), Network))
    print("1 XOR 1 = 0?:", EvalNetwork(np.array([1,1]), Network))

def test_update():
    """A few tests to make sure that the perceptron class performs as expected. Nothing should show up in the output if all the assertions pass."""
    def sum_almost_equal(array1, array2, tol = 1e-6):
        return sum(abs(array1 - array2)) < tol

    p1 = Perceptron(np.array([1,1,1]),0)
    p1.update(np.array([[2,0,-3]]), np.array([1]))
    print(p1.weights)
    assert sum_almost_equal(p1.weights, np.array([1.2, 1, 0.7]))

    p2 = Perceptron(np.array([1,2,3]),0)
    p2.update(np.array([[3,2,1],[4,0,-1]]),np.array([0,0]))
    print(p2.weights)
    assert sum_almost_equal(p2.weights, np.array([0.7, 1.8, 2.9]))

    p3 = Perceptron(np.array([3,0,2]),0)
    p3.update(np.array([[2,-2,4],[-1,-3,2],[0,2,1]]),np.array([0,1,0]))
    print(p3.weights)
    assert sum_almost_equal(p3.weights, np.array([2.7, -0.3, 1.7]))

def test_activation():
    """A few tests to make sure that the perceptron class performs as expected. Nothing should show up in the output if all the assertions pass."""
    p1 = Perceptron(np.array([1, 2]), 0.)
    assert p1.activate(np.array([ 1,-1])) == 0 # < threshold --> 0
    assert p1.activate(np.array([-1, 1])) == 1 # > threshold --> 1
    assert p1.activate(np.array([ 2,-1])) == 0 # on threshold --> 0

# === other functions ===

def activate(strength):
    # Try out different functions here. Input strength will be a number, with
    # another number as output.
    return np.power(strength,2)

def activation_derivative(activate, strength):
    #numerically approximate
    return (activate(strength+1e-5)-activate(strength-1e-5))/(2e-5)
