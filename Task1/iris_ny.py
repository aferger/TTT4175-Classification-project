import numpy as np
from sklearn.datasets import load_iris
import scipy.special as ss

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']
labels = iris['target']


num_classes = 3
num_features = 4

setosa  = data[:50] # class 1 [1,0,0]
versicolor = data[50:100] # class 2 [0,1,0]
virginica = data[100:] # class 3 [0,0,1]

# Training sets
N_train = 30

setosa_train = setosa[:N_train]
versicolor_train = versicolor[:N_train]
virginica_train = virginica[:N_train]
X_train = np.vstack([setosa_train, versicolor_train, virginica_train])
print("X_train: ", X_train)
print("Len: ", len(X_train))

# Test sets
setosa_test = setosa[N_train:]
versicolor_test = versicolor[N_train:]
virginica_test = virginica[N_train:]

t_setosa = [1, 0, 0]
t_versicolor = [0, 1, 0]
t_virginica = [0, 0, 1]

#recomended random function for sigmoid
#creates a random matrix with the same shape as W
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(1 / fan_in)  # Xavier-uniform range
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

# W = C x D = 3klasser x 4målinger = weight
W = xavier_init(num_classes, num_features)
#print("W init: ", W)

w_io = np.zeros((1, num_classes))  # Init bias
#print("w_io: ", w_io)

# convert values to probabilites
def softmax(g):
    exp_g = np.exp(g - np.max(g, axis=1, keepdims=True)) 
    return exp_g / np.sum(exp_g, axis=1, keepdims=True)

# Gradient descent
alpha = 0.1
epochs = 1000