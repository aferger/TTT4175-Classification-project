import numpy as np
from sklearn.datasets import load_iris
import scipy.special as ss

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']
label = iris['target']


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
#print("X_train: ", X_train)

# Test sets
setosa_test = setosa[N_train:]
versicolor_test = versicolor[N_train:]
virginica_test = virginica[N_train:]

X_test = np.vstack([setosa_test, versicolor_test, virginica_test])

# Labels
t_train = []
for i in range(0, len(X_train)):
    if label[i] == 0:
        t_train.append([1,0,0])
    elif label[i] == 1:
        t_train.append([0,1,0])
    elif label[i] == 2:
        t_train.append([0,0,1])
#print("t_train: ", t_train)

# convert values to probabilites
def softmax(g):
    exp_g = np.exp(g - np.max(g, axis=1, keepdims=True)) 
    return exp_g / np.sum(exp_g, axis=1, keepdims=True)

#print("softmax(g) = ", softmax(g))

def WgradMSE(g):
    # divide by m to avoid unstable learning
    m = X_train.shape[0]
    dg = softmax(g) - t_train
    return np.dot(X_train.T, dg) / m

#print("gradMSE :", WgradMSE(g))
#print("len(gradMSE): ", len(WgradMSE(g)))

def w_ioGradMSE(g):
    m = X_train.shape[0]
    dg = softmax(g) - t_train
    return np.sum(dg, axis=0, keepdims=True) / m
#print("bdragMSE: ", bgradMSE(g))

#recomended random function for sigmoid
#creates a random matrix with the same shape as W
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(1 / fan_in)  # Xavier-uniform range
    return np.random.uniform(-limit, limit, (fan_out, fan_in))

# W = C x D = 3klasser x 4målinger = weight
W = xavier_init(num_classes, num_features)
print("W init: ", W)

# Gradient descent
alpha = 0.1
iterations = 10000

w_io = np.zeros((1, num_classes))  # Init bias
w_io = np.random.randn(1, num_classes) * 0.01 #Better training with normal distribution


for iteration in range(iterations):
    g = np.dot(W.T, X_train.T).T + w_io

    W -= alpha * WgradMSE(g)
    w_io -= alpha * w_ioGradMSE(g)

print("W trained: ", W)

#----------------------------------------------------------
# Confusion Matrix
w1 = W[:,0]
w2 = W[:,1]
w3 = W[:,2]

confusion_matrix = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]

confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)


for i in range(0, len(X_test)):
    g = [np.dot(w1, X_test[i]), np.dot(w2,X_test[i]), np.dot(w3, X_test[i])]
    #print("g: ", g)
    gj = np.max(g)
    #print("gj: ", gj)
    if i < 20:
        if gj == g[0]:
            confusion_matrix[0][0] += 1
        if gj == g[1]:
            confusion_matrix[0][1] += 1
        if gj == g[2]:
            confusion_matrix[0][2] += 1
    if 20 <= i < 40:
        if gj == g[0]:
            confusion_matrix[1][0] += 1
        if gj == g[1]:
            confusion_matrix[1][1] += 1
        if gj == g[2]:
            confusion_matrix[1][2] += 1
    if i >= 40:
        if gj == g[0]:
            confusion_matrix[2][0] += 1
        if gj == g[1]:
            confusion_matrix[2][1] += 1
        if gj == g[2]:
            confusion_matrix[2][2] += 1

print(confusion_matrix)