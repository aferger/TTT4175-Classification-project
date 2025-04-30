import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import scipy.special as ss

#######################################
###   IRIS DATASET CLASSIFICATION   ###
# Removing one feature at a time

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']

num_classes = 3
num_features = 4
N_train = 30

setosa  = data[:50] # class 1 [1,0,0]
versicolor = data[50:100] # class 2 [0,1,0]
virginica = data[100:] # class 3 [0,0,1]


#-----------------------------------------------------------------------

## FUNCTION DEFINITIONS ##

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#recomended random function for sigmoid
#creates a random matrix with the same shape as W
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(1 / fan_in)  # Xavier-uniform range
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def WgradMSE(z):
    m = X_train.shape[0]
    g = [(sigmoid(z) - t_train) * sigmoid(z) * (1 - sigmoid(z))]
    g = np.array(g).reshape(m, num_classes)
    return np.dot(g.T, X_train) / m # divide by m to avoid unstable learning

def w_oGradMSE(z):
    m = X_train.shape[0]
    g = [(sigmoid(z) - t_train) * sigmoid(z) * (1 - sigmoid(z))]
    g = np.array(g).reshape(m, num_classes)
    return np.sum(g, axis=0, keepdims=True) / m # divide by m to avoid unstable learning

## GLOBAL VARIABLES ##
# Gradient descent
alpha = 0.01
iterations = 10000

#-------------------------------------------------------------------

# Removing the feature with the most overlap
for i in range(4):
    num_features = 4
    X = np.vstack([setosa, versicolor, virginica])
    print("X: ", X[:8])
    
    X = np.delete(X, i, axis=1) # remove one and one feature from the data set
    print("X without feature", i, ": ", X[:8])
    X_train = np.vstack([X[:N_train], X[50:50+N_train], X[100:100+N_train]])
    X_test = np.vstack([X[N_train:50], X[50+N_train:100], X[100+N_train:]])
    t_train = np.vstack([np.tile([1,0,0], (N_train, 1)), np.tile([0,1,0], (N_train, 1)), np.tile([0,0,1], (N_train, 1))])

    num_features -= 1 # remove one feature
    W = xavier_init(num_classes, num_features)
    w_o = np.zeros((1, num_classes))  # Init bias

    ## TRAINING ##
    for iteration in range(iterations):
        g = np.dot(X_train, W.T) + w_o
        g_k = sigmoid(g)

        W -= alpha * WgradMSE(g)
        w_o -= alpha * w_oGradMSE(g)

    ## CONFUSION MATRIX ##

    # initalization
    confusion_matrix_train = np.zeros((num_classes, num_classes), dtype=int)
    confusion_matrix_test = np.zeros((num_classes, num_classes), dtype=int)

    # adding to confusion matrices
    for i in range(len(X_train)):
        g = sigmoid(np.dot(X_train[i], W.T) + w_o)

        true_train_class = np.argmax(t_train[i])
        predicted_class = np.argmax(g)

        confusion_matrix_train[true_train_class][predicted_class] += 1

    for i in range(len(X_test)):
        g = sigmoid(np.dot(X_test[i], W.T) + w_o)
        
        true_test_class = i//20
        predicted_class = np.argmax(g)

        confusion_matrix_test[true_test_class][predicted_class] += 1

    print("Confution matrix on traing set with", num_features, "features:")
    print(confusion_matrix_train)
    print("Confution matrix on test set with", num_features, "features:")
    print(confusion_matrix_test)

    ## ERROR RATE ##
    wrongly_classified_train = confusion_matrix_train.sum(axis=1) - np.diag(confusion_matrix_train)
    total_classified_train = confusion_matrix_train.sum(axis=1)

    wrongly_classified_test = confusion_matrix_test.sum(axis=1) - np.diag(confusion_matrix_test)
    total_classified_test = confusion_matrix_test.sum(axis=1)

    ERR_T = wrongly_classified_train.sum() / total_classified_train.sum() * 100
    ERR_setosa = wrongly_classified_train[0] / total_classified_train[0] * 100
    ERR_versicolor = wrongly_classified_train[1] / total_classified_train[1] * 100
    ERR_virginica = wrongly_classified_train[2] / total_classified_train[2] * 100

    print("ERROR RATE TRAIN SET: ")
    print("Total error rate: ", ERR_T)
    print("Error rate setosa: ", ERR_setosa)
    print("Error rate versicolor: ", ERR_versicolor)
    print("Error rate virginica: ", ERR_virginica)

    ERR_T = wrongly_classified_test.sum() / total_classified_test.sum() * 100
    ERR_setosa = wrongly_classified_test[0] / total_classified_test[0] * 100
    ERR_versicolor = wrongly_classified_test[1] / total_classified_test[1] * 100
    ERR_virginica = wrongly_classified_test[2] / total_classified_test[2] * 100

    print("ERROR RATE TEST SET: ")
    print("Total error rate: ", ERR_T)
    print("Error rate setosa: ", ERR_setosa)
    print("Error rate versicolor: ", ERR_versicolor)
    print("Error rate virginica: ", ERR_virginica)
    print("-----")