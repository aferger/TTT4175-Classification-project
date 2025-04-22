import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import scipy.special as ss


## VARIABLE DEFINITIONS ##

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']

num_classes = 3
num_features = 4
N_first = 30

setosa  = data[:50] # class 1 [1,0,0]
versicolor = data[50:100] # class 2 [0,1,0]
virginica = data[100:] # class 3 [0,0,1]

# training with the first 30 samples of each class
X = np.vstack([setosa, versicolor, virginica])
X_train = np.vstack([X[:N_first], X[50:50+N_first], X[100:100+N_first]])
X_test = np.vstack([X[N_first:50], X[50+N_first:100], X[100+N_first:]])
t_train = np.vstack([np.tile([1,0,0], (N_first, 1)), np.tile([0,1,0], (N_first, 1)), np.tile([0,0,1], (N_first, 1))])

# lists for separated features
setosa_plengths = []
setosa_pwidths = []
setosa_slengths = []
setosa_swidths = []
versicolor_plengths = []
versicolor_pwidths = []
versicolor_slengths = []
versicolor_swidths = []
virginica_plengths = []
virginica_pwidths = []
virginica_slengths = []
virginica_swidths = []
for i in range(0, len(setosa)):
    setosa_plengths.append(setosa[i][0])
    setosa_pwidths.append(setosa[i][1])
    setosa_slengths.append(setosa[i][2])
    setosa_swidths.append(setosa[i][3])
for i in range(0, len(versicolor)):
    versicolor_plengths.append(versicolor[i][0])
    versicolor_pwidths.append(versicolor[i][1])
    versicolor_slengths.append(versicolor[i][2])
    versicolor_swidths.append(versicolor[i][3])
for i in range(0, len(virginica)):
    virginica_plengths.append(virginica[i][0])
    virginica_pwidths.append(virginica[i][1])
    virginica_slengths.append(virginica[i][2])
    virginica_swidths.append(virginica[i][3])

## --------------------------------------------------------------------------------- ##

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
    g = sigmoid(z) - t_train
    return np.dot(g.T, X_train) / m # divide by m to avoid unstable learning

def w_oGradMSE(z):
    m = X_train.shape[0]
    g = sigmoid(z) - t_train
    return np.sum(g, axis=0, keepdims=True) / m # divide by m to avoid unstable learning

def confusionMatrix():
    # initalization
    confusion_matrix_train = np.zeros((num_classes, num_classes), dtype=int)
    confusion_matrix_test = np.zeros((num_classes, num_classes), dtype=int)

    # adding to confusion matrixes
    for i in range(len(X_train)):
        z = np.dot(X_train[i], W.T) + w_o
        g = sigmoid(z)

        predicted_class = np.argmax(g)
        true_train_class = np.argmax(t_train[i])

        confusion_matrix_train[true_train_class][predicted_class] += 1

    for i in range(len(X_test)):
        z = np.dot(X_test[i], W.T) + w_o
        g = sigmoid(z)
        
        predicted_class = np.argmax(g)
        true_test_class = i//20

        confusion_matrix_test[true_test_class][predicted_class] += 1

    print("Confution matrix on training set:")
    print(confusion_matrix_train)
    print("Confution matrix on test set:")
    print(confusion_matrix_test)

    return confusion_matrix_train, confusion_matrix_test

def errorRateTrain(confusion_matrix_train):
    # initalization
    wrongly_classified_train = confusion_matrix_train.sum(axis=1) - np.diag(confusion_matrix_train)
    total_classified_train = confusion_matrix_train.sum(axis=1)

    # calculation
    ERR_T = wrongly_classified_train.sum() / total_classified_train.sum()
    ERR_setosa = wrongly_classified_train[0] / total_classified_train[0]
    ERR_versicolor = wrongly_classified_train[1] / total_classified_train[1]
    ERR_virginica = wrongly_classified_train[2] / total_classified_train[2]

    print("ERROR RATE TRAIN SET: ")
    print("Total error rate: ", ERR_T)
    print("Error rate setosa: ", ERR_setosa)
    print("Error rate versicolor: ", ERR_versicolor)
    print("Error rate virginica: ", ERR_virginica)

def errorRateTest(confusion_matrix_test):
    wrongly_classified_test = confusion_matrix_test.sum(axis=1) - np.diag(confusion_matrix_test)
    total_classified_test = confusion_matrix_test.sum(axis=1)

    ERR_T = wrongly_classified_test.sum() / total_classified_test.sum()
    ERR_setosa = wrongly_classified_test[0] / total_classified_test[0]
    ERR_versicolor = wrongly_classified_test[1] / total_classified_test[1]
    ERR_virginica = wrongly_classified_test[2] / total_classified_test[2]

    print("-----")
    print("ERROR RATE TEST SET: ")
    print("Total error rate: ", ERR_T)
    print("Error rate setosa: ", ERR_setosa)
    print("Error rate versicolor: ", ERR_versicolor)
    print("Error rate virginica: ", ERR_virginica)

## --------------------------------------------------------------------------------- ##

## INITIALIZATION AND TRAINING ##

# W = C x D
W = xavier_init(num_classes, num_features)
w_o = np.zeros((1, num_classes))  # Init bias

# Gradient descent
alpha = 0.1
iterations = 100_000

print("W init: ", W)
for iteration in range(iterations):
    z = np.dot(X_train, W.T) + w_o
    W -= alpha * WgradMSE(z)
    w_o -= alpha * w_oGradMSE(z)
print("W trained: ", W)

## --------------------------------------------------------------------------------- ##

## HISTOGRAM PLOT ##
#                                             SKAL DETTE KUN VÆRE TRAINING SET?
fig, ax = plt.subplots(2, 2)
ax[0][0].hist(setosa_slengths, bins=10, color='r', alpha=0.5, label='Setosa sepal length')
ax[0][1].hist(setosa_swidths, bins=10, color='r', alpha=0.5, label='Setosa sepal width')
ax[1][0].hist(setosa_plengths, bins=10, color='r', alpha=0.5, label='Setosa petal length')
ax[1][1].hist(setosa_pwidths, bins=10, color='r', alpha=0.5, label='Setosa petal width')

ax[0][0].hist(versicolor_slengths, bins=10, color='g', alpha=0.5, label='Versicolor sepal length')
ax[0][1].hist(versicolor_swidths, bins=10, color='g', alpha=0.5, label='Versicolor sepal width')
ax[1][0].hist(versicolor_plengths, bins=10, color='g', alpha=0.5, label='Versicolor petal length')
ax[1][1].hist(versicolor_pwidths, bins=10, color='g', alpha=0.5, label='Versicolor petal width')

ax[0][0].hist(virginica_slengths, bins=10, color='b', alpha=0.5, label='Virginica sepal length')
ax[0][1].hist(virginica_swidths, bins=10, color='b', alpha=0.5, label='Virginica sepal width')
ax[1][0].hist(virginica_plengths, bins=10, color='b', alpha=0.5, label='Virginica petal length')
ax[1][1].hist(virginica_pwidths, bins=10, color='b', alpha=0.5, label='Virginica petal width')

ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
ax[1][1].legend()

plt.show()

## --------------------------------------------------------------------------------- ##

## CONFUSION MATRIX AND ERROR RATE ##

#with all features
confusion_matrix_train, confusion_matrix_test = confusionMatrix()
errorRateTrain(confusion_matrix_train)
errorRateTest(confusion_matrix_test)

X = np.vstack([setosa, versicolor, virginica])
for i in range(num_features-1):
    X = np.delete(X, 1, axis=1) # remove one and one feature from the data set
    X_train = np.vstack([X[:N_first], X[50:50+N_first], X[100:100+N_first]])
    X_test = np.vstack([X[N_first:50], X[50+N_first:100], X[100+N_first:]])
    t_train = np.vstack([np.tile([1,0,0], (N_first, 1)), np.tile([0,1,0], (N_first, 1)), np.tile([0,0,1], (N_first, 1))])

    num_features -= 1 # remove one feature
    W = xavier_init(num_classes, num_features)
    w_io = np.zeros((1, num_classes))  # Init bias

    ## TRAINING ##
    for iteration in range(iterations):
        g = np.dot(X_train, W.T) + w_io
        g_k = sigmoid(g)

        W -= alpha * WgradMSE(g)
        w_io -= alpha * w_oGradMSE(g)

    # confusion matrix and error rate
    confusion_matrix_train, confusion_matrix_test = confusionMatrix()
    errorRateTrain(confusion_matrix_train)
    errorRateTest(confusion_matrix_test)
    print("-----")

## --------------------------------------------------------------------------------- ##
