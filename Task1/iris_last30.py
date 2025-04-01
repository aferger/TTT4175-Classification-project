import numpy as np
from sklearn.datasets import load_iris


## VARIABLE DEFINITIONS ##

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']
label = iris['target']

num_classes = 3
num_features = 4

setosa  = data[:50] # class 1 [1,0,0]
versicolor = data[50:100] # class 2 [0,1,0]
virginica = data[100:] # class 3 [0,0,1]

# Training sets
N_train = 20

setosa_train = setosa[N_train:]
versicolor_train = versicolor[N_train:]
virginica_train = virginica[N_train:]
X_train = np.vstack([setosa_train, versicolor_train, virginica_train])
print("len of training set: ", len(X_train))

# Test sets
setosa_test = setosa[:N_train]
versicolor_test = versicolor[:N_train]
virginica_test = virginica[:N_train]
X_test = np.vstack([setosa_test, versicolor_test, virginica_test])
               
# Labels
len_train = 30
t_setosa = np.tile([1,0,0], (len_train, 1))
t_versicolor = np.tile([0,1,0], (len_train, 1))
t_virginica = np.tile([0,0,1], (len_train, 1))
t_train = np.vstack([t_setosa, t_versicolor, t_virginica])


## FUNCTION DEFINITIONS ##

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#recomended random function for sigmoid
#creates a random matrix with the same shape as W
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(1 / fan_in)  # Xavier-uniform range
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def WgradMSE(g):
    # divide by m to avoid unstable learning
    m = X_train.shape[0]
    dg = sigmoid(g) - t_train
    return np.dot(dg.T, X_train) / m

def w_ioGradMSE(g):
    # divide by m to avoid unstable learning
    m = X_train.shape[0]
    dg = sigmoid(g) - t_train
    return np.sum(dg, axis=0, keepdims=True) / m


## INITIALIZATION ##

# W = C x D
W = xavier_init(num_classes, num_features)
print("W init: ", W)
w_io = np.zeros((1, num_classes))  # Init bias

# Gradient descent
alpha = 0.01
iterations = 10000

## TRAINING ##
for iteration in range(iterations):
    g = np.dot(X_train, W.T) + w_io
    g_k = sigmoid(g)

    W -= alpha * WgradMSE(g)
    w_io -= alpha * w_ioGradMSE(g)

print("W trained: ", W)

#----------------------------------------------------------

## CONFUTION MATRIX ##

# initalization
confusion_matrix_train = np.zeros((num_classes, num_classes), dtype=int)
confusion_matrix_test = np.zeros((num_classes, num_classes), dtype=int)

# adding to confusion matrixes
for i in range(len(X_train)):
    g = sigmoid(np.dot(X_train[i], W.T) + w_io)

    true_train_class = np.argmax(t_train[i])
    #print("True train class: ", true_train_class)
    predicted_class = np.argmax(g)

    confusion_matrix_train[true_train_class][predicted_class] += 1

for i in range(len(X_test)):
    g = sigmoid(np.dot(X_test[i], W.T) + w_io)
    
    predicted_class = np.argmax(g)
    true_test_class = i//20

    confusion_matrix_test[true_test_class][predicted_class] += 1

print("Confution matrix on traing set:")
print(confusion_matrix_train)
print("Confution matrix on test set:")
print(confusion_matrix_test)

#------------------------------------------------------------------

## ERROR RATE ##
wrongly_classified_train = confusion_matrix_train.sum(axis=1) - np.diag(confusion_matrix_train)
total_classified_train = confusion_matrix_train.sum(axis=1)

wrongly_classified_test = confusion_matrix_test.sum(axis=1) - np.diag(confusion_matrix_test)
total_classified_test = confusion_matrix_test.sum(axis=1)

ERR_T = wrongly_classified_train.sum() / total_classified_train.sum()
ERR_setosa = wrongly_classified_train[0] / total_classified_train[0]
ERR_versicolor = wrongly_classified_train[1] / total_classified_train[1]
ERR_virginica = wrongly_classified_train[2] / total_classified_train[2]

print("ERROR RATE TRAIN SET: ")
print("Total error rate: ", ERR_T)
print("Error rate setosa: ", ERR_setosa)
print("Error rate versicolor: ", ERR_versicolor)
print("Error rate virginica: ", ERR_virginica)

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

#------------------------------------------------------------