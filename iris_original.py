import numpy as np
from sklearn.datasets import load_iris
import scipy.special as ss

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
labels = iris['target']

num_classes = 3
num_features = 4

setosa  = data[:50] # class 1 [1,0,0]
versicolor = data[50:100] # class 2 [0,1,0]
virginica = data[100:] # class 3 [0,0,1]

# Training sets
N_train = 30

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

setosa_train = setosa[:N_train]
setosa_train = (setosa_train - mean) / std
versicolor_train = versicolor[:N_train]
versicolor_train = (versicolor_train - mean) / std
virginica_train = virginica[:N_train]
virginica_train = (virginica_train - mean) / std

# Test sets
setosa_test = setosa[N_train:]
versicolor_test = versicolor[N_train:]
virginica_test = virginica[N_train:]

from itertools import chain
test_samples = list(chain(setosa_test, versicolor_test, virginica_test))

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
print("W init: ", W)

def sigmoid_matrix(W, x):
    z_k = W.dot(x)
    return 1/(1+np.exp(-z_k))

def sigmoid_vector(w, x):
    z_k = np.dot(w,x)
    #print("Vector sigmoid: ", z_k)
    return 1/(1+np.exp(-z_k))

# Confusion Matrix
w1 = W[0]
w2 = W[1]
w3 = W[2]

confusion_matrix = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]

print(len(test_samples))

for flower in test_samples[:20]:
    g = [sigmoid_vector(w1, flower), sigmoid_vector(w2, flower), sigmoid_vector(w3, flower)]
    #print("g: ", g)
    gj = np.max(g)
    #print("gj: ", gj)

    if gj == g[0]:
        confusion_matrix[0][0] += 1
    if gj == g[1]:
        confusion_matrix[0][1] += 1
    if gj == g[2]:
        confusion_matrix[0][2] += 1

for flower in test_samples[20:40]:
    g = [sigmoid_vector(w1, flower), sigmoid_vector(w2, flower), sigmoid_vector(w3, flower)]
    #print("g: ", g)
    gj = np.max(g)
    #print("gj: ", gj)

    if gj == g[0]:
        confusion_matrix[1][0] += 1
    if gj == g[1]:
        confusion_matrix[1][1] += 1
    if gj == g[2]:
        confusion_matrix[1][2] += 1

for flower in test_samples[40:]:
    g = [sigmoid_vector(w1, flower), sigmoid_vector(w2, flower), sigmoid_vector(w3, flower)]
    #print("g: ", g)
    gj = np.max(g)
    #print("gj: ", gj)

    if gj == g[0]:
        confusion_matrix[2][0] += 1
    if gj == g[1]:
        confusion_matrix[2][1] += 1
    if gj == g[2]:
        confusion_matrix[2][2] += 1

print(confusion_matrix)

# ---------------------------------------------------------------------

def predict_class(W, x):
    g = np.dot(W, x)  # g = Wx
    g = ss.softmax(g) # creates a probability distribution
    #print("Softmax: ", g)
    return np.argmax(g)  # returns g_j



# Confusion Matrix
confusion_matrix = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]

print(len(test_samples))

for i, flower in enumerate(test_samples):
    true_class = i // 20  # setosa: 0, versicolor: 1, virginica: 2
    predicted_class = predict_class(W, flower)

    confusion_matrix[true_class][predicted_class] += 1

print("Confusion Matrix:")
print(confusion_matrix)

#------------------------------------------------------------
#Error rate
wrong_setosa = 0
wrong_versicolor = 0
wrong_virginica = 0
total_setosa = 0
total_versicolor = 0
total_virginica = 0
for i in range(0, len(confusion_matrix)):
    for j in range(0, len(confusion_matrix[i])):
        if i == 0:
            if i == j:
                total_setosa += confusion_matrix[i][j]
            else: 
                wrong_setosa += confusion_matrix[i][j]
                total_setosa += confusion_matrix[i][j]
        elif i == 1:
            if i == j:
                total_versicolor += confusion_matrix[i][j] 
            else:
                wrong_versicolor += confusion_matrix[i][j]
                total_versicolor += confusion_matrix[i][j]
        elif i == 2:
            if i == j:
                total_virginica += confusion_matrix[i][j]
            else:
                wrong_virginica += confusion_matrix[i][j]
                total_virginica += confusion_matrix[i][j]

ERR_T = (wrong_setosa + wrong_versicolor + wrong_virginica) / (total_setosa + total_versicolor + total_virginica)
ERR_setosa = wrong_setosa / total_setosa
ERR_versicolor = wrong_versicolor / total_versicolor
ERR_virginica = wrong_virginica / total_virginica

print("Total error rate: ", ERR_T)
print("Error rate setosa: ", ERR_setosa)
print("Error rate versicolor: ", ERR_versicolor)
print("Error rate virginica: ", ERR_virginica)

#------------------------------------------------------------