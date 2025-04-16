import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import scipy.special as ss

## CREATING TRAINING SET AND HISTOGRAM PLOTS ##

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']
labels = iris['target']

num_classes = 3
num_features = 4
N_train = 30

setosa  = data[:50] # class 1 [1,0,0]
versicolor = data[50:100] # class 2 [0,1,0]
virginica = data[100:150] # class 3 [0,0,1]

#print("Setosa train: \n", setosa_train)
#print("Versicolor train: \n", versicolor_train)
#print("Virginica train: \n", virginica_train)

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

# Plotting                                              SKAL DETTE KUN VÆRE TRAINING SET?
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


#-------------------------------------------------------------------


# Finding the feature with the most overlap
# The feature with the most overlap is the one with the lowest Fisher score
# The Fisher Score
petal_length = [setosa_plengths,versicolor_plengths, virginica_plengths]
petal_width = [setosa_pwidths, versicolor_pwidths, virginica_pwidths]
sepal_length = [setosa_slengths, versicolor_slengths, virginica_slengths]
sepal_width = [setosa_swidths, versicolor_swidths, virginica_swidths]

def fisher_score(feature):
    n_c = 30
    numerator = 0
    denumerator = 0
    for c in feature:
        numerator += n_c * (np.mean(c)-np.mean(feature))**2
        denumerator += n_c * np.std(c)**2

    return numerator/denumerator

print(f"\n----- The fisher scores for each feature: -----")
print(f"Sepal length: {fisher_score(sepal_length)}")
print(f"Sepal width: {fisher_score(sepal_width)}")
print(f"Petal length: {fisher_score(petal_length)}")
print(f"Petal width: {fisher_score(petal_width)}\n")
print(f"The fisher score of the feature petal width is lowest at {fisher_score(petal_width)} which means it is the feature with the most overlap.")


#-----------------------------------------------------------------------

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

## GLOBAL VARIABLES ##
# Gradient descent
alpha = 0.01
iterations = 10000

#------------------------------------------------------------------
# Removing the feature with the most overlap (petal width)

X = np.vstack([setosa, versicolor, virginica])
X_threefeatures = np.delete(X, 1, axis=1) # remove petal width

# Training sets
X_train = np.vstack([X_threefeatures[:N_train], X_threefeatures[50:50+N_train], X_threefeatures[100:100+N_train]])

# Test sets
X_test = np.vstack([X_threefeatures[N_train:50], X_threefeatures[50+N_train:100], X_threefeatures[100+N_train:]])

# Labels
t_setosa = np.tile([1,0,0], (N_train, 1))
t_versicolor = np.tile([0,1,0], (N_train, 1))
t_virginica = np.tile([0,0,1], (N_train, 1))
t_train = np.vstack([t_setosa, t_versicolor, t_virginica])

## INITIALIZATION ##

# W = C x D
num_features -= 1 # remove petal width
W = xavier_init(num_classes, num_features)
w_io = np.zeros((1, num_classes))  # Init bias


## TRAINING ##
for iteration in range(iterations):
    g = np.dot(X_train, W.T) + w_io
    g_k = sigmoid(g)

    W -= alpha * WgradMSE(g)
    w_io -= alpha * w_ioGradMSE(g)


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

print("Confution matrix on traing set with three features:")
print(confusion_matrix_train)
print("Confution matrix on test set with three features:")
print(confusion_matrix_test)

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

#------------------------------------------------------------------
# Removing the feature with the second most overlap (petal length)

X_twofeatures = np.delete(X_threefeatures, 1, axis=1) # remove petal width
# Training sets
X_train = np.vstack([X_twofeatures[:N_train], X_twofeatures[50:50+N_train], X_twofeatures[100:100+N_train]])
# Test sets
X_test = np.vstack([X_twofeatures[N_train:50], X_twofeatures[50+N_train:100], X_twofeatures[100+N_train:]])


## INITIALIZATION ##

# W = C x D
num_features -= 1 # remove petal width
W = xavier_init(num_classes, num_features)

## TRAINING ##
for iteration in range(iterations):
    g = np.dot(X_train, W.T) + w_io
    g_k = sigmoid(g)

    W -= alpha * WgradMSE(g)
    w_io -= alpha * w_ioGradMSE(g)


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

print("Confution matrix on traing set with two features:")
print(confusion_matrix_train)
print("Confution matrix on test set with two features:")
print(confusion_matrix_test)

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

#------------------------------------------------------------------
#Removing the feature with the third most overlap (sepal width)

X_onefeature = np.delete(X_twofeatures, 1, axis=1) # remove petal width
# Training sets
X_train = np.vstack([X_onefeature[:N_train], X_onefeature[50:50+N_train], X_onefeature[100:100+N_train]])
# Test sets
X_test = np.vstack([X_onefeature[N_train:50], X_onefeature[50+N_train:100], X_onefeature[100+N_train:]])


## INITIALIZATION ##

# W = C x D
num_features -= 1 # remove petal width
W = xavier_init(num_classes, num_features)


## TRAINING ##
for iteration in range(iterations):
    g = np.dot(X_train, W.T) + w_io
    g_k = sigmoid(g)

    W -= alpha * WgradMSE(g)
    w_io -= alpha * w_ioGradMSE(g)


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

print("Confution matrix on traing set with one feature:")
print(confusion_matrix_train)
print("Confution matrix on test set with one feature:")
print(confusion_matrix_test)

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

#------------------------------------------------------------------