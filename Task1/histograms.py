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

setosa_train  = data[:30] # class 1 [1,0,0]
versicolor_train = data[50:80] # class 2 [0,1,0]
virginica_train = data[100:130] # class 3 [0,0,1]

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
for i in range(0, len(setosa_train)):
    setosa_plengths.append(setosa_train[i][0])
    setosa_pwidths.append(setosa_train[i][1])
    setosa_slengths.append(setosa_train[i][2])
    setosa_swidths.append(setosa_train[i][3])
for i in range(0, len(versicolor_train)):
    versicolor_plengths.append(versicolor_train[i][0])
    versicolor_pwidths.append(versicolor_train[i][1])
    versicolor_slengths.append(versicolor_train[i][2])
    versicolor_swidths.append(versicolor_train[i][3])
for i in range(0, len(virginica_train)):
    virginica_plengths.append(virginica_train[i][0])
    virginica_pwidths.append(virginica_train[i][1])
    virginica_slengths.append(virginica_train[i][2])
    virginica_swidths.append(virginica_train[i][3])

# Plotting
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

# finding the feature with the most overlap
# it is easy to see from the histogram plot that petal lengt and width have the most overlap

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

