import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import scipy.special as ss

iris = load_iris() # sepal length, sepal width, petal length, petal width
data = iris['data']
labels = iris['target']

num_classes = 3
num_features = 4

setosa_train  = data[:30] # class 1 [1,0,0]
versicolor_train = data[50:80] # class 2 [0,1,0]
virginica_train = data[100:130] # class 3 [0,0,1]

print("Setosa: \n", setosa_train)
print("Versicolor: \n", versicolor_train)
print("Virginica: \n", virginica_train)

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

