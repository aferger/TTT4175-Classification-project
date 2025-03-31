import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Last inn datasettet
iris = load_iris()
X = iris.data  # 4 features
y = iris.target  # 3 klasser

# Del opp trenings- og testsett (de første 30 til trening, siste 20 til test)
X_train = np.vstack([X[:30], X[50:80], X[100:130]])
#print("X_train: ", X_train)
y_train = np.hstack([y[:30], y[50:80], y[100:130]])

X_test = np.vstack([X[30:50], X[80:100], X[130:150]])
y_test = np.hstack([y[30:50], y[80:100], y[130:150]])

# One-hot encoding av labels
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

#print("y_train_onehot: ", y_train_onehot)
#print("t_test_onehot: ", y_test_onehot)

# Modellparametere
#np.random.seed(42)
n_features = X_train.shape[1]  # 4
n_classes = 3  # 3 klasser
def xavier_init(fan_in, fan_out):
    limit = np.sqrt(1 / fan_in)  # Xavier-uniform range
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

# W = C x D = 3klasser x 4målinger = weight
W = xavier_init(n_features, n_classes)

b = np.zeros((1, n_classes))  # Init bias
#print("b: ", b)

# Softmax-funksjon
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # For numerisk stabilitet
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Krysstapsentropi-tap
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Legger til epsilon for stabilitet

# Gradient descent
learning_rate = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    # Forward pass
    Z = np.dot(X_train, W) + b
    y_pred = softmax(Z)
    
    # Tap
    loss = cross_entropy_loss(y_train_onehot, y_pred)
    losses.append(loss)
    
    # Backpropagation
    m = X_train.shape[0]
    dZ = y_pred - y_train_onehot
    #print("dZ: ", dZ)
    #print("len(dZ): ", len(dZ))
    dW = np.dot(X_train.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    
    
    # Oppdater parametere
    W -= learning_rate * dW
    b -= learning_rate * db
    
    # Print loss noen ganger
    #if epoch % 100 == 0:
        #print(f"Epoch {epoch}, Loss: {loss:.4f}")

#print("g = ", Z)
#print("bgradMSE: ", db)
print("W:\n", W)

# Plot loss
#plt.plot(losses)
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.title("Training Loss")
#plt.show()

# Evaluering på testsettet
Z_test = np.dot(X_test, W) + b
y_test_pred = softmax(Z_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Beregn nøyaktighet
accuracy = np.mean(y_test_pred_labels == y_test) * 100
#print(f"Test Accuracy: {accuracy:.2f}%")

import numpy as np
from sklearn.metrics import confusion_matrix

# Lag forvirringsmatrisen
conf_matrix = confusion_matrix(y_test, y_test_pred_labels)

# Lag en numpy-matrise med labels
labels = iris.target_names  # ['setosa', 'versicolor', 'virginica']
conf_matrix_with_labels = np.vstack([[""] + list(labels)] + np.column_stack([labels, conf_matrix]))

# Print matrisen på en lesbar måte
print("\nConfusion Matrix:\n")
for row in conf_matrix_with_labels:
    print("  ".join(f"{str(x):<10}" for x in row))