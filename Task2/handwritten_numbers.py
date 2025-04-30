# To run tasks, see end of file.

## Loading data written by Hojjat Khodabakhsh at https://www.kaggle.com/code/hojjatk/read-mnist-dataset
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import time
import seaborn as sns
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    

#
# Verify Reading Dataset via MnistDataloader class
#
#%matplotlib inline
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = 'Task2/database_handwritten_num'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
## END loading of dataset ----------



n_features = 28*28 # 784
n_classes = 10
n_train = len(x_train) # 60_000
n_test = len(x_test) #10_000

# Training set: x_train (images), y_train (labels)
# Test set: x_test, y_test

def classsorted_train(x_train, y_train):
    n_train = len(y_train)
    classsorted_train_img = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(0,n_train):
        label = y_train[i]
        train_img = np.array(x_train[i]).flatten()
        classsorted_train_img[label].append(train_img)
    return classsorted_train_img


# NN-based classifier using Euclidean distance

def nearest_neighbor(img, x_template, y_template):   
    distances = np.linalg.norm(x_template - img, axis=1) # Euclidean distance
    min_dist_idx = np.argmin(distances) # index of smallest distance
    return y_template[min_dist_idx] # Class of smallest distance

def confusion_matrix_NN(n_classes, x_test_flat, y_test, x_template, y_template, x_test):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    misclassified_images = []
    misclassified_images_predicted_label = []
    correct_classified_images = []
    for i, test_image in enumerate(x_test_flat):
        true_class = y_test[i] # True class 
        predicted_class = nearest_neighbor(test_image, x_template, y_template) # Predicted class (NN)
        confusion_matrix[true_class][predicted_class] += 1 # Update confusion matrix

        if true_class == predicted_class:
            correct_classified_images.append(x_test[i]) # Add correctly classified image
        else:
            misclassified_images.append(x_test[i]) # Add misclassified image
            misclassified_images_predicted_label.append(predicted_class) # Add its predicted class
        
    return confusion_matrix, misclassified_images, correct_classified_images, misclassified_images_predicted_label


def confusion_matrix_prosent(confusion_matrix, x_test, y_test):
    sorted_test = classsorted_train(x_test, y_test)
    confusion_matrix_rate = []
    for i, row in enumerate(confusion_matrix):
        row_prosent = []
        num_examples = len(sorted_test[i])
        for j in range(0,len(row)):
            row_prosent.append(row[j]/num_examples *100)
        confusion_matrix_rate.append(np.array(row_prosent))

    return np.array(confusion_matrix_rate)

# Error rate
def find_error_rate(confusion_matrix):
    total_predictions = np.sum(confusion_matrix)
    correct_predictions = np.trace(confusion_matrix) # (sum of diagonal elements)
    incorrect_predictions = total_predictions - correct_predictions

    error_rate = incorrect_predictions / total_predictions
    print(f"Error Rate: {error_rate:.4f} ({error_rate * 100:.2f}%)")
    return error_rate


def plot_confusion_matrix(conf_mat, class_labels=None, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


# Plot of some misclassified numbers --------------------------------------------------------

def plot_classified_images(classified_images, title, true_label=0):
    plt.title(title)
    plt.axis('off')  # Hide the axis labels

    # Add the first image to the figure (top-left position)
    plt.subplot(2, 2, 1)
    plt.imshow(classified_images[0])  

    # Add the second image to the figure (top-right position)
    plt.subplot(2, 2, 2)
    plt.imshow(classified_images[1])  

    # Add the third image to the figure (bottom-left position)
    plt.subplot(2, 2, 3)  
    plt.imshow(classified_images[2]) 

    # Add the fourth image to the figure (bottom-right position)
    plt.subplot(2, 2, 4)
    plt.imshow(classified_images[3])  

    if true_label != 0:
        print("Predicted classes of misclassified images: ")
        print(f"Top left: {true_label[0]}")
        print(f"Top right: {true_label[1]}")
        print(f"Bottom left: {true_label[2]}")
        print(f"Bottom right: {true_label[3]}")

    plt.show()

# -------------------------------------------------------------------------------

def task1():
    print("----------------- NN classifier using whole training set ---------------------")
    labels = [0,1,2,3,4,5,6,7,8,9]
    x_train_flat = np.array([np.array(img, dtype=np.float32).flatten() / 255.0 for img in x_train]) # Normalize training set
    x_test_flat = np.array([np.array(img, dtype=np.float32).flatten() / 255.0 for img in x_test]) # Normalize test set

    start = time.time()
    confusion_matrix, misclassified_images, correct_classified_images, true_label = confusion_matrix_NN(n_classes, x_test_flat, y_test, x_train_flat, y_train, x_test)
    end = time.time()

    print(confusion_matrix)
    plot_classified_images(misclassified_images, "Misclassified images", true_label)
    plot_classified_images(correct_classified_images, "Correctly classified images")
    confusion_matrix_rate = confusion_matrix_prosent(confusion_matrix, x_test, y_test)
    plot_confusion_matrix(confusion_matrix_rate, class_labels=labels)
    find_error_rate(confusion_matrix)

    print(f"Time: {end-start} seconds")



# 2. Clustering 
from sklearn.cluster import KMeans

def class_template_clusters(n_clusters, train_v):
    C = []
    for i in range(len(train_v)):
        kmeans = KMeans(n_clusters, random_state=42)
        kmeans.fit_predict(train_v[i])
        Ci = kmeans.cluster_centers_
        C.append(Ci)
    return C

def clustered_template(M, x_train, y_train):
    train_v = classsorted_train(x_train, y_train)
    C = class_template_clusters(M,train_v) #Len 10 x 64
    x_templates = np.vstack(C)  # Shape: (640, 784)
    y_templates = np.array([i for i in range(n_classes) for _ in range(M)])  # Shape: (640,)
    return x_templates, y_templates

def confusion_matrix_NN_cluster(n_classes, x_test, y_test, x_template, y_template):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    misclassified_images = []
    correct_classified_images = []
    misclassified_images_predicted_label = []
    for i, test_image in enumerate(x_test):
        true_class = y_test[i]  # The correct class
        predicted_class = nearest_neighbor(np.array(test_image).flatten(), x_template, y_template) # Classify test image
        confusion_matrix[true_class][predicted_class] += 1  # Update confusion matrix

        if true_class == predicted_class:
            correct_classified_images.append(test_image)
        else:
            misclassified_images.append(test_image)
            misclassified_images_predicted_label.append(predicted_class)
    
    return confusion_matrix, misclassified_images, correct_classified_images, misclassified_images_predicted_label

def task2b():
    print("----------------- NN classifier using clustering ---------------------")
    M = 64
    x_templates, y_templates = clustered_template(M, x_train, y_train)

    start = time.time()
    confusion_matrix, misclassified_images, correct_classified_images, pred_labels = confusion_matrix_NN_cluster(n_classes, x_test, y_test, x_templates, y_templates)
    end = time.time()

    print(confusion_matrix)
    confusion_matrix_rate = confusion_matrix_prosent(confusion_matrix, x_test, y_test)
    plot_classified_images(misclassified_images, "Misclassified images", pred_labels)
    plot_classified_images(correct_classified_images, "Correctly classified images")
    plot_confusion_matrix(confusion_matrix_rate, class_labels=[0,1,2,3,4,5,6,7,8,9])
    find_error_rate(confusion_matrix)
    print(f"Time: {end-start}")   


def K_nearest_neighbor(img, x_template, y_template, K):    
    distances = np.linalg.norm(x_template - img, axis=1)

    # Nearest K neighbors:
    label_dist = []
    for i in range(0,K):
        idx = np.argmin(distances)
        label_dist.append([y_template[idx], distances[idx]])
        distances = np.delete(distances, idx)
    
    # Class with most votes and (if tied) class with the lowest sum of distances.
    num_labels = {}
    for i in range(0,len(label_dist)):
        label = label_dist[i][0]
        if label in num_labels.keys():
            num_labels[label][0] += 1
            num_labels[label][1] += label_dist[i][1]
        else:
            num_labels[label] = [1, label_dist[i][1]]
    
    highest_num = 0
    lowest_sum = 10**20
    pred_label = 0
    for label, num in num_labels.items():
        if num[0] > highest_num:
            highest_num = num[0]
            pred_label = label
            lowest_sum = num[1]
        elif num[0] == highest_num and num[1] < lowest_sum:
            pred_label = label
            lowest_sum = num[1]

    return pred_label


def find_confusion_matrix_KNN(n_classes, x_test, y_test, x_template, y_template, K):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    misclassified_images = []
    correct_classified_images = []
    misclassified_images_predicted_labels = []
    for i, test_image in enumerate(x_test):
        true_class = y_test[i]  # The correct class
        predicted_class = K_nearest_neighbor(np.array(test_image).flatten(), x_template, y_template, K) # Classify test image
        confusion_matrix[true_class][predicted_class] += 1  # Update confusion matrix

        if true_class == predicted_class:
            correct_classified_images.append(test_image)
        else:
            misclassified_images.append(test_image)
            misclassified_images_predicted_labels.append(predicted_class)
    
    return confusion_matrix, misclassified_images, correct_classified_images, misclassified_images_predicted_labels


def task2c():
    print("----------------- KNN classifier using clustering ---------------------")
    K = 7
    M = 64
    x_templates, y_templates = clustered_template(M, x_train, y_train)

    start = time.time()
    confusion_matrix, misclassified_images, correct_classified_images, pred_labels = find_confusion_matrix_KNN(n_classes, x_test, y_test, x_templates, y_templates, K)
    end = time.time()
    print(confusion_matrix)
    confusion_matrix_rate = confusion_matrix_prosent(confusion_matrix, x_test, y_test)
    plot_classified_images(misclassified_images, "Misclassified images", pred_labels)
    plot_classified_images(correct_classified_images, "Correctly classified images")
    plot_confusion_matrix(confusion_matrix_rate, class_labels=[0,1,2,3,4,5,6,7,8,9])
    find_error_rate(confusion_matrix)
    print(f"Time: {end-start}")




# -------------------- BEGIN PLOTTING ERROR RATES -----------------------

def error_rate_of_clusters(M):
    x_templates, y_templates = clustered_template(M, x_train, y_train)
    confusion_matrix, misclassified_images, correct_classified_images, pred_labels = confusion_matrix_NN_cluster(n_classes, x_test, y_test, x_templates, y_templates)
    error_rate = find_error_rate(confusion_matrix)
    return error_rate

def plot_error_rate_clusters(M_max):
    errors =[]
    Ms = np.arange(1,M_max+1,7)
    for M in Ms:
        errors.append(error_rate_of_clusters(M)*100)
    plt.plot(Ms,errors, marker='o')
    plt.xlabel('Number of clusters (M)')
    plt.ylabel('Error rate (%)')
    plt.title('Error Rate vs M in NN-classifier')
    plt.show()

M_max = 210
#plot_error_rate_clusters(M_max)

def error_rate_of_K(K):
    M=64
    x_templates, y_templates = clustered_template(M, x_train, y_train)
    confusion_matrix, misclassified_images, correct_classified_images = find_confusion_matrix_KNN(n_classes, x_test, y_test, x_templates, y_templates, K)
    error_rate = find_error_rate(confusion_matrix)
    return error_rate


def plot_error_rate_K(K_max):
    errors = []
    Ks = np.arange(1,K_max+1)
    for K in Ks:
        errors.append(error_rate_of_K(K)*100)
    plt.plot(Ks,errors, marker='o')
    plt.xlabel('K value')
    plt.ylabel('Error rate (%)')
    plt.title('Error Rate vs K in KNN-classifier')
    plt.show()
K_max = 15
#plot_error_rate_K(K_max)

# ------------------- END PLOTTING ERROR RATES --------------------------


# --------------- Run tasks ------------------
# For it to be easier to observe each task, only remove '#' for one task at a time

#task1() # Task 1
#task2b() # Task 2b
#task2c() # Task 2c