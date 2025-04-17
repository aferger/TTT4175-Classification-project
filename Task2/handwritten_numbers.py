#
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

#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    








n_features = 28*28 # 784
n_classes = 10
n_train = len(x_train) # 60_000
n_test = len(x_test) #10_000

# Training set: x_train (images), y_train (labels)
# Test set: x_test, y_test


# NN-based classifier using Euclidean distance

def euclidean_distance(x, ref): # x = one example (784 features), ref = mu for 784 features
    x = np.array(x).flatten()
    mu = ref
    diff = x - mu
    return np.dot(diff.T, diff)

def feature_vectors(n_train, train_images, train_labels):
    feature_vec0 = []
    feature_vec1 = []
    feature_vec2 = []
    feature_vec3 = []
    feature_vec4 = []
    feature_vec5 = []
    feature_vec6 = []
    feature_vec7 = []
    feature_vec8 = []
    feature_vec9 = []
    for i in range(n_train):
        if train_labels[i] == 0:
            feature0 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec0.append(feature0)

        elif train_labels[i] == 1:
            feature1 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec1.append(feature1)

        elif train_labels[i] == 2:
            feature2 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec2.append(feature2)

        elif train_labels[i] == 3:
            feature3 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec3.append(feature3)

        elif train_labels[i] == 4:
            feature4 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec4.append(feature4)

        elif train_labels[i] == 5:
            feature5 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec5.append(feature5)

        elif train_labels[i] == 6:
            feature6 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec6.append(feature6)

        elif train_labels[i] == 7:
            feature7 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec7.append(feature7)

        elif train_labels[i] == 8:
            feature8 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec8.append(feature8)

        elif train_labels[i] == 9:
            feature9 = np.array(train_images[i]).flatten() # list of len 784
            feature_vec9.append(feature9)
    
    feature_vec0 = np.array(feature_vec0)
    feature_vec1 = np.array(feature_vec1)
    feature_vec2 = np.array(feature_vec2)
    feature_vec3 = np.array(feature_vec3)
    feature_vec4 = np.array(feature_vec4)
    feature_vec5 = np.array(feature_vec5)
    feature_vec6 = np.array(feature_vec6)
    feature_vec7 = np.array(feature_vec7)
    feature_vec8 = np.array(feature_vec8)
    feature_vec9 = np.array(feature_vec9)

    return [feature_vec0, feature_vec1, feature_vec2, feature_vec3, feature_vec4, feature_vec5, feature_vec6, feature_vec7, feature_vec8, feature_vec9]


def compute_ref1(feature_vec):
    mu = []
    for i in range(len(feature_vec)):
        mu.append(np.sum(feature_vec[i], axis=0)/len(feature_vec[i]))
    return mu

def nearest_neighbour(x, mu):
    min_dist = 10**20
    for i in range(len(mu)):
        dist = euclidean_distance(x,mu[i])
        if min_dist > dist:
            min_dist = dist
            best_class = i
    return best_class

# Confusion matrix
def find_confusion_matrix(n_classes, x_test, y_test, mu):

    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    misclassified_images = []
    correct_classified_images = []

    for i, test_image in enumerate(x_test):
        true_class = y_test[i]  # The correct class
        predicted_class = nearest_neighbour(np.array(test_image).flatten(), mu)  # Classify test image

        confusion_matrix[true_class][predicted_class] += 1  # Update confusion matrix

        if true_class == predicted_class:
            correct_classified_images.append(test_image)
        else:
            misclassified_images.append(test_image)
    
    return confusion_matrix, misclassified_images, correct_classified_images


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
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


def task1a():
    print("----------------- Task 1a ---------------------")
    feature_vector = feature_vectors(n_train, x_train, y_train)
    mu = compute_ref1(feature_vector)

    start = time.time()
    confusion_matrix, misclassified_images, correct_classified_images = find_confusion_matrix(n_classes, x_test, y_test, mu)
    end = time.time()

    plot_confusion_matrix(confusion_matrix, class_labels=[0,1,2,3,4,5,6,7,8,9])
    find_error_rate(confusion_matrix)

    print(f"Time: {end-start} seconds")


# Plot of some misclassified numbers --------------------------------------------------------

def plot_classified_images(classified_images, title):
    plt.title(title)
    plt.axis('off')  # Hide the axis labels

    # Add the first image to the figure (top-left position)
    plt.subplot(2, 2, 1)  # 2 rows, 2 columns, first position
    plt.imshow(classified_images[0])  

    # Add the second image to the figure (top-right position)
    plt.subplot(2, 2, 2)  # 2 rows, 2 columns, second position
    plt.imshow(classified_images[1])  

    # Add the third image to the figure (bottom-left position)
    plt.subplot(2, 2, 3)  # 2 rows, 2 columns, third position
    plt.imshow(classified_images[2]) 

    # Add the fourth image to the figure (bottom-right position)
    plt.subplot(2, 2, 4)  # 2 rows, 2 columns, fourth position
    plt.imshow(classified_images[3])  

    plt.show()

def task1bc():
    print("---------------- Task 1b and 1c --------------------")
    feature_vector = feature_vectors(n_test, x_train, y_train)
    mu = compute_ref1(feature_vector)
    confusion_matrix, misclassified_images, correct_classified_images = find_confusion_matrix(n_classes, x_test, y_test, mu)
    plot_classified_images(misclassified_images, "Misclassified images")
    plot_classified_images(correct_classified_images, "Correctly classified images")


# -------------------------------------------------------------------------------






# 2. Clustering 
from sklearn.cluster import KMeans

M = 64  #number of clusters
n_train = 6000
train_v = feature_vectors(n_test, x_train, y_train)


def class_template_clusters(n_clusters, train_v):
    C = []
    for i in range(len(train_v)):
        kmeans = KMeans(n_clusters, random_state=42)
        kmeans.fit_predict(train_v[i])
        Ci = kmeans.cluster_centers_
        C.append(Ci)
    return C

C = class_template_clusters(M,train_v) #Len 10 x 64
all_templates = np.vstack(C)  # Shape: (640, 784)
template_labels = np.array([i for i in range(10) for _ in range(64)])  # Shape: (640,)


# Confusion matrix

def NN_label(image, templates, template_labels):
    image_flat = np.array(image).flatten()

    #Euclidean distance:
    distances = np.linalg.norm(templates-image_flat, axis=1)
    
    nearest_idx = np.argmin(distances)
    return template_labels[nearest_idx]


def confusion_matrix_64_template(all_templates, template_labels, y_test):
    from sklearn.metrics import confusion_matrix
    y_pred = []
    for img in x_test:
        pred_label = NN_label(img, all_templates, template_labels)
        y_pred.append(pred_label)
    y_pred = np.array(y_pred)

    confusion_matrix = confusion_matrix(y_test, y_pred)
    return confusion_matrix

def task2ab():
    print("------------------ Task 2a and 2b -------------------")
    start = time.time()
    confusion_matrix = confusion_matrix_64_template(all_templates, template_labels, y_test)
    end = time.time()
    plot_confusion_matrix(confusion_matrix, class_labels=[0,1,2,3,4,5,6,7,8,9])
    find_error_rate(confusion_matrix)
    print(f"Time: {end-start}")
    return




# KNN classifier with K = 7

def KNN_labels(image, templates, template_labels, K=1):
    image_flat = np.array(image).flatten()

    #Euclidean distance:
    distances = np.linalg.norm(templates-image_flat, axis=1)
    
    # Finding K closest distances:
    label_dist = []
    for i in range(0, K):
        idx = np.argmin(distances)
        label_dist.append([template_labels[idx], distances[idx]])
        distances = np.delete(distances, idx)
    
    return label_dist


def confusion_matrix_7NN(all_templates, template_labels, y_test):
    from sklearn.metrics import confusion_matrix
    y_pred = []
    for img in x_test:
        label_dist = KNN_labels(img, all_templates, template_labels,7)
        
        # Checking the number of each labels
        pred_labels = [row[0] for row in label_dist]
        dists = [row[1] for row in label_dist]
        num_labels = {}
        for label in pred_labels:
            if label in num_labels.keys():
               num_labels[label] += 1
            else:
               num_labels[label] = 1

        highest_num = 0
        min_dist = 10**20
        idx = 0
        for label, num in num_labels.items():
            if num > highest_num:
                highest_num = num
                pred_label = label
                min_dist = dists[idx]
            elif num == highest_num:
                dist = dists[idx]
                if dist < min_dist:
                    pred_label = label
            idx += 1

        y_pred.append(pred_label)
    y_pred = np.array(y_pred)

    confusion_matrix = confusion_matrix(y_test, y_pred)
    return confusion_matrix
    

def task2c():
    print("------------ Task 2c -------------")
    start = time.time()
    confusion_matrix = confusion_matrix_7NN(all_templates, template_labels, y_test)
    end = time.time()
    plot_confusion_matrix(confusion_matrix, class_labels=[0,1,2,3,4,5,6,7,8,9])
    find_error_rate(confusion_matrix)
    print(f"Time: {end-start}")



# --------------- Run tasks ------------------
# For it to be easier to observe each task, only remove '#' for one task at a time

task1a() # Task 1a
# task1bc() # Task 1b and 1c
# task2ab() # Task 2a and 2b
# task2c() # Task 2c
