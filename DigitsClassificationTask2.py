import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from ConfusionMatrixPrinter import _test_cm, pretty_plot_confusion_matrix
from pandas import DataFrame
from scipy.cluster.vq import kmeans
from collections import Counter


# Reference to find execution time of the program
start_time = time.time()


def fetch_data(test_size=10000, standardize=True):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def distance(img1, img2):
    return np.sqrt(np.sum((img2 - img1)**2))


def nearest_neighbor_cluster(img1, clusters):
    """
    Finds the index of the dataset that minimizes the distance to img1.

    Parameters:
        img1:           the data to classify
        clusters:       array clustered classes
    
    Returns:    
        nn_index:       the index of the nearest neighbor in the training examples, also represent the class
    """

    min_distance = np.inf
    nn_index = 0

    # Iterating through the classes
    for i in range(len(clusters)):
        # Iterating through the clusters
        for j in range(len(clusters[i])):
            img = clusters[i][j]
            temp_distance = distance(img, img1)

            if temp_distance < min_distance:
                min_distance = temp_distance
                nn_index = i

    return nn_index


def weighted_most_common(classes, min_distances):
    """
    Returns the most common element in classes based on weights in min_distances.
    
    Parameters:
        classes:            list of elements of which to find the most common
        min_distances:      weights for each element in classes
        
    Returns:
        class_id:           the id in classes that is most common, weighted
    """

    k = len(classes)
    sum_distances = np.sum(min_distances)
    weights = []
    counter = Counter(classes).most_common()

    # Must use arrays
    classes = np.array(classes)
    min_distances = np.array(min_distances)

    for i in range(k):
        weights.append(min_distances[i] / sum_distances)
    
    weight_list = []

    for i in range(len(counter)):
        mask = (classes == counter[i][0])
        w = min_distances[mask]
        weight_sum = 0

        for j in range(counter[i][1]):
            weight_sum += (1 / w[j])

        weight_list.append(weight_sum)
    
    class_index_in_counter = np.argmax(weight_list)
    prediction = counter[class_index_in_counter][0]

    return prediction





def k_nearest_neighbor_cluster(img1, clusters, k=1):
    """
    Finds the k nearest neighbors and performs a voting among the neighbors for making a prediction.
    
    Parameters:
        img1:           the image to find the distance to
        clusters:       the clustered data to use as example images
        k:              the number of nearest neighbors to take into account
        
    Returns:
        prediction:     the predicted class of which minimize the distance from img1 to the k nearest neighbors
    """

    min_distances = [np.inf for i in range(k)] # holds the k lowest distances
    neighbors = [[] for i in range(k)] # the k nearest neighbors


    # Find the k nearest neighbors
    for i in range(len(clusters)):
        for j in range(len(clusters[0])):
            img = clusters[i][j]
            temp_distance = distance(img, img1)

            if temp_distance < max(min_distances):
                index = np.argmax(min_distances)
                min_distances[index] = temp_distance
                neighbors[index] = [i, j]

    # Voting among the neighbors
    classes = [neighbors[i][0] for i in range(k)]
    prediction = weighted_most_common(classes, min_distances) # Counter(classes).most_common(1)[0][0]
    
    return prediction


def create_clusters(data, labels, M=64):
    """
    Extract each class and performs clustering on it. Then all clustered classes are put back together.

    Parameters:
        data:               the data to cluster
        labels:             label of the training data to see which class each example belongs to
        M:                  the number of clusters in each class

    Returns:
        clustered_data:     the clustered data, three dimensional [class][cluster][data point]
        clustered_labels:   labels for the clustered data
    """

    clustered_data = []

    # Perform clustering on each class
    for class_id in range(10):
        # Separating to a single class
        mask = (labels == str(class_id))
        class_data = data[mask]

        # Cluster one class at a time and add to clustered_data
        centroids, distortion = kmeans(class_data, M)
        clustered_data.append(centroids)
    
    clustered_data = np.array(clustered_data)
    return clustered_data


def test(clustered_data, test_data, test_labels, test_num=1000, k=1):
    """
    Perform test on a determined set of test data.

    Parameters:
        clustered_data:             example data as cluster, label given as index
        test_data:                  the data to test on
        test_labels:                labels for test_data
        test_num:                   how many samples of the test data to test on

    Returns:
        error_rate:                 the error rate of the tests, false classification / total classifications
        confusion_matrix:           the confusion matrix of the test
        indexes_false_classified:   indexes of all falsesly classified images
        indexes_correct_classified: indexes of all correctly classified images
    """

    # To find error rate and confusion matrix
    total_classifications = test_num
    total_correct = 0
    total_false = 0

    # What indexes of the test set were we unable to classify correctly, each element as [index of true, predicted value]
    indexes_false_classified = []

    # What indexes of the test set were we able to classify correctly, each element as [index of label]
    indexes_correct_classified = []

    # Confusion matrix is a 10 x 10 matrix
    confusion_matrix = [[0 for i in range(10)] for j in range(10)]


    # Prediction on all the tests
    for i in range(test_num):
        prediction = k_nearest_neighbor_cluster(test_data[i], clustered_data, k)
        
        if prediction == int(test_labels[i]):
            total_correct += 1
            indexes_correct_classified.append(i)
        else:
            total_false += 1
            indexes_false_classified.append([i, prediction])

        confusion_matrix[int(test_labels[i])][prediction] += 1 # add a count to the correct location in confusion matrix

        # Progress
        p = total_classifications / 10
        if (i+1) % p == 0:
            print("{} %".format((i+1)*100/test_num))


    # In case we count wrong
    if total_correct + total_false != total_classifications:
        print("Error! Number of total classifications incorrect.")


    error_rate = np.round(total_false / total_classifications, 4) * 100 # error rate in percentage
    confusion_matrix = np.round(np.array(confusion_matrix) / test_num, 4) * 100 # confusion matrix in percentage

    return error_rate, confusion_matrix, indexes_false_classified, indexes_correct_classified


# Load MNIST dataset
train_data, train_labels, test_data, test_labels = fetch_data()

# Normalization of data
train_data, test_data = train_data / 255, test_data / 255

M = 64 # clusters
test_num = 1000 # chunk of test data

# Create clusters for the training data
clustered_data = create_clusters(train_data, train_labels, M)


############# Task 2 a-b ##############
"""
NN classifier on training data clustered into 64 clusters for each class.
"""

# Results from testing on a chunk of the data
error_rate, confusion_matrix, indexes_false_classified, indexes_correct_classified = test(clustered_data, test_data, test_labels, test_num=10000)
print("Test on all 10000 test images with k=1")
print("Error rate test set:\t", error_rate, "%")
print("Confusion matrix for test set: (True \\ Predicted)")
print(confusion_matrix)
print()
print("--- %s seconds ---" % (time.time() - start_time))
print()

# Print the confusion matrix
df_cm = DataFrame(confusion_matrix, index=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
pretty_plot_confusion_matrix("conf_matrix_NN_clustering.png", df_cm,title='Confusion matrix for NN-classifier with clustering',cmap="YlOrBr",pred_val_axis='x')


############# Task 2 c ##############
"""
KNN classifier on the same clustered data, but with k = 7.
"""

# Results from testing on test images
error_rate, confusion_matrix, indexes_false_classified, indexes_correct_classified = test(clustered_data, test_data, test_labels, test_num=10000, k=7)
print("Test on all 10000 test images with k=7")
print("Error rate test set:\t", error_rate, "%")
print("Confusion matrix for test set: (True \\ Predicted)")
print(confusion_matrix)
print()
print("--- %s seconds ---" % (time.time() - start_time))
print()

df_cm = DataFrame(confusion_matrix, index=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
pretty_plot_confusion_matrix("conf_matrix_KNN_clustering.png", df_cm,title='Confusion matrix for KNN-classifier with clustering and k = 7',cmap="YlOrBr",pred_val_axis='x')


"""
Finding and plotting error rates for different values of k in the k nearest neighbor classifier.
"""
'''
error_rates = []
x = []

for k in range(9):
    x.append(k+1)
    error_rate, confusion_matrix, indexes_false_classified, indexes_correct_classified = test(clustered_data, test_data, test_labels, test_num=10000, k=k+1)
    error_rates.append(error_rate)


plt.title("Error rate as a function of k with weighting of the neighbors")
plt.stem(x, error_rates)
plt.xlabel("Number of nearest neighbors k")
plt.ylabel("Error rate [%]")
plt.grid()
plt.savefig("error_rates_with_weighting_and_normalization.png")
'''