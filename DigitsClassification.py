import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


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
    return np.sqrt(np.sum(img2 - img1)**2)


def nearest_neighbor(img1, dataset):
    """
    Finds the index of the dataset that minimizes the distance to img1.

    Parameters:
        img1:       the data to classify
        dataset:    dataset used for training
    
    Returns:    nn_index
    """
    min_distance = np.inf
    nn_index = 0

    for i in range(len(dataset)):
        img = dataset[i]
        temp_distance = distance(img, img1)

        if temp_distance < min_distance:
            min_distance = temp_distance
            nn_index = i

    return nn_index

    



# Load MNIST dataset
train_data, train_labels, test_data, test_labels = fetch_data()






########## Task 1 a) ##########
"""
Use the nearest neighbor predictor to classify the first 1000 data points in 
the test data. Find the confusion matrix and error rate.
"""

# Chunk of data to test
test_num = 1000


def test(train_data, train_labels, test_data, test_labels, test_num=1000):
    # To find error rate and confusion matrix
    total_classifications = test_num
    total_correct = 0
    total_false = 0

    # We want to know what indexes of the test set we were unable to classify correctly, each element as [index of true, predicted value]
    indexes_false_classified = []

    # Confusion matrix is a 10 x 10 matrix
    confusion_matrix = [[0 for i in range(10)] for j in range(10)]


    # Prediction on all the tests
    for i in range(test_num):
        prediction = int(train_labels[nearest_neighbor(test_data[i], train_data)])
        
        if prediction == test_labels[i]:
            total_correct += 1
        else:
            total_false += 1
            indexes_false_classified.append([i, prediction])

        confusion_matrix[int(test_labels[i])][prediction] += 1 # add a count to the correct location in confusion matrix


    # In case we count wrong
    if total_correct + total_false != total_classifications:
        print("Error! Number of total classifications incorrect.")

    error_rate = np.round(total_false / total_classifications, 2) * 100 # error rate in percentage
    confusion_matrix = np.round(np.array(confusion_matrix) / test_num, 2) * 100 # confusion matrix in percentage

    return error_rate, confusion_matrix, indexes_false_classified


# Results from testing on a chunk of the data
error_rate, confusion_matrix, indexes_false_classified = test(train_data, train_labels, test_data, test_labels, test_num)
print("Error rate test set:\t", error_rate, "%")
print("Confusion matrix for test set: (True \\ Predicted)")
print(confusion_matrix)
print("--- %s seconds ---" % (time.time() - start_time))
print()


# Plotting some of the falsely classified images
for i in range(5):
    img = test_data[indexes_false_classified[i, 0]].reshape(28, 28)

    plt.imshow(img, cmap='hot')
    plt.title("True: %s       Prediction: %s" % (test_labels[indexes_false_classified[i, 0]], indexes_false_classified[i, 1]))
    plt.colorbar()
    plt.show()