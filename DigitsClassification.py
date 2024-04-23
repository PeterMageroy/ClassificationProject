import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
    
    

def fetch_data(test_size=10000, standardize=True):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# Load MNIST dataset
train_data, train_labels, test_data, test_labels = fetch_data()


"""
Classifier:
    Go through train_data, find the index of the minimum distance data value, look up index in train_labels.
"""