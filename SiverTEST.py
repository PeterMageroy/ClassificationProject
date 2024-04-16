import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import struct
from keras.datasets import mnist



class KNN:
    def __init__(self, k=1):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [np.sqrt(np.sum(x_train - x)**2) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Importing data   
(train_X, train_y), (test_X, test_y) = mnist.load_data()
# Reshaping data
X_train = train_X.reshape(60000, 784)
X_test = test_X.reshape(10000, 784)
y_train = train_y
y_test = test_y

training_data = 60000
testing_data = 100 

clf = KNN()
clf.fit(X_train[:training_data,:], y_train[:training_data])
predictions = clf.predict(X_test[:testing_data,:])
accuracy = np.sum(predictions == y_test[:testing_data]) / len(y_test[:testing_data])
print(accuracy)
