from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# how many nearest neighbors are we taking into account
k = 1

# calculate euclidean distance
def dist(x, y):   
    return np.sqrt(np.sum((x-y)**2))

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

"""
# creating an average over all training data for all classes
average_data = []
for i in range(10):
    temp = []
    for m in range(28):
        row = []
        for n in range(28):
            row.append(np.mean([train_X[j][m][n] for j in np.where(train_y == i)]))
        temp.append(row)
    average_data.append(temp)

print(len(average_data))
correct = 0

for m in range(len(test_y)):
    distances = []
    for i in range(len(average_data)):
        distances.append(dist(average_data[i], test_X[m]))
    print("True value:\t", test_y[m], "\tGuess value:\t", np.argmin(distances))
    if test_y[m] == np.argmin(distances):
        correct += 1
print(correct)
"""

for i in range(10):
    plt.imshow(test_X[i])
    plt.colorbar()
    plt.show()