from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from ConfusionMatrixPrinter import _test_cm, pretty_plot_confusion_matrix
from pandas import DataFrame


iris = datasets.load_iris() # dataset, features can be found in iris.data


C = 3 # number of classes
D = 4 # number of features


# Sigmoid function
def sigmoid(x):
    return (1/(1+np.exp(-x)))


# Function for MSE
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# Function for computing the gradients
def compute_gradient(X, y, y_pred):
    grad_W = -2 * np.dot(X.T, (y - y_pred)) / len(y)
    grad_wo = -2 * np.mean(y - y_pred)
    return grad_W, grad_wo
 

def dataset_split(train_num, test_num, invert_order=False):
    """
    Arranging training and test data from the iris dataset.

    Parameters:
        train_num:      number of training data points
        test_num:       number of test data points
        invert_order:   returns the first data as test and the last 
                        as training data.

    Returns:    training_features, training_labels, test_features, test_labels
    """

    # Define training and test features
    setosa_data = iris.data[0:50]
    versicolour_data = iris.data[50:100]
    virginica_data = iris.data[100:150]

    if invert_order:
        setosa_data = np.flip(setosa_data)
        versicolour_data = np.flip(versicolour_data)
        virginica_data = np.flip(virginica_data)

    if train_num + test_num > 50:
        print("Error: Too many test and training points requested.")

    setosa_training = setosa_data[:train_num]
    setosa_test = setosa_data[train_num:train_num+test_num]

    versicolour_training = versicolour_data[:train_num]
    versicolour_test = versicolour_data[train_num:train_num+test_num]

    virginica_training = virginica_data[:train_num]
    virginica_test = virginica_data[train_num:train_num+test_num]

    training_features = np.concatenate((setosa_training, versicolour_training, virginica_training))
    test_features = np.concatenate((setosa_test, versicolour_test, virginica_test))


    # Define training and test labels
    total_training_samples = len(training_features)
    total_test_samples = len(test_features)

    training_samples = total_training_samples // C
    test_samples = total_test_samples // C

    labels_training = []
    labels_training.extend([[1, 0, 0]] * training_samples)
    labels_training.extend([[0, 1, 0]] * training_samples)
    labels_training.extend([[0, 0, 1]] * training_samples)
    labels_training = np.array(labels_training)

    labels_test = []
    labels_test.extend([[1, 0, 0]] * test_samples)
    labels_test.extend([[0, 1, 0]] * test_samples)
    labels_test.extend([[0, 0, 1]] * test_samples)
    labels_test = np.array(labels_test)

    return training_features, labels_training, test_features, labels_test


def train(training_features, labels_training, learning_rate=0.01, epochs=1000):
    """
    Training to adjust the weights and bias of the linear classifier.

    Parameters:
        training_features:      data for training
        labels_training:        labels to measure the loss
        learning_rate:          coefficient for the gradient descent
        epochs:                 number of iterations

    Returns:    W, w_o
    """

    # Initialize random weights and bias
    W = np.random.randn(C, D)  # weights
    w_o = np.random.randn()  # bias

    # Iterating the training
    for epoch in range(epochs):
        g = sigmoid(np.dot(training_features, W.T) + w_o) # compute predictor
        loss = mse_loss(labels_training, g) # compute loss for the present predictor
        grad_W, grad_wo = compute_gradient(training_features, labels_training, g) # compute the gradients
        
        # Update weights and bias using gradient descent
        W -= learning_rate * grad_W.T
        w_o -= learning_rate * grad_wo
        
        # Print loss every few epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE Loss = {loss}")
    
    return W, w_o


def test(test_features, labels_test, W, w_o):
    """
    Performs a test on the given test data and returns evaluation measures.
    
    Parameters:
        test_features:      test data
        labels_test:        labels used to evaluate performance
        W:                  weights given from training
        w_o:                bias given from training
        
    Returns:    error_rate, confusion_matrix
    """

    total_classifications = len(test_features)
    total_correct = 0
    total_false = 0
    confusion_matrix = [[0 for i in range(3)] for j in range(3)]

    # Iterate through the tests
    for i in range(len(test_features)):
        g = sigmoid(np.dot(test_features[i], W.T) + w_o)

        # Count correct and false classifications
        if np.argmax(labels_test[i]) == np.argmax(g):
            total_correct += 1
        else:
            total_false += 1
        
        confusion_matrix[np.argmax(labels_test[i])][np.argmax(g)] += 1 # add a count to the correct location in confusion matrix

    # In case we count wrong
    if total_correct + total_false != total_classifications:
        print("Error! Number of total classifications incorrect.")

    error_rate = np.round(total_false / total_classifications, 2) * 100 # error rate in percentage
    confusion_matrix = np.round(np.array(confusion_matrix) / len(test_features), 2) * 100 # confusion matrix in percentage

    return error_rate, confusion_matrix




############## Task 1 a ##############

# Split data into features and labels for training and test
training_features, labels_training, test_features, labels_test = dataset_split(30, 20)

# Perform training
W, w_o = train(training_features, labels_training)

# Perform test on test data
error_test, confusion_test = test(test_features, labels_test, W, w_o)
print("Error rate test set:\t", error_test, "%")
print("Confusion matrix for test set: (True \\ Predicted)")
print(confusion_test)
print()

# Perform test on training data
error_train, confusion_train = test(training_features, labels_training, W, w_o)
print("Error rate training set:\t", error_train, "%")
print("Confusion matrix for training set: (True \\ Predicted)")
print(confusion_train)
print()




############## Task 1 d ##############

# Split data into features and labels for training and test
training_features, labels_training, test_features, labels_test = dataset_split(30, 20, invert_order=True)

# Perform training
W, w_o = train(training_features, labels_training)

# Perform test on test data
error_test, confusion_test = test(test_features, labels_test, W, w_o)
print("Error rate test set:\t", error_test, "%")
print("Confusion matrix for test set: (True \\ Predicted)")
print(confusion_test)
print()

# Perform test on training data
error_train, confusion_train = test(training_features, labels_training, W, w_o)
print("Error rate training set:\t", error_train, "%")
print("Confusion matrix for training set: (True \\ Predicted)")
print(confusion_train)
print()


# Print the confusion matrices
"""
df_cm = DataFrame(confusion_train, index=["setosa","versicolour","virginica"], columns=["setosa","versicolour","virginica"])
pretty_plot_confusion_matrix(df_cm,title='Training set',cmap="RdYlGn",pred_val_axis='x')

df_cm = DataFrame(confusion_test, index=["setosa","versicolour","virginica"], columns=["setosa","versicolour","virginica"])
pretty_plot_confusion_matrix(df_cm,title='Test set',cmap="RdYlGn",pred_val_axis='x')
"""

"""
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
ax.set(xlabel=iris.feature_names[2], ylabel=iris.feature_names[3])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()
#plt.savefig("ClassesPetalScatterplot.png")
"""