from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


C = 3 # number of classes
D = 4 # number of features


iris = datasets.load_iris() # dataset, features in iris.data


# Define training and test data
setosa_training = iris.data[:30]
setosa_test = iris.data[30:50]

versicolor_training = iris.data[50:80]
versicolor_test = iris.data[80:100]

virginica_training = iris.data[100:130]
virginica_test = iris.data[130:150]

training_features = np.concatenate((setosa_training, versicolor_training, virginica_training))


# Define labels for training
total_samples = len(training_features)
samples_per_class = total_samples // C

labels_training = []

labels_training.extend([[1, 0, 0]] * samples_per_class)
labels_training.extend([[0, 1, 0]] * samples_per_class)
labels_training.extend([[0, 0, 1]] * samples_per_class)

labels = np.array(labels_training)



# Initialize weights and bias
W = np.random.randn(C, D)  # weights
w_o = np.random.randn()  # bias


# Function for MSE
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# Function for computing the gradients
def compute_gradient(X, y, y_pred):
    # Compute gradients
    grad_W = -2 * np.dot(X.T, (y - y_pred)) / len(y)
    grad_wo = -2 * np.mean(y - y_pred)
    return grad_W, grad_wo


# Training parameters
learning_rate = 0.01
epochs = 1000


# Training
for epoch in range(epochs):
    g = np.dot(training_features, W.T) + w_o # compute predictor
    loss = mse_loss(labels_training, g) # compute loss for present predictor
    grad_W, grad_wo = compute_gradient(training_features, labels_training, g) # compute the gradients
    
    # Update weights and bias using gradient descent
    W -= learning_rate * grad_W.T
    w_o -= learning_rate * grad_wo
    
    # Print loss every few epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: MSE Loss = {loss}")




"""
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
ax.set(xlabel=iris.feature_names[2], ylabel=iris.feature_names[3])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

plt.savefig("ClassesPetalScatterplot.png")
"""
