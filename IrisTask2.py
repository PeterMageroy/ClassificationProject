from sklearn import datasets
import matplotlib.pyplot as plt

# Importing iris dataset
iris = datasets.load_iris()

# Divide into classes
setosa = iris.data[:50]
setosa_training = setosa[:30]
setosa_testing = setosa[30:]

versicolor = iris.data[50:100]
versicolor_training = versicolor[:30]
versicolor_testing = versicolor[30:]

virginica = iris.data[100:]
virginica_training = virginica[:30]
virginica_testing = virginica[30:]