import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from ConfusionMatrixPrinter import _test_cm, pretty_plot_confusion_matrix
from pandas import DataFrame

cm = np.array([[84, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
               [0, 125, 0, 0, 0, 0, 1, 0, 0, 0], 
               [1, 3, 103, 2, 1, 0, 0, 3, 3, 0], 
               [0, 0, 0, 102, 0, 1, 0, 1, 1, 2], 
               [0, 1, 1, 0, 101, 0, 1, 0, 1, 5], 
               [0, 0, 0, 3, 1, 78, 0, 0, 3, 2], 
               [2, 0, 0, 0, 0, 0, 85, 0, 0, 0], 
               [0, 0, 2, 1, 0, 0, 0, 94, 0, 2], 
               [2, 0, 1, 4, 0, 2, 0, 0, 79, 1], 
               [0, 0, 0, 1, 2, 0, 0, 2, 2, 87]])

df_cm = DataFrame(cm, index=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
pretty_plot_confusion_matrix("conf_matrix_NN_TEST.png", df_cm,title='Confusion matrix for NN-classifier',pred_val_axis='x', fz=8, cmap='icefire',show_null_values=2)