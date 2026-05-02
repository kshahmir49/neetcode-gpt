import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        l = []
        for i,j in zip(y_true,y_pred):
            l.append(i*np.log(j+1e-7) + (1-i)*np.log(1-j+1e-7))
        return np.round((-1/len(y_true))*sum(l),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        l=[]
        for i,j in zip(y_true,y_pred):
            c = []
            for a,b in zip(i,j):
                c.append(a*np.log(b+1e-7))
            l.append(sum(c))
        return np.around((-1/len(y_true))*sum(l),4)
