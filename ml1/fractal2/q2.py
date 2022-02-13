import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


data = pd.read_csv("../dataset/iris.data",sep = ',',header=None)
data[4] = data[4].replace({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
x=data.iloc[:,0:4]
y=data.iloc[:,4]

class PCA:
    def __init__(self):
        self.n_components = 2
        self.components = None
        self.mean = None

    def fit(self, X):
        # compute mean value
        self.mean = np.mean(X, axis=0)
        """
        1. subtract each with mean and transpose
        2. apply coverity and compute eign values and vectors
        3. apply sort of eign values and filter the eign vectors from transpose with sorted eign values
        """
        eigen_values, eigen_vectors = np.linalg.eig(np.cov((X - self.mean).T))
        sorted_eign_values = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_eign_values]
        eigen_vectors = eigen_vectors.T[sorted_eign_values]
        self.components = eigen_vectors[0:self.n_components]

    def transform(self, X):
        return np.dot(X - self.mean, self.components.T)
    
pca = PCA()
pca.fit(x)
X_predicted = pca.transform(x)
print(X_predicted[0:10,:])
x1 = X_predicted[:, 0]
x2 = X_predicted[:, 1]


plt.scatter(x1, x2,c=y, edgecolor='green',alpha=0.8,cmap=plt.cm.get_cmap('viridis', 3))
plt.xlabel('Principal Component X')
plt.ylabel('Principal Component Y')
plt.colorbar()
plt.show()