"""
ECSE4840 EXAM2 Question 3
Support Vector Machine
Author:Yixiang Wang
"""

#import all the libraries needed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Define the data points and their labels
#X = np.array([[1, 1], [2, 1], [2, 2], [4, 5], [6, 5], [5, 7], [7, 7]])
#y = np.array([1, 1, 1, -1, -1, -1, -1])
#for question (b) do the following change:
X = np.array([[1, 1], [2, 1], [2, 2], [4, 5], [6, 5], [5, 7], [7, 7],[5,1]])
y = np.array([1, 1, 1, -1, -1, -1, -1,-1])

# Initialize the model using a linear kernel
model = SVC(kernel='linear', C=1e5)  # Large C for hard-margin SVM

# Fit the model on the provided data
model.fit(X, y)

# Get the support vectors, weights and intercept
support_vectors = model.support_vectors_#get the support vectors
w = model.coef_[0] #get the weight vectors
beta = model.intercept_[0]
margin = 2 / np.linalg.norm(w)

print("support vectors are \n{}".format(support_vectors))
print("weight vectors are {}".format(w))
print("smooth constant beta is {}".format(beta))
print("maximized margin is {}".format(margin))
print("Decision boundary is \n{} x1+{} x2+{}=0".format(w[0],w[1],beta))

# Plot the data points and the support vectors
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, zorder=10, edgecolor='k', s=20)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', s=100, zorder=10, edgecolor='k')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# Get the separating hyperplane
Z = (np.dot(xy, w) + beta).reshape(XX.shape)

# Plot the decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Label the axes and show the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Support Vector Machine (SVM) Decision Boundary')
plt.show()
