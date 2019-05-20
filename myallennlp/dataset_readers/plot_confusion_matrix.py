"""
================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm1 = np.array([[0,345,401,225],[324,0,106,22],[405,101,0,62],[238,47,101,0]])
    cm2 = np.array([[0,40,65,24],[26,0,22,4],[42,12,0,8],[24,7,8,0]])


    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    im = ax1.imshow(cm1, interpolation='nearest', cmap=cmap)
    # We want to show all ticks...
    ax1.set(xticks=np.arange(cm1.shape[1]),
           yticks=np.arange(cm1.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=["Null","A0","A1","A2"], yticklabels=["Null","A0","A1","A2"],
           title="Baseline Confusion Matrix",
           ylabel='True role',
           xlabel='Baseline predicted role')

    im2 = ax2.imshow(cm2, interpolation='nearest', cmap=cmap)


    ax2.set(xticks=np.arange(cm2.shape[1]),
           yticks=np.arange(cm2.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=["Null","A0","A1","A2"], yticklabels=["Null","A0","A1","A2"],
           title="Error Correction Matrix",
           ylabel='Corrected role',
           xlabel='Baseline predicated role')

    # Rotate the tick labels and set their alignment.
  #  plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            ax1.text(j, i, format(cm1[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm1[i, j] > thresh else "black")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm2.max() / 2.
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            ax2.text(j, i, format(cm2[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm2[i, j] > thresh else "black")
    fig.tight_layout()
    return ax1


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')


plt.show()
