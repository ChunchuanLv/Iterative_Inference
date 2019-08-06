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
import matplotlib
font = {'family' : "Times New Roman",
        'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)

if __name__ == "__main__":

    cmap=plt.cm.Blues
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
    fmt =  'd'
    thresh = cm1.max() / 2.
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            ax1.text(j, i, format(cm1[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm1[i, j] > thresh else "black")

    # Loop over data dimensions and create text annotations.
    fmt =  'd'
    thresh = cm2.max() / 2.
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            ax2.text(j, i, format(cm2[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm2[i, j] > thresh else "black")
    fig.tight_layout()
   # fig.savefig("matrix.pdf", bbox_inches='tight')


    np.set_printoptions(precision=2)



    plt.show()
