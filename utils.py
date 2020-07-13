from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_pred, rotate=True):
    """
    y_true: numpy array
    y_pred: numpy array
    """
    test_labels = set(y_true)    
    pred_labels = set(y_pred)
    labels = sorted(list(test_labels | pred_labels)) # union of sets
    cm = confusion_matrix(y_true, y_pred,labels=labels)
    
    sns.set(style="ticks")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    ax.set_xticklabels([''] + labels)
    
    if rotate:
        cm = np.rot90(cm,1) 
        ax.set_yticklabels([''] + labels[::-1])
        cax = ax.matshow(cm,cmap='Greens', interpolation='none')
        ax.xaxis.set_ticks_position('bottom')   
    else:
        ax.set_yticklabels([''] + labels)
        cax = ax.matshow(cm,cmap='Greens', interpolation='none')
    
    fig.colorbar(cax)
    plt.title('Confusion matrix')
    
    # unknown intention
#     ax.xaxis.set_major_locator(plt.MaxNLocator(len(labels)))
#     ax.yaxis.set_major_locator(plt.MaxNLocator(len(labels)))
        
    thresh = cm.max()*0.8    #threshold to switch the color of the text inside the confusion matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # old
#     plt.xlabel('Predicted values')
#     plt.ylabel('True values')
    # new
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.tight_layout()
    plt.show()

