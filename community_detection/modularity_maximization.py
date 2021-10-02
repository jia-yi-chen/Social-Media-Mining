import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans, whiten
from itertools import combinations


def run(A,k):

    """
    A - Adjacecny matrix
    k - cluster num
    """
    num_node = A.shape[0]
    d = np.sum(A, axis=0)
    num_edge = int(np.sum(d, axis=0)/2)

    ### Calculate Modularity matrix
    B = A - np.outer(d, d.T) / (2*num_edge)

    ### obtain the eigenvalues and eigenvectors of the laplacian
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    ### sort the eigenvalues (from largest to smallest)
    sorted_eigenvalues = np.argsort(eigenvalues)[::-1]

    ### obtain top-k largest eigenvalues
    indices = sorted_eigenvalues[:k]

    ### obtain corresponding top eigenvectors => features
    # rows: different nodes; columns: different features/the (k-1)eigenvectors
    X = np.zeros((num_node, k))
    for j, index in enumerate(indices):
        X[:,j]=eigenvectors[:, index]

    ### use k-means to group all nodes into k clusters
    X = whiten(X)  # Normalize each feature space (columns of X)
    centroids,_ = kmeans(X, k)
    cluster_assign, _ = vq(X, centroids)

    ### plot clusters
    plot_obj = [X, centroids, cluster_assign]

    return cluster_assign, plot_obj






def eval_F(predictions, ground_truth):

    cluster0 = np.where(predictions == 0)[0]
    cluster1 = np.where(predictions == 1)[0]
    h0_l0 = np.where(ground_truth[cluster0] == 0)[0] # correct/wrong prediction
    h0_l1 = np.where(ground_truth[cluster0] == 1)[0] # wrong/correct prediction
    h1_l0 = np.where(ground_truth[cluster1] == 0)[0] # wrong/correct prediction
    h1_l1 = np.where(ground_truth[cluster1] == 1)[0] # correct/wrong prediction

    ### True Positive
    TP = 0
    for hili in [h0_l0, h0_l1, h1_l0, h1_l1]:
        if len(hili)>=2:
            comb = list(combinations(hili, 2))
            TP += len(comb)

    ### False Positive
    FP = len(h0_l0) * len(h1_l0) + len(h0_l1) * len(h1_l1)

    ### False_Negative
    FN = len(h0_l0) * len(h0_l1) + len(h1_l0) * len(h1_l1)

    ### precision, recall, and F
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F = 2 * precision * recall / (precision + recall)

    return F