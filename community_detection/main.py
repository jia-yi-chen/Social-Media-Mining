"""
Created on April 2021
author: Jiayi Chen
"""
import numpy as np
import matplotlib.pyplot as plt
import spectral_clustering
import modularity_maximization



def read_adj(f):
    """
    Load adjacency matrix
    f: file name
    """
    with open(f, 'r') as f:
        F = f.readlines()
    mode=""
    for line in F:
        a = line.split()
        if mode=="read_edges":
            i = int(a[0])-1
            j = int(a[1])-1
            adjacency[i,j] = 1
            adjacency[j,i] = 1
        if a[0]=="*Vertices":
            mode="read_nodes"
            N=int(a[1])
        if a[0]=="*Edges":
            mode="read_edges"
            adjacency = np.zeros((N, N))
    return adjacency




#### read in Adjacency matrix
A = read_adj("karate/karate.paj")
N = A.shape[0]




#### ground_truth community assignments
ground_truth = np.zeros(N).astype(int)
cluster0 = "Mr. Hi's"
cluster1 = "Officers'"
cluster1_nodes = [9, 14, 15, 18, 20]; cluster1_nodes.extend(range(22, 34))
ground_truth[cluster1_nodes] = 1
print("\n\n===== Ground-truth Labels ===== ")
print("Comminities:", ground_truth)
print("   Community 1:", np.where(ground_truth == 0)[0])
print("   Community 2:", np.where(ground_truth == 1)[0])




#### Run the spectral clustering algorithm
print("\n\n===== Spectral Clustering (k=2) ===== ")
k=2
community_assignments, fs = spectral_clustering.run(A, k, mode="normalized")
F_measure = spectral_clustering.eval_F(community_assignments, ground_truth)
# report results
print("Comminities:", community_assignments)
print("   Community 1:", np.where(community_assignments == 0)[0])
print("   Community 2:", np.where(community_assignments == 1)[0])
print("F-measure =", F_measure)
# plot





#### Run the modularity_maximization algorithm
print("\n\n===== Modularity Maximization (k=2) ===== ")
k=2
community_assignments, fm = modularity_maximization.run(A, k)
F_measure = modularity_maximization.eval_F(community_assignments, ground_truth)
# report results
print("Comminities:", community_assignments)
print("   Community 1:", np.where(community_assignments == 0)[0])
print("   Community 2:", np.where(community_assignments == 1)[0])
print("F-measure =", F_measure)





###### plot clusters ########
# Spectral
cluster0 = fs[0][fs[2] == 0]
cluster1 = fs[0][fs[2] == 1]
plt.scatter(cluster0[:, 0], np.zeros(cluster0.shape[0]), marker='o', color='#39C8C6')
plt.scatter(cluster1[:, 0], np.zeros(cluster1.shape[0]), marker='o', color='#D3500C')
plt.plot(fs[1][0], 0, marker='o', color='#39C8C6')
plt.plot(fs[1][1], 0, marker='o', color='#D3500C')
plt.title("Spectral Clustering (k=2) clusters ")
# Spectral shown as ground truth
plt.figure()
c0 = np.where(fs[2] == 0)[0]
c1 = np.where(fs[2] == 1)[0]
h0_l0 = np.where(ground_truth[c0] == 0)[0]  # correct/wrong prediction
h0_l1 = np.where(ground_truth[c0] == 1)[0]  # wrong/correct prediction
h1_l0 = np.where(ground_truth[c1] == 0)[0]  # wrong/correct prediction
h1_l1 = np.where(ground_truth[c1] == 1)[0]  # correct/wrong prediction
plt.scatter(cluster0[h0_l0, 0], np.zeros(h0_l0.shape[0]), marker='x', color='#39C8C6')
plt.scatter(cluster0[h0_l1, 0], np.zeros(h0_l1.shape[0]), marker='+', color='#D3500C')
plt.scatter(cluster1[h1_l0, 0], np.zeros(h1_l0.shape[0]), marker='x', color='#39C8C6')
plt.scatter(cluster1[h1_l1, 0], np.zeros(h1_l1.shape[0]), marker='+', color='#D3500C')
# plt.plot(fs[1][0], 0, marker='o', color='#39C8C6')
# plt.plot(fs[1][1], 0, marker='o', color='#D3500C')
plt.title("Spectral Clustering (k=2) shown by ground truth labels")
# Modularity
plt.figure()
cluster0 = fm[0][fm[2] == 0]
cluster1 = fm[0][fm[2] == 1]
plt.scatter(cluster0[:, 0], cluster0[:, 1], marker='o', color='#39C8C6')
plt.scatter(cluster1[:, 0], cluster1[:, 1], marker='o', color='#D3500C')
plt.plot(fm[1][0, 0], fm[1][0, 1], marker='o', color='#39C8C6')
plt.plot(fm[1][1, 0], fm[1][1, 1], marker='o', color='#D3500C')
plt.title("Modularity Maximization (k=2) clusters")
# Modularity shown as ground truth
plt.figure()
c0 = np.where(fm[2] == 0)[0]
c1 = np.where(fm[2] == 1)[0]
h0_l0 = np.where(ground_truth[c0] == 0)[0]  # correct/wrong prediction
h0_l1 = np.where(ground_truth[c0] == 1)[0]  # wrong/correct prediction
h1_l0 = np.where(ground_truth[c1] == 0)[0]  # wrong/correct prediction
h1_l1 = np.where(ground_truth[c1] == 1)[0]  # correct/wrong prediction
plt.scatter(cluster0[h0_l0, 0], cluster0[h0_l0, 1], marker='x', color='#39C8C6')
plt.scatter(cluster0[h0_l1, 0], cluster0[h0_l1, 1], marker='+', color='#D3500C')
plt.scatter(cluster1[h1_l0, 0], cluster1[h1_l0, 1], marker='x', color='#39C8C6')
plt.scatter(cluster1[h1_l1, 0], cluster1[h1_l1, 1], marker='+', color='#D3500C')
plt.title("Modularity Maximization (k=2) shown by ground truth labels")
plt.show()

