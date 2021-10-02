import arxivscraper as ax
import pandas as pd
import networkx as nx
import numpy as np
from numpy.linalg import inv

# scraper = ax.Scraper(category='stat', date_from='2019-03-28',date_until='2019-04-09',t=10)
# scraper = ax.Scraper(category='physics:math-ph', date_from='2019-03-28',date_until='2019-07-01',t=10)
scraper = ax.Scraper(category='stat', date_from='2013-03-28',date_until='2014-07-01',t=10)
output = scraper.scrape()
if output==1:
    print("Records not found!")
cols = ('id', 'title', 'categories',  'authors')
df = pd.DataFrame(output, columns=cols)
df.to_csv('out2.csv')






# initialize Graph
df0 = pd.read_csv('out2.csv',converters={'authors': lambda x: list(x.strip("[]").replace("'","").split(", "))})
df = df0[['authors']]
G=nx.Graph()
row_num=len(df.index)



# create nodes and edges (from record list)
node_dict_ind={} # name: node_index
node_dict={}  # node_index: name
edgelist = []
N_max=350
i=0
for r in range(row_num):
    # ensure there are at most 500 nodes
    if i > N_max:
        break
    authors = df.at[r, 'authors']
    for author in authors:
        if author not in node_dict_ind.keys():
            node_dict_ind[author]=i
            node_dict[i]=author
            i+=1
    for ki in range(len(authors)):
        for kj in range(ki+1, len(authors)):
            edgelist.append((node_dict_ind[authors[ki]], node_dict_ind[authors[kj]]))
G.add_nodes_from(node_dict.keys())
print("The number of nodes:", G.number_of_nodes())
G.add_edges_from(edgelist)
print("The number of edges:", G.number_of_edges())





# Adjacency Matrix & Degree
A = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
D = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
print("Adjacency list:", G.adj)
for i in G.adj.keys():
    for j in G.adj[i].keys():
        A[i,j]=1
print("Adjacency Matrix=\n", A)
print("Node degrees", dict(G.degree))
for i in G.nodes:
    D[i,i] = G.degree[i]
print("Degree Matrix=\n", D)






########### Visualize Graph
import matplotlib.pyplot as plt
plt.figure(1)
nx.draw_kamada_kawai(G, node_color='b', edge_color='r', node_size=[G.degree[i]*10+1 for i in G.nodes], edge_size=0.1)





########## clustering coefficient  #############
print("Global Clustering Coefficient:\n", nx.transitivity(G))
print("Local Clustering Coefficient:\n", nx.clustering(G))





###### PageRank (their are zero-degree nodes) ######
# using equations in the book
alpha=0.95
beta=0.1
N=G.number_of_nodes()
AinvD = np.dot(A, inv(D+ np.eye(N) * 0.000001))
centrality = beta * \
             np.dot(inv(np.eye(N) - alpha * AinvD),
                    np.ones(N))
print("\nCentrality vector", {i:centrality[i] for i in range(N)})
top10_indices = np.argsort(centrality)[-10:][::-1]
print("top10 rank nodes/authors=\n",[node_dict[index] for index in top10_indices])
print("top-10 PageRank scores (centrality values):\n", centrality[top10_indices])
# calculate by networkx (same results)
pr = nx.pagerank(G, alpha=0.95)
pr=np.array([pr[i] for i in pr.keys()])
top10_indices = np.argsort(pr)[-10:][::-1]
print("Networkx - top10 rank nodes/authors=\n",[node_dict[index] for index in top10_indices])
print("Networkx - top-10 PageRank scores:\n", pr[top10_indices])




# degree distribution
plt.figure(2)
def draw_degree_distribution(degree_dict):
    degrees=[degree_dict[i] for i in degree_dict.keys()]
    plt.hist(degrees)
    plt.xlabel('Node degree')
    plt.title('Degree Distribution')
draw_degree_distribution(dict(G.degree))
plt.show()







########## diameter  #############
print("\n")
if not nx.is_connected(G):
    print("The graph is not connected, so the diameter=infinity")
    largest_cc = max(nx.connected_components(G), key=len)
    print("Graph Diameter of the largest component of G:\n", nx.diameter(G.subgraph(largest_cc).copy()))
else:
    print("Graph Diameter:\n", nx.diameter(G))