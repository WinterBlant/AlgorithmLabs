import numpy as np
import networkx as ntx
import matplotlib.pyplot as plt
import time


def generate_random_source(nodes):
    rng = np.random.default_rng()
    return rng.integers(nodes)


def generate_adj_matrix(nodes, edges):
    rng = np.random.default_rng()
    mat = np.zeros((nodes, nodes))
    count = 0
    while count < edges:
        i = rng.integers(nodes)
        j = rng.integers(nodes)
        if i != j and mat[i][j] != 1:
            mat[i][j] = rng.integers(1, 1000)
            mat[j][i] = mat[i][j]
            count += 1
    return mat


adj_mat = generate_adj_matrix(100, 500)
G = ntx.Graph(adj_mat)
ntx.draw_networkx(G, pos=ntx.kamada_kawai_layout(G), node_size=50, with_labels=True, font_size=4, width=0.5)
plt.savefig("graph.png", dpi=1000, quality=95)
plt.show()
elapsed_time = []
source = generate_random_source(100)
print(source)
for i in range(10):
    t0 = time.perf_counter()
    path_d = ntx.single_source_dijkstra_path_length(G, source)
    elapsed_time.append(time.perf_counter() - t0)
print(sum(elapsed_time) / 10)
elapsed_time = []
for i in range(10):
    t0 = time.perf_counter()
    path_d = ntx.single_source_bellman_ford_path_length(G, source)
    elapsed_time.append(time.perf_counter() - t0)
print(sum(elapsed_time) / 10)