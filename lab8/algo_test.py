import numpy as np
import networkx as ntx
import matplotlib.pyplot as plt
import time


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


def test(adj_mat, elapsed_time, i):
    G = ntx.Graph(adj_mat)
    ntx.draw_networkx(G, pos=ntx.kamada_kawai_layout(G), node_size=50, with_labels=True, font_size=4, width=0.5)
    plt.savefig("graph"+i+".png", dpi=1000, quality=95)
    plt.show()
    t0 = time.perf_counter()
    ntx.floyd_warshall(G)
    elapsed_time[0].append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    ntx.johnson(G)
    elapsed_time[1].append(time.perf_counter() - t0)
    print("finished_test")
    return elapsed_time


elapsed_time = []
for i in range(2):
    elapsed_time.append([])
adj_mat = generate_adj_matrix(100, 500)
elapsed_time = test(adj_mat, elapsed_time, 1)
adj_mat = generate_adj_matrix(100, 4800)
elapsed_time = test(adj_mat, elapsed_time, 2)
adj_mat = generate_adj_matrix(300, 500)
elapsed_time = test(adj_mat, elapsed_time, 3)
adj_mat = generate_adj_matrix(100, 2500)
elapsed_time = test(adj_mat, elapsed_time, 4)
adj_mat = generate_adj_matrix(300, 43000)
elapsed_time = test(adj_mat, elapsed_time, 5)
print(elapsed_time)