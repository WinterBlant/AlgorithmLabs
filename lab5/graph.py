import numpy as np
import networkx as ntx
import matplotlib.pyplot as plt


def bfs_shortest_path(graph, start, goal, visited):
    queue = [[start]]
    if start == goal:
        return start
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if not visited[node]:
            neighbours = graph[node]
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                if neighbour == goal:
                    return new_path
            visited[node] = True
    return "No path exists"


def dfs(graph, node, visited, component):
    component.append(node)
    visited[node] = True
    for i in range(0, len(graph[node])):
        if not visited[graph[node][i]]:
            dfs(graph, graph[node][i], visited, component)


def connected_components(graph, visited):
    cc = []
    j = 0
    for i in range(0, len(visited)):
        if not visited[i]:
            cc.append([])
            dfs(graph, i, visited, cc[j])
            j += 1
    return cc


def generate_adj_matrix(nodes, edges):
    rng = np.random.default_rng()
    mat = np.zeros((nodes, nodes))
    count = 0
    while count < edges:
        i = rng.integers(nodes)
        j = rng.integers(nodes)
        if i != j and mat[i][j] != 1:
            mat[i][j] = 1
            mat[j][i] = 1
            count += 1
    return mat


adj_mat = generate_adj_matrix(100, 200)
G = ntx.Graph(adj_mat)
adj_list = ntx.to_dict_of_lists(G)
ntx.draw_networkx(G, pos=ntx.kamada_kawai_layout(G), node_size=50, with_labels=True, font_size=4, width=0.5)
plt.savefig("graph.png", dpi=1000, quality=95)
plt.show()
visited = np.full(100, False)
cc = connected_components(adj_list, visited)
visited = np.full(100, False)
spath = bfs_shortest_path(adj_list, 2, 56, visited)
print(cc)
print(spath)
print(adj_mat[0:3])
print(adj_list)
