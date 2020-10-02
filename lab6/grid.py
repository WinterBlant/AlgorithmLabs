import numpy as np
import matplotlib.pyplot as plt
import networkx as ntx


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


grid = np.array([[1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                 [0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                 [1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
G = ntx.grid_2d_graph(10, 10)
obstacles = np.argwhere(grid).tolist()
for (u, v) in G.edges():
    if list(v) in obstacles or list(u) in obstacles:
        G.edges[u, v]['weight'] = 100000
        G.edges[u, v]['color'] = 'white'
    else:
        G.edges[u, v]['weight'] = 0
        G.edges[u, v]['color'] = 'black'
pos = dict((n, n) for n in G.nodes())
color_map = []
for node in G:
    if list(node) in obstacles:
        color_map.append('#FF0000')
    else:
        color_map.append('#32CD32')
colors = [G[u][v]['color'] for u, v in G.edges()]
ntx.draw(G, pos, node_size=500,  with_labels=True, node_color=color_map, width=1, edge_color=colors, font_size=9)
plt.savefig("grid.png", dpi=1000, quality=95)
plt.show()
start = (0, 5)
end = (7, 1)
shortest_path = ntx.astar_path(G, start, end, dist)
print(shortest_path)
start = (4, 2)
end = (8, 9)
shortest_path = ntx.astar_path(G, start, end, dist)
print(shortest_path)
start = (1, 3)
end = (2, 4)
shortest_path = ntx.astar_path(G, start, end, dist)
print(shortest_path)
start = (9, 0)
end = (0, 9)
shortest_path = ntx.astar_path(G, start, end, dist)
print(shortest_path)
start = (8, 3)
end = (1, 6)
shortest_path = ntx.astar_path(G, start, end, dist)
print(shortest_path)