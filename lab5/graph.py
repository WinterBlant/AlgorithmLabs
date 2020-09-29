import numpy as np
import networkx as ntx

def generate_random_point():
    rng = np.random.default_rng()
    return rng.integers(1, endpoint=True)


a = np.zeros((100, 100))