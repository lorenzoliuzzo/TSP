import numpy as np
import matplotlib.pyplot as plt

def load_tsp(filename):
    with open(filename) as f:
        lines = f.readlines()
        node_coord_start = None
        dimension = None
        
        i = 0
        while not dimension or not node_coord_start:
            line = lines[i]
            if line.startswith('DIMENSION'):
                dimension = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i + 1
            i += 1
        
        xs = []
        ys = []
        
        for i in range(node_coord_start, node_coord_start + dimension):               
            parts = lines[i].split()
            xs.append(float(parts[2]))
            ys.append(float(parts[1]))

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        
        xs = (xs - x_min) / (x_max - x_min)
        ys = (ys - y_min) / (y_max - y_min)
        return np.column_stack([xs, ys])


def distance_matrix(tsp):
    n_cities = tsp.shape[0]    
    D = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = np.linalg.norm(tsp[i] - tsp[j])
            D[i, j] = dist
            D[j, i] = dist
    return D


def path_length(path, D):
    length = 0.0
    for i in range(D.shape[0] - 1):
        length += D[path[i], path[i + 1]]
    length += D[path[-1], path[0]]
    return length


def plot_path(path, length, tsp):
    y = tsp[path]
    plt.plot(y[:, 0], y[:, 1], 'o')
    plt.plot(y[:, 0], y[:, 1], '-', label=f'length={length:.2f}')
    plt.plot([y[-1, 0], y[0, 0]], [y[-1, 1], y[0, 1]], '-', color='C1')
    plt.legend()
    plt.tight_layout()
    plt.show()