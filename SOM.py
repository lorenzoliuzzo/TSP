import numpy as np
from tqdm import tqdm


def get_closest(points, origin):
    distances = np.linalg.norm(points - origin, axis=1)
    return np.argmin(distances)


def get_neighborhood(points, origin, r):
    closest_idx = get_closest(points, origin)
    domain = points.shape[0]
    deltas = np.abs(closest_idx - np.arange(domain))
    distances = np.minimum(deltas, domain - deltas)
    return np.exp(-0.5 * distances**2 / r**2)


def solve(
    tsp, 
    iters=10000, 
    network_rel_size=3, 
    radius_rel_size=0.1, 
    lr=0.1, 
    radius_decay=0.9997, 
    lr_decay=0.99997,
):
    
    cities = tsp.copy()
    n_cities = cities.shape[0]
    
    n = network_rel_size * n_cities
    network = np.random.rand(n, 2)
    radius = radius_rel_size * n
    
    for i in tqdm(range(iters)):
        idx = np.random.choice(n_cities)
        city = cities[idx]
        gaussian = get_neighborhood(network, city, radius)
        network += gaussian[:, np.newaxis] * lr * (city - network)

        lr *= lr_decay
        radius *= radius_decay

        if radius < 1:
            print(f'Radius has completely decayed, finishing execution at {i} iterations')
            break
        if lr < 0.001:
            print(f'Learning rate has completely decayed, finishing execution at {i} iterations')
            break
            
    return network