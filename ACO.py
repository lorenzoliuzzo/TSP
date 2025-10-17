import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple

import utils


@dataclass
class Ant:
    alpha: float
    beta: float
        
        
def tour(ant: Ant, seed: int, pheromones: np.ndarray, distances: np.ndarray) -> np.array:   
    """
    Perform a tour (path through all cities) using probabilistic transition rule.

    Args:
        ant (Ant): Ant instance containing alpha and beta parameters.
        seed (int): Random seed for reproducibility.
        pheromones (np.ndarray): NxN pheromone matrix.
        distances (np.ndarray): NxN distance matrix.

    Returns:
        np.array: A permutation of cities representing the tour.
    """
    rng = np.random.default_rng(seed)

    n_cities = distances.shape[0]
    path = np.zeros(n_cities, dtype=int)

    # Start at a random city
    start = rng.integers(n_cities)
    path[0] = start

    # Track visited cities
    visited_mask = np.zeros(n_cities, dtype=bool)        
    visited_mask[start] = True

    last = start
    for i in range(1, n_cities):
        unvisited = np.where(~visited_mask)[0]
        trails = pheromones[last, unvisited] 
        deltas = distances[last, unvisited]

        # Transition probabilities
        probs = (trails ** ant.alpha) / (deltas ** ant.beta)
        probs /= np.sum(probs)

        current = rng.choice(unvisited, p=probs)
        path[i] = current
        visited_mask[current] = True
        last = current

    return path


class AntColony:
    ants: List[Ant]
    pheromones: np.ndarray
    rho: float
    
    paths: np.ndarray
    lengths: np.array
    
    best_path: np.array
    best_length: float
    
    
    def __init__(self, n_cities, n_ants, rho) -> None:
        self.ants = [Ant(alpha=1, beta=2) for _ in range(n_ants)]
        self.rho = rho
        self.paths = np.zeros((n_ants, n_cities), dtype=int)
        self.lengths = np.zeros(n_ants, dtype=float)
        

    def reset(self) -> None:
        """Reset pheromones and best solution for a new run."""
        self.pheromones = np.ones((n_cities, n_cities), dtype=float)
        np.fill_diagonal(self.pheromones, 0)
        self.best_path = np.zeros(n_cities, dtype=int)
        self.best_length = float('inf')
                
            
    def update_best(self, path, length) -> None:
        """Update the best known path if a better one is found."""
        if length < self.best_length:
            self.best_path = path
            self.best_length = length
            
            
    def explore(self, seed, D) -> None:
        """
        Perform a sequential exploration: all ants perform tours.
        
        Args:
            seed (int): Random seed for reproducibility.
            D (np.ndarray): NxN distance matrix.
            
        Returns: None
        """
        rng = np.random.default_rng(seed)
        for i, ant in enumerate(self.ants):
            seed = rng.integers(2**32 - 1)
            path = tour(ant, seed, self.pheromones, D)
            length = utils.path_length(path, D)
            
            self.paths[i] = path
            self.lengths[i] = length
            self.update_best(path, length)
        
        
    def explore_parallel(self, seed, n_workers, D) -> None:
        """
        All ants in the colony perform a tour (path through all cities) using probabilistic transition rule, in parallel.
        
        Args:
            seed (int): Random seed for reproducibility.
            n_workers (int): Number of parallel processes.            
            D (np.ndarray): NxN distance matrix.
            
        Returns: None
        """
        rng = np.random.default_rng(seed)
        n_ants = len(self.ants)
        seeds = rng.integers(2**32 - 1, size=n_ants)
        tasks = [(self.ants[i], seeds[i], self.pheromones, distances) for i in range(n_ants)]
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            tours = pool.starmap(tour, tasks)
            
        for i, path in enumerate(tours):
            length = utils.path_length(path, D)
            self.paths[i] = path            
            self.lengths[i] = length            
            self.update_best(path, length)
        
        
    def update_pheromones(self) -> None:   
        """Evaporate pheromones and deposit new pheromones based on current ant paths."""

        self.pheromones *= (1 - self.rho)

        n_ants, n_cities = self.paths.shape
        for i in range(n_ants):
            path = self.paths[i]
            incr = 1 / self.lengths[i]
            for k in range(n_cities - 1):
                a, b = path[k], path[k + 1]
                self.pheromones[a, b] += incr
                self.pheromones[b, a] += incr  

            start, end = path[0], path[-1]
            self.pheromones[start, end] += incr
            self.pheromones[end, start] += incr

            
    def run(self, seed, n_iters, D) -> None:
        """
        Run the ACO algorithm sequentially for a number of iterations.

        Args:
            seed (int): Random seed for reproducibility.
            n_iters (int): Number of iterations.
            D (np.ndarray): NxN distance matrix.
        """
        rng = np.random.default_rng(seed)
        seeds = rng.integers(2**32 - 1, size=n_iters)
                
        self.reset()
        for seed in tqdm(seeds):
            self.explore(seed, D)
            self.update_pheromones()    
    
    
    def run_parallel(self, seed, n_workers, n_iters, D) -> None:
        """
        Run the ACO algorithm using parallel exploration.

        Args:
            seed (int): Random seed for reproducibility.
            n_workers (int): Number of parallel worker processes.
            n_iters (int): Number of iterations.
            D (np.ndarray): NxN distance matrix.
        """
        rng = np.random.default_rng(seed)
        seeds = rng.integers(2**32 - 1, size=n_iters)
        
        self.reset()
        for seed in tqdm(seeds):
            self.explore_parallel(seed, n_workers, D)
            self.update_pheromones()
            
            

if __name__ == '__main__':
    n_cities = 100
    n_ants = 200
    n_iters = 100
    
    print(f"Running ACO with {n_ants} ants for {n_iters} iterations to solve a TSP with {n_cities} cities.")
    
    tsp = utils.load_tsp('data/it16862.tsp')
    x = tsp[np.random.choice(tsp.shape[0], n_cities)]
    D = utils.distance_matrix(x)
    
    rng = np.random.default_rng()
    colony = AntColony(n_cities, n_ants, 0.4)
    
    seed = rng.integers(2**32 - 1)
    colony.run(seed, n_iters, D)
    
    print(f"Best path found has length of {colony.best_length}.")

    n_workers = 4
    
    print(f"Running ACO with {n_ants} ants on {n_workers} cores for {n_iters} iterations to solve a TSP with {n_cities} cities.")

    seed = rng.integers(2**32 - 1)
    colony.run_parallel(seed, n_workers, n_iters, D)

    print(f"Best path found has length of {colony.best_length}.")