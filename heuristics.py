import os
# pip install argparse
import argparse
# pip install numpy
import numpy as np
# pip install tqdm
from tqdm import tqdm
from utility_functions import *

class GreedyAlgorithm():

    """ Greedy Algorithm to solve the Travelling Salesman Problem (TSP). """

    def __init__(self):

        """
            path: Point indexes in the order they are visited
            costs: Costs of all trips between two consecutive points along the path
            total_cost: Total cost of path found (sum of all costs)
        """

        self.path = list()
        self.costs = list()
        self.total_cost = None
        

    def optimize(self, distance_matrix, cycle = True, verbose = True):

        """
            This optimization algorithm always makes the choice that is best at that moment (greedy).
            It begins by selecting randomly the starting point and then selects the edge with the minimum cost. 
            It continually selects the next best choices until it visits all points.

            distance_matrix: Square and symmetric matrix containing the distances, taken pairwise, between the points
            cycle: True if the traveling salesman must return to the starting point
            verbose: Print progress bar

        """
    
        n, _ = distance_matrix.shape

        start_point = np.random.choice(range(0, n))
        self.path.append(start_point)
        mask = np.zeros((n,), dtype = bool)
        mask[start_point] = True

        # Progress bar
        if verbose is True:
            pbar = tqdm(total = n, desc = 'Iterations')

        for i in range(n-1):
            
            wi = distance_matrix[self.path[i],:]
            next_point = np.ma.masked_array(wi, mask).argmin()
            self.path.append(next_point)
            self.costs.append(distance_matrix[self.path[i], next_point])
            mask[next_point] = True
            if verbose is True:
                # Increasing number of iterations
                pbar.update(1)

        if cycle is True:
            # Returning to the starting point (+1 for city indexes)
            self.costs.append(distance_matrix[self.path[-1], start_point])
            self.path.append(start_point)

        self.path = np.array(self.path) + 1
        if verbose is True:
            pbar.update(1)

        self.total_cost = sum(self.costs)
        if verbose is True:
            pbar.close()
            
class VariableNeighborhoodDescent():

    """ Variable Neighborhood Descent to solve the Travelling Salesman Problem (TSP). """

    def __init__(self):

        """
            path: Point indexes in the order they are visited
            costs: Costs of all trips between two consecutive points along the path
            total_cost: Total cost of path found (sum of all costs)
            niter: Number of iterations of the local search
        """

        self.path = list()
        self.costs = list()
        self.total_cost = None
        self.niter = 200000

    def initialization(self, distance_matrix, mode, cycle = True):

        """
            Creates an initial route.

            distance_matrix: Square and symmetric matrix containing the distances, taken pairwise, between the points
            mode: Strategy to create the initial solution (random or greedy)
            cycle: True if the traveling salesman must return to the starting point
        """

        if mode == 'random':
            
            n = distance_matrix.shape[0]
            path = np.random.choice(np.arange(1, n + 1), size = n, replace = False)

            if cycle is True:
                path = np.append(path, path[0])

        elif mode == 'greedy':

            GA = GreedyAlgorithm()
            GA.optimize(distance_matrix, cycle, verbose = False)
            path = GA.path

        return path

    def swap(self, path):

        """
            Swaps the position of two points on the route.

            path: Point indexes in the order they are visited
        """

        n = len(path)
        path_ = path.copy()

        i = np.random.randint(1, n-1)
        j = np.random.randint(1, n-1)
    
        path[i] = path_[j]
        path[j] = path_[i]

        return path

    def two_opt(self, path):

        """
            2-opt movement take two arcs from the route and reconnect these arcs with each other.

            path: Point indexes in the order they are visited
        """

        n = len(path)
        idx = np.random.randint(1, n-1, size = 2)
        i, j = sorted(idx)
    
        head = path[0:i]
        neck = path[j:-n+i-1:-1]
        tail = path[j+1:n]

        path = np.concatenate((head, neck, tail))

        return path
    
    def local_search(self, path, total_cost, k, distance_matrix):

        """
            Local search starts from an initial solution and evolves that single solution into a mostly better and better solution through neighborhood structures.

            path: Point indexes in the order they are visited
            total_cost: Total cost of path found (sum of all costs) 
            k: Current neighborhood structure
            distance_matrix: Square and symmetric matrix containing the distances, taken pairwise, between the points
        """

        i = 0
        while i < self.niter:
        
            if k == 0:
                path_ = self.two_opt(path)
            elif k == 1:
                path_ = self.swap(path)

            total_cost_ = evaluate_path(path_, distance_matrix)

            if(total_cost_ < total_cost):
                path = path_
                total_cost = total_cost_

            i += 1

        return path, total_cost
        

    def optimize(self, distance_matrix, mode = 'random', kmax = 1, cycle = True):

        """
            Local search heuristic that explores two neighborhood structures in a deterministic way.

            distance_matrix: Square and symmetric matrix containing the distances, taken pairwise, between the points
            mode: Strategy to create the initial solution (random or greedy)
            kmax: Maximum number of neighborhood structures
            cycle: True if the traveling salesman must return to the starting point
        """

        self.path = self.initialization(distance_matrix, mode, cycle)
        self.total_cost = evaluate_path(self.path, distance_matrix)
        k = 0

        while k < kmax:

            path_, total_cost_ = self.local_search(self.path, self.total_cost, k, distance_matrix)

            if(total_cost_ < self.total_cost):
                self.path = path_
                self.total_cost = total_cost_
                k = 0
            else:
                k += 1

class IteratedLocalSearch():

    """ Iterated Local Search to solve the Travelling Salesman Problem (TSP). """

    def __init__(self):

        """
            path: Point indexes in the order they are visited
            costs: Costs of all trips between two consecutive points along the path
            total_cost: Total cost of path found (sum of all costs)
            liter: Number of iterations of the local search
            giter: Number of iterations of the ILS
        """

        self.path = list()
        self.costs = list()
        self.total_cost = None

    def initialization(self, distance_matrix, mode, cycle = True):

        """
            Creates an initial route.

            distance_matrix: Square and symmetric matrix containing the distances, taken pairwise, between the points
            mode: Strategy to create the initial solution (random or greedy)
            cycle: True if the traveling salesman must return to the starting point
        """

        if mode == 'random':
            
            n = distance_matrix.shape[0]
            path = np.random.choice(np.arange(1, n + 1), size = n, replace = False)

            if cycle is True:
                path = np.append(path, path[0])

        elif mode == 'greedy':

            GA = GreedyAlgorithm()
            GA.optimize(distance_matrix, cycle, verbose = False)
            path = GA.path

        return path

    def swap(self, path):

        """
            Swaps the position of two points on the route.

            path: Point indexes in the order they are visited
        """

        n = len(path)
        path_ = path.copy()

        i = np.random.randint(1, n-1)
        j = np.random.randint(1, n-1)
    
        path[i] = path_[j]
        path[j] = path_[i]

        return path

    def two_opt(self, path):

        """
            2-opt movement take two arcs from the route and reconnect these arcs with each other.

            path: Point indexes in the order they are visited
        """

        n = len(path)
        idx = np.random.randint(1, n-1, size = 2)
        i, j = sorted(idx)
    
        head = path[0:i]
        neck = path[j:-n+i-1:-1]
        tail = path[j+1:n]

        path = np.concatenate((head, neck, tail))

        return path
    
    def local_search(self, path, total_cost, liter, distance_matrix):

        """
            Local search starts from an initial solution and evolves that single solution into a mostly better and better solution through neighborhood structures.

            path: Point indexes in the order they are visited
            total_cost: Total cost of path found (sum of all costs) 
            distance_matrix: Square and symmetric matrix containing the distances, taken pairwise, between the points
        """

        i = 0
        while i < liter:
        
            path_ = self.two_opt(path)
            total_cost_ = evaluate_path(path_, distance_matrix)

            if(total_cost_ < total_cost):
                path = path_
                total_cost = total_cost_

            i += 1

        return path, total_cost

    def perturbation(self, path, level):

        for i in range(level):
            path = self.swap(path)
        
        return path

    def optimize(self, distance_matrix, giter = 100, liter = 20000, mode = 'random', cycle = True):

        level = 1
        self.path = self.initialization(distance_matrix, mode, cycle)
        self.total_cost = evaluate_path(self.path, distance_matrix)

        self.path, self.total_cost = self.local_search(self.path, self.total_cost, liter, distance_matrix)

        pbar = tqdm(total = giter, desc = 'Iterations')

        for i in range(giter):
            
            path = self.perturbation(self.path, level)
            total_cost = evaluate_path(path, distance_matrix)
            path, total_cost = self.local_search(path, total_cost, liter, distance_matrix)

            if(total_cost < self.total_cost):
                self.total_cost = total_cost
                self.path = path
                level = 1
            else:
                level += 1

            pbar.update(1)

        pbar.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--instance', type = str, default = './tsp_instances/instances/lin105.tsp', help = 'instance path')
    parser.add_argument('--cycle', action = 'store_true', help = 'True if the trip must start and end at the same point')
    parser.add_argument('--heuristic', type = str, default = 'VND', help = 'Heuristic name (can be Greedy, VND, or ILS)')
    args = parser.parse_args()
    print(args)

    data = read_instance(args.instance)

    distance_matrix = compute_weights(data)

    if args.heuristic.upper() == 'VND':

        print('Solving Traveling Salesman Problem using Variable Neighborhood Descent')

        VND = VariableNeighborhoodDescent()
        VND.optimize(distance_matrix, 'random', 1, args.cycle)

        print('Is the path feasible?', check_feasibility(VND.path, args.cycle))
        print('Best path:', VND.path)
        print('Best cost:', VND.total_cost)

    elif args.heuristic.upper() == 'GREEDY':

        print('Solving Traveling Salesman Problem using Greedy Algorithm')

        GA = GreedyAlgorithm()
        GA.optimize(distance_matrix, args.cycle)

        print('Is the path feasible?', check_feasibility(GA.path, args.cycle))
        print('Path:', GA.path)
        print('Total cost:', GA.total_cost)

    elif args.heuristic.upper() == 'ILS':

        print('Solving Traveling Salesman Problem using Iterated Local Search')

        ILS = IteratedLocalSearch()
        ILS.optimize(distance_matrix, giter = 100, liter = 10000, mode = 'random', cycle = args.cycle)

        print('Is the path feasible?', check_feasibility(ILS.path, args.cycle))
        print('Path:', ILS.path)
        print('Total cost:', ILS.total_cost)
    
    else:
        raise AssertionError('This heuristic has not been implemented yet.')

    opt_instance = '.' + args.instance.replace('/instances/', '/opt/').split('.')[-2] + '.opt.tour'
    if os.path.exists(opt_instance):
        opt_path = read_path(opt_instance, args.cycle)
        print('Optimal cost:', evaluate_path(opt_path, distance_matrix))