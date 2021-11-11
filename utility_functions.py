# pip install numpy
import numpy as np

def read_instance(instance):

    """
        Reads instances of the Traveling Salesman Problem (TSP).
        (http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/)
    """

    try:
        f = open(instance, 'r')
    except:
        raise AssertionError('Incorrect instance path.')

    data = dict()
    for line in f.readlines():
        
        if 'EOF' in line:
            break
        if 'coords' in data:
            # Removing empty strings in the line (e.g., rat99.tsp)
            line = [i for i in line.split(' ') if i]
            try:
                x = int(line[1])
                y = int(line[2])
            except:
                x = int(float(line[1]))
                y = int(float(line[2]))
            data['coords'][i,:] = [x, y]
            i += 1
        if 'EDGE_WEIGHT_TYPE' in line:
            data['dfunc'] = line.split(': ')[-1].split('\n')[0]
        if 'DIMENSION' in line:
            data['dim'] = int(line.split(': ')[-1].split('\n')[0])
        if 'NODE_COORD_SECTION' in line:
            data['coords'] = np.zeros((data['dim'],2))
            i = 0

    return data

def compute_weights(data):

    """
        Calculates the distance matrix.
        The distance from one point to another can be Euclidean (EUC_2D) or pseudo-Euclidean (ATT).
    """
    
    idx = 0
    distance_matrix = np.zeros((data['dim'], data['dim']))
    
    for i in range(data['dim']):
        for j in range(data['dim'])[idx:]:
            
            xd = data['coords'][i,0] - data['coords'][j,0]
            yd = data['coords'][i,1] - data['coords'][j,1]
            
            if data['dfunc'] == 'EUC_2D':
                # rint: converts input to the nearest integer
                distance_matrix[i,j] = np.rint(np.sqrt(xd**2 + yd**2))
                distance_matrix[j,i] = np.rint(np.sqrt(xd**2 + yd**2))
            elif data['dfunc'] == 'ATT':
                rij = np.sqrt((xd**2 + yd**2)/10)
                tij = np.rint(rij)
                if tij < rij:
                    distance_matrix[i,j] = tij + 1
                    distance_matrix[j,i] = tij + 1
                else:
                    distance_matrix[i,j] = tij
                    distance_matrix[j,i] = tij
            else:
                print('Unknown distance function.')
        
        idx += 1
    
    return distance_matrix.astype('int')

def check_feasibility(path, cycle):

    """
        Checks the feasibility of a found path.
    """
    
    n = path.shape[0]
    
    if cycle is True:
        points = np.arange(1, n)
        # Starting point is the same as ending point and there is no repeated point on the path
        if(path[0] == path[-1] and all(points == sorted(path[:-1]))):
            return True
        else:
            return False
    else:
        points = np.arange(1, n+1)
        # There is no repeated point on the path
        if(all(points == sorted(path))):
            return True
        else:
            return False

def read_path(instance, cycle):

    """
        Reads the optimal solution of a given instance.
    """

    try:
        f = open(instance, 'r')
    except:
        raise AssertionError('Incorrect path or optimal path unknown.')
    start = False

    for line in f.readlines():
        
        if '-1' in line:
            break    
        if 'DIMENSION' in line:
            n = int(line.split(': ')[-1].split('\n')[0])
        if start is True:
            opt_path[i] = line.split(' ')[0].split('\n')[0]
            i += 1
        if 'TOUR_SECTION' in line:
            if cycle is True:
                opt_path = np.zeros((n+1,))
            else:
                opt_path = np.zeros((n,))
            start = True
            i = 0

    if cycle is True:
        opt_path[i] = opt_path[0]
    opt_path = opt_path.astype('int')

    if check_feasibility(opt_path, cycle) is True:
        return opt_path
    else:
        raise AssertionError('Infeasible path.')

def evaluate_path(opt_path, distance_matrix):

    n = opt_path.shape[0]
    opt_costs = list()

    for i in range(0, n-1):
        opt_costs.append(distance_matrix[opt_path[i]-1, opt_path[i+1]-1])
    
    return sum(opt_costs)
