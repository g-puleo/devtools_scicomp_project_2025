import os
import yaml
def distance( point1: list[float], point2: list[float]) -> float:
    """Calculates the squared Euclidean distance between two points."""
    N = len(point1)
    dist = 0.0
    for i in range(N):
        dist += (point1[i] - point2[i])**2

    return dist
def majority_vote(neighbours : list[int]) :
    
    classes_present = set(neighbours)
    dict_counts = {}
    for idx in classes_present: 

        count = 0

        for neighbour in neighbours: 
            if neighbour==idx:
                count+=1
        
        dict_counts[idx] = count

    highest = list(dict_counts.values())[0]
    idx_max = list(dict_counts.keys())[0]
    for idx in dict_counts: 
        if dict_counts[idx]>highest:
            highest = dict_counts[idx]
            idx_max = idx
        
    return idx_max

def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs

def read_file(filename):
    data = []
    labels = []
    labelsdict = {'b':0, 'g': 1}

    with open(filename) as f: 
        for line in f:
            words = line.split(sep=',')
            data.append( [float(x) for x in words[2:-1]])
            labels.append(labelsdict[words[-1][0]])
    
    return data, labels