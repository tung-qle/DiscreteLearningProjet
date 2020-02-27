from graph_cut import GraphCut
from graph import Graph
from push_relabel import PushRelabel
import numpy as np
import time

def random_graph(n, epsilon = 0.2, source = 0, target = 3):
    connections = []
    nodes = []
    for i in range(n):
        nodes.append(i)
        if i == target:
            continue
        for j in range(n):
            if j == i:
                continue
            r = 2
            if i == source or j == target:
                r = 150
            t = np.random.rand()
            if (t <= epsilon):
                w = np.random.randint(r) + 1
                connections.append((i,j,w))
    return nodes, connections

def check_validity(graph, s, t, pr = False):
    result = True
    balance = np.zeros(len(graph))
    for node, neighbors in graph.items():
        if not pr:
            for v, capacity, weight in neighbors:
                if capacity < weight:
                    print("Overflow", capacity - weight)
                    result = False
                balance[v] += weight
                balance[node] -= weight
        else:
            for v, (capacity, weight) in neighbors.items():
                if capacity < weight:
                    print("Overflow", capacity - weight)
                    result = False
                balance[v] += weight
                balance[node] -= weight

    for node in graph.keys():
        if node == s or node == t:
            continue
        if abs(balance[node]) > 1e-10:
            print("Unbalanced", balance[node])
            result = False
    return result

s = 0
t = 3
nodes, connections = random_graph(500, epsilon = 0.05, source = s, target = t)
start = time.time()
g = PushRelabel(nodes, connections)
g.min_cut(s, t)
print(g.calculate_max_flow(s,t))
print(g.calculate_cut_value())
print(check_validity(g.graph, s, t, True))
end = time.time()
print("Time: ", end - start)

start = time.time()
g = GraphCut(nodes, connections)
g.min_cut(s, t)
print(g.calculate_max_flow(s,t))
print(g.calculate_cut_value())
print(check_validity(g.graph, s, t, False))
end = time.time()
print("Time: ", end - start)
