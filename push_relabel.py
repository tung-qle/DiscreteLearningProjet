from graph import Graph
import numpy as np
import queue

class PushRelabel(object):
    """
    Performing a minimal graphcut using Push-Relabel algorithm.
    """
    def __init__(self, nodes, connections):
        """
        __init__: Initialize graph
        nodes: A list of vertices
        connections: A list of connections
        mincut: True if belongs to s, False otherwise
        excess: amount of flow excess
        height: label of vertices
        """
        self.graph = dict((node, dict()) for node in nodes)
        self.reverse_graph = dict((node, dict()) for node in nodes)
        for node1, node2, capacity in connections:
            self.graph[node1][node2] = (capacity, 0)
            self.reverse_graph[node2][node1] = (capacity, 0)
        self.mincut = np.full(len(self.graph), False)
        self.excess = np.zeros(len(self.graph))
        self.height = np.zeros(len(self.graph))

    def residual_graph_build(self):
        connections = []
        for node, neighbors in self.graph.items():
            for neighbor, (c, f) in neighbors.items():
                if c > f:
                    connections.append((node, neighbor, c - f))
                if f > 0:
                    connections.append((neighbor, node, f))
        return connections

    def relabel(self, u):
        min_height = np.inf
        neighbors = self.graph[u]
        for v, (capacity, weight) in neighbors.items():
            if capacity > weight:
                min_height = min(min_height, self.height[v])

        neighbors = self.reverse_graph[u]
        for v, (capacity, weight) in neighbors.items():
            if weight > 0:
                min_height = min(min_height, self.height[v])
        self.height[u] = min_height + 1

    def discharge(self, u):
        if self.excess[u] <= 0:
            return

        neighbors = self.graph[u]
        for v, (capacity, weight) in neighbors.items():
            if capacity > weight and self.height[u] > self.height[v]:
                send = min(self.excess[u], capacity - weight)
                self.graph[u][v] = (capacity, weight + send)
                self.reverse_graph[v][u] = (capacity, weight + send)
                self.excess[u] -= send
                self.excess[v] += send
            if self.excess[u] <= 0:
                return

        neighbors = self.reverse_graph[u]
        for v, (capacity, weight) in neighbors.items():
            if weight > 0 and self.height[u] > self.height[v]:
                send = min(self.excess[u], weight)
                self.reverse_graph[u][v] = (capacity, weight - send)
                self.graph[v][u] = (capacity, weight - send)
                self.excess[u] -= send
                self.excess[v] += send
            if self.excess[u] <= 0:
                return

        self.relabel(u)

    def min_cut(self, s, t):
        q = [vertex for vertex in self.graph.keys() if vertex != s and vertex != t]
        self.height[s] = len(self.graph)

        neighbors_s = self.graph[s]
        for v, (capacity, weight) in neighbors_s.items():
            neighbors_s[v] = (capacity, capacity)
            self.reverse_graph[v][s] = (capacity, capacity)
            self.excess[v] = capacity

        p = 0
        while p < len(q):
            u = q[p]
            old_height = self.height[u]
            self.discharge(u)
            if self.height[u] > old_height:
                q.insert(0, q.pop(p))
                p = 0
            else:
                p += 1

        residual_graph = Graph(self.graph.keys(), self.residual_graph_build())
        _, visited, _ = residual_graph.augmenting_path(s, t)
        # print(visited)
        self.mincut = visited

    def calculate_cut_value(self):
        result = 0
        for node, neighbors in self.graph.items():
            if not self.mincut[node]:
                continue

            for neighbor, (c, _) in neighbors.items():
                if not self.mincut[neighbor]:
                    result = result + c
        return result

    def calculate_max_flow(self, s, t):
        neighbor_s = self.graph[s]
        result = 0
        for _, (_, w) in neighbor_s.items():
            result = result + w

        neighbor_s = self.reverse_graph[s]
        for _, (_, w) in neighbor_s.items():
            result = result - w
        return result
