import numpy as np
import queue

class Graph(object):
    def __init__(self, nodes, connections):
        """
        __init__: Initialize graph
        nodes: A list of vertices
        connections: A list of connections
        """
        self.graph = dict((node, []) for node in nodes)
        for node1, node2, weight in connections:
            self.graph[node1].append((node2, weight))

    def breath_first_search(self, s, t):
        """
        Performing breath first search to find a path from vertex s to t
        Input:
        s: source vertex
        t: target vertex
        Return:
        Boolean variable: True if there exists an augmenting path, False otherwise
        parent: parent[v] is the parent of vertex v in BFS tree
        visited: True if a vertex is reachable from source vertex, False otherwise
        inc: The amount of flow increases
        """
        q = queue.Queue()
        parent = np.zeros(len(self.graph)).astype(int)
        visited = np.full(len(self.graph), False)
        inc = np.full(len(self.graph), np.inf)
        visited[s] = True
        q.put(s)
        while (not q.empty()):
            v = q.get()
            if v == t:
                return True, parent, visited, inc[t]
            for neighbor, w in self.graph[v]:
                if not visited[neighbor]:
                    parent[neighbor] = v
                    inc[neighbor] = min(inc[v], w)
                    visited[neighbor] = True
                    q.put(neighbor)

        return False, parent, visited, inc[t]

    def augmenting_path(self, s, t):
        """
        Performing augmenting path for Ford–Fulkerson method
        Input:
        s: source vertex
        t: target vertex
        Return:
        connections: augmenting edges
        increase: amount of flow increase
        """
        flag, parent, visited, inc = self.breath_first_search(s,t)
        connections = []
        increase = 0
        if flag:
            increase = inc
            v = t
            while (v != s):
                connections.append((parent[v],v))
                v = parent[v]
        return connections, visited, increase
