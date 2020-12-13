#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple, deque
from pprint import pprint as pp
 
 
inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')
 
class Graph():
    def __init__(self, edges = None):
        if edges is None: 
            self.edges = []
            self.vertices = set([])
            return
        self.edges = edges2 = [Edge(*edge) for edge in edges]
        self.vertices = set(sum(([e.start, e.end] for e in edges2), []))
 
    def add_edges(self, edges):
        edges2 = [Edge(*edge) for edge in edges]
        self.edges += edges2
        self.vertices = self.vertices.union(set(sum(([e.start, e.end] for e in edges2), [])))

    def dijkstra(self, source, dest):
        assert source in self.vertices
        dist = {vertex: inf for vertex in self.vertices}
        previous = {vertex: None for vertex in self.vertices}
        dist[source] = 0
        q = self.vertices.copy()
        neighbours = {vertex: set() for vertex in self.vertices}
        for start, end, cost in self.edges:
            neighbours[start].add((end, cost))
        #pp(neighbours)
 
        while q:
            u = min(q, key=lambda vertex: dist[vertex])
            q.remove(u)
            if dist[u] == inf or u == dest:
                break
            for v, cost in neighbours[u]:
                alt = dist[u] + cost
                if alt < dist[v]:                                  # Relax (u,v,a)
                    dist[v] = alt
                    previous[v] = u
        s, u = deque(), dest
        while previous[u]:
            s.appendleft(u)
            u = previous[u]
        s.appendleft(u)
        return s
 
if __name__ == "__main__":
    graph = Graph([(
                        (66.90684928029391, 24.009954216151783, 56.16392824523247, 3.493660688633727, 0.8789486800220921, 1.6642136810185195),
                        (38.9853503588711, -34.749675293677825, 61.44823178129354, 3.6166198555862263, 0.13615031994229435, 4.755446638583987),
                        7
                    ),
                    (
                        (38.9853503588711, -34.749675293677825, 61.44823178129354, 3.6166198555862263, 0.13615031994229435, 4.755446638583987),
                        (64.72712990564624, 35.6093328601184, 23.644419456285227, 2.5785622299132247, 2.8530905473117962, 0.46925432802386763),
                        9
                    ),
                    (
                        (38.9853503588711, -34.749675293677825, 61.44823178129354, 3.6166198555862263, 0.13615031994229435, 4.755446638583987),
                        (0.0, 5.0, 0.0, 0.0, 0.0, 0.0),
                        1
                    )])
    pp(graph.dijkstra(
        (66.90684928029391, 24.009954216151783, 56.16392824523247, 3.493660688633727, 0.8789486800220921, 1.6642136810185195),
        (64.72712990564624, 35.6093328601184, 23.644419456285227, 2.5785622299132247, 2.8530905473117962, 0.46925432802386763)
    ))
