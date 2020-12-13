#!/usr/bin/env python3

import gurobipy as g
from copy import copy

class Base:
  def __init__(self, id):
    self.id = id

  def __eq__(self, other):
    return self.id == other.id

  def __neq__(self, other):
    return self.id != other.id

  def __hash__(self):
    return hash(self.id)

  def __repr__(self):
    return str(self.id)

class Node(Base):
  def __init__(self, id, incoming=None, outgoing=None):
    '''Takes an integer id,
    a set of incoming edges E- s.t. ∀(u, v) ∈ E-: v = self,
    a set of outgoing edges E+ s.t. ∀(v, w) ∈ E+: v = self.'''
    super(Node, self).__init__(id)
    self.id = id
    self.incoming = set() if incoming is None else incoming
    self.outgoing = set() if outgoing is None else outgoing

  def __copy__(self):
    return type(self)(id=self.id, incoming=copy(self.incoming), outgoing=copy(self.outgoing))

  def add_outgoing_edge(self, e):
    self.outgoing |= {e}

  def add_incoming_edge(self, e):
    self.incoming |= {e}

  def remove_outgoing_edges(self, E):
    self.outgoing -= E

  def remove_incoming_edges(self, E):
    self.incoming -= E

  def get_incoming_edges(self):
    return self.incoming

  def get_outgoing_edges(self):
    return self.outgoing

class DirEdge:
  def __init__(self, tail, head, l, u):
    '''Takes a Node object tail, and a Node object head, where e = (tail, head).
    The values l, u correspond to the lower (l), and upper (u) capacities.'''
    self.tail = tail
    self.head = head
    self.l = l
    self.u = u
    self.tail.add_outgoing_edge(self)
    self.head.add_incoming_edge(self)

  def __repr__(self):
    return str((self.tail.id, self.head.id, self.l, self.u))

  def __eq__(self, other):
    return self.tail == other.tail \
      and self.head == other.head \
      and self.l == other.l \
      and self.u == other.u

  def __neq__(self, other):
    return self.tail != other.tail \
      or self.head != other.head \
      or self.l != other.l \
      or self.u != other.u

  def __hash__(self):
    return hash((self.tail.id, self.head.id, self.l, self.u))

  def get_flow_requirements(self):
    return self.l, self.u

  def get_vertices(self):
    return self.tail, self.head

class Graph:
  def __init__(self, V=None, E=None, **kwargs):
    '''Takes a set V of Node objects, and a set E of Edge objects.'''
    self.V = set() if V is None else V
    self.E = set() if E is None else E

  def add_edge(self, e):
    '''Takes an Edge object e and adds it to the graph.'''
    self.V |= {e.tail, e.head}
    self.E |= {e}

  def get_max_flow(self):
    '''Constructs a linear program that computes the maximum flow of the graph.
    Returns a nonnegative real valued number.'''

    ''' IMPLEMENTATION NOTE ABOUT SOURCE AND SINK NODES:
    These nodes are virtual; that is, they will match the real ones by id, but they
    are not the same objects, and therefore, don't contain the same content.
    For instance: V & {source} = {source}, which is virtual, and shouldn't be used.
    However, V - (V - {source}) = {real_source}.'''
    source = Node(id=0)
    sink = Node(id=1)
    model = g.Model()
    model.setParam('OutputFlag', 0)
    x = {e: model.addVar(lb=e.l, ub=e.u, vtype=g.GRB.CONTINUOUS, name=str(e)) for e in self.E}
    model.update()
    for v in self.V - {source, sink}:
      inflow = sum([x[e] for e in v.get_incoming_edges()])
      outflow = sum([x[e] for e in v.get_outgoing_edges()])
      model.addConstr(inflow == outflow, name=str(v))
    model.setObjective(sum([x[e] for s in self.V - (self.V - {source}) for e in s.get_outgoing_edges()]), sense=g.GRB.MAXIMIZE)
    model.optimize()
    # print([x[e].x for e in self.E])
    return model.objVal

  def get_induced_graph_by_edges(self, E=None):
    '''Takes a set E of Edge objects (belonging to the graph).
    Returns a new graph G', which is the vertex-induced graph constructed from all the vertices in E.'''
    V = {copy(e.tail) for e in E} | {copy(e.head) for e in E}
    for v in V:
      v.remove_incoming_edges(self.E - E)
      v.remove_outgoing_edges(self.E - E)
    return Graph(V=V, E=E)

class Agent(Base):
  def __init__(self, id):
    '''Takes an integer id.'''
    super(Agent, self).__init__(id)

  def get_id(self):
    return self.id
