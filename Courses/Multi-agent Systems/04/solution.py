#!/usr/bin/env python3

from math import factorial, log
import pprint
import random
import gurobipy as g
from copy import copy
from itertools import permutations, chain, combinations

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
    self.V = set() if V is None else V
    self.E = set() if E is None else E

  def add_edge(self, e):
    self.V |= {e.tail, e.head}
    self.E |= {e}

  def get_max_flow(self):
    # These nodes are virtual, meaning that they will match the real ones by id, but they don't
    # contain the same contents. For instance: V & {source} = {source}, which is virtual, and
    # shouldn't be used. However, V - (V - {source}) = {real_source}.
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
    V = {copy(e.tail) for e in E} | {copy(e.head) for e in E}
    for v in V:
      v.remove_incoming_edges(self.E - E)
      v.remove_outgoing_edges(self.E - E)
    return Graph(V=V, E=E)

class Agent(Base):
  def __init__(self, id):
    super(Agent, self).__init__(id)

  def get_id(self):
    return self.id

class Game:
  def banzhaf(self):
    v = self.compute_values_of_coalitions()
    n = len(self.A)
    acc_marginals = {a: 0 for a in self.A}
    for p in self._powerset(self.A.keys()):
      for a in self.A.keys():
        if a in p:
          A = frozenset(p)
          acc_marginals[a] += v[A]
          acc_marginals[a] -= v[A - {a}]
    for a in self.A:
      acc_marginals[a] /= 2**(n-1)
    return acc_marginals

  def shapley(self):
    v = self.compute_values_of_coalitions()
    n = len(self.A)
    acc_marginals = {a: 0 for a in self.A}
    for p in permutations(self.A.keys()):
      for i in range(n):
        a = p[i]
        sp = p[:i + 1]
        A = frozenset(sp)
        acc_marginals[a] += v[A]
        acc_marginals[a] -= v[A - {a}]
    for a in self.A:
      acc_marginals[a] /= factorial(n)
    return acc_marginals

  def shapley_approx(self, m):
    v = self.compute_values_of_coalitions()
    n = len(self.A)
    acc_marginals = {a: 0 for a in self.A}
    P = list(permutations(self.A.keys()))
    for i in range(m):
      p = random.choice(P)
      for i in range(n):
        a = p[i]
        sp = p[:i + 1]
        A = frozenset(sp)
        acc_marginals[a] += v[A]
        acc_marginals[a] -= v[A - {a}]
    for a in self.A:
      acc_marginals[a] /= m
    return acc_marginals

class FlowGame(Graph, Game):
  def __init__(self, A=None, G=None):
    super().__init__(A=A, G=G)
    self.A = A
    self.G = G

  def _powerset(self, iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

  def get_worth_of_coalition(self, A=None):
    E = set().union(*[self.A[a] for a in A])
    G = self.G.get_induced_graph_by_edges(E)
    return G.get_max_flow()

  def compute_values_of_coalitions(self):
    v = {}
    for p in self._powerset(self.A.keys()):
      A = frozenset(p)
      v[A] = self.get_worth_of_coalition(A)
    return v

def prepare(args):
  input_file = open(args.input, "r")
  size_of_V, size_of_E, size_of_N = map(int, input_file.readline().split(" "))
  G = Graph()
  V = {id: Node(id) for id in range(size_of_V)}
  A = {Agent(id): set() for id in range(1, size_of_N + 1)}
  for i in range(size_of_E):
    tail_id, head_id, cap_e, agent_id = map(int, input_file.readline().split(" "))
    a = Agent(agent_id)
    u, v = V[tail_id], V[head_id]
    e = DirEdge(tail=u, head=v, l=0, u=cap_e)
    G.add_edge(e)
    A[a] |= {e}
  return A, G

def run(FG):
  import time

  out = "\x1bc"
  out += "=============================================================\n"
  out += "Banzhaf indices for a Flow Game with |V|={}, |E|={}, |A|={}.\n".format(len(G.V), len(G.E), len(A.keys()))
  out += "-------------------------------------------------------------\n"
  start = time.time()
  out += pprint.pformat(FG.banzhaf())
  end = time.time()
  out += "\n"
  out += "-------------------------------------------------------------\n"
  out += "Elapsed time of {} seconds.\n".format(end - start)
  out += "=============================================================\n"
  out += "\n\n"
  out += "=============================================================\n"
  out += "Shapley indices for a Flow Game with |V|={}, |E|={}, |A|={}.\n".format(len(G.V), len(G.E), len(A.keys()))
  out += "-------------------------------------------------------------\n"
  start = time.time()
  out += pprint.pformat(FG.shapley())
  end = time.time()
  out += "\n"
  out += "-------------------------------------------------------------\n"
  out += "Elapsed time of {} seconds.\n".format(end - start)
  out += "=============================================================\n"
  out += "\n"
  out += "=============================================================\n"
  out += "Shapley approx. for a Flow Game with |V|={}, |E|={}, |A|={}.\n".format(len(G.V), len(G.E), len(A.keys()))
  out += "-------------------------------------------------------------\n"
  start = time.time()
  n = len(FG.A)
  m = int(n*log(factorial(n))**log(n))
  out += pprint.pformat(FG.shapley_approx(m=m))
  end = time.time()
  out += "\n"
  out += "-------------------------------------------------------------\n"
  out += "Elapsed time of {} seconds.\n".format(end - start)
  out += "=============================================================\n"
  out += "\n"
  print(out)

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.io", type=str, help="Name of file to read.")
    parser.add_argument("--output", default="output.io", type=str, help="Name of file to write to.")
    args = parser.parse_args()

    A, G = prepare(args)
    FG = FlowGame(A, G)
    run(FG)
