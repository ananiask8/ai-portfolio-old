#!/usr/bin/env python2

import sys
from copy import copy, deepcopy
from math import sqrt

class Point:
  def __init__(self, x, y, idx):
    self.x = x
    self.y = y
    self.id = str((x, y))
    self.idx = idx

class Node:
  def __init__(self, id = None, outbound = None, inbound = None, parents = None, children = None, meta = None):
    self.id = id
    self.outbound = outbound if outbound is not None else {}
    self.inbound = inbound if inbound is not None else {}
    self.parents = parents if parents is not None else set()
    self.children = children if children is not None else set()
    self.meta = meta if meta is not None else None

  def __deepcopy__(self, memo):
    result = Node(self.id, self.outbound, self.inbound, self.parents, self.children, self.meta)
    return result

  def __eq__(self, other):
    return self.id == other.id

  def __neq__(self, other):
    return self.id is not other.id

  def __hash__(self):
    return hash(self.id)

  def add_edge(self, e):
    if e.u == self:
      self.outbound[e.v] = e
      self.children.add(e.v)
    else:
      self.inbound[e.u] = e
      self.parents.add(e.u)

  def remove_child(self, v):
    del self.outbound[v]
    self.children.remove(v)

  def remove_parent(self, u):
    del self.inbound[u]
    self.parents.remove(u)

class Edge:
  def __init__(self, id, u = None, v = None, direction = None):
    self.id = id
    self.u = u
    self.v = v
    self.direction = direction
    if u is not None and v is not None:
      u.add_edge(self)
      v.add_edge(self)

  def __eq__(self, other):
    return self.id == other.id

  def __neq__(self, other):
    return self.id is not other.id

  def __hash__(self):
    return hash(self.id)

  def is_forward_arc(self):
    if self.direction is None:
      return None
    return self.direction == "forward"

class Graph:
  def __init__(self, V = None, E = None, s = None, t = None):
    self.s = s
    self.t = t
    if V is not None and E is not None:
      self.V = V
      self.E = E
    else:
      self.V = set()
      self.E = set()

  def add_edge(self, e):
    self.V.add(e.u)
    self.V.add(e.v)
    self.E.add(e)

  def add_node(self, v):
    self.V.add(v)

  def remove_edge(self, e):
    if len(e.u.inbound) + len(e.u.outbound) == 0: self.V.remove(e.u)
    if len(e.v.inbound) + len(e.v.outbound) == 0: self.V.remove(e.v)
    if e in self.E: self.E.remove(e)
    if e.v in e.u.outbound: del e.u.outbound[e.v]
    if e.u in e.v.inbound: del e.v.inbound[e.u]

  def remove_node(self, x):
    for u, e in x.inbound.items():
      self.E.remove(e)
      u.remove_child(x)
      x.remove_parent(u)

    for v, e in x.outbound.items():
      self.E.remove(e)
      v.remove_parent(x)
      x.remove_child(v)

    self.V.remove(x)
    del x

  def pprint(self):
    out = ''
    for e in self.E:
      out += e.id + '\r\n'
    print(out)

def feasible_flow(G, l, u, f, s, t):
  b = {}
  for v in G.V:
    b[v] = sum([l[e] for e in v.inbound.values()]) - sum([l[e] for e in v.outbound.values()]) 

  s_p = Node("s'")
  G.add_node(s_p)
  G.s = s_p
  t_p = Node("t'")
  G.add_node(t_p)
  G.t = t_p

  e_inf = Edge("%s->%s" % (t.id, s.id), t, s)
  G.add_edge(e_inf)
  l[e_inf] = 0
  u[e_inf] = float("inf")
  f[e_inf] = 0

  for v in b.keys():
    if b[v] > 0:
      e = Edge("%s->%s" % (s_p.id, v.id), s_p, v)
      G.add_edge(e)
      l[e] = 0
      u[e] = b[v]
      f[e] = 0
    elif b[v] < 0:
      e = Edge("%s->%s" % (v.id, t_p.id), v, t_p)
      G.add_edge(e)
      l[e] = 0
      u[e] = -b[v]
      f[e] = 0

  old_l = {}
  for e in G.E:
    old_l[e] = l[e]
    u[e] = u[e] - l[e]
    l[e] = 0

  r = ford_fulkerson(G, l, u, f, s_p, t_p)
  if r == -1: return -1
  G.remove_node(s_p)
  G.remove_node(t_p)
  G.remove_edge(e_inf)
  for e in G.E: # Restoring original values
    u[e] = u[e] + old_l[e]
    r[e] = r[e] + old_l[e]
    l[e] = old_l[e]

  return r

def build_augmenting_path(came_from, final):
  stack = [final]
  visited = set([final])
  result = []
  capacity = [float("inf")]
  while len(stack) > 0:
    current = stack[-1]
    if came_from[current] is None: break

    break_out = False
    for next_dict in came_from[current]:
      next = next_dict["node"]
      cap = next_dict["capacity"]
      if next in visited: continue

      break_out = True
      visited.add(next)
      if current in next.parents:
        e = next.inbound[current]
        e.direction = "backward"
      else:
        e = next.outbound[current]
        e.direction = "forward"

      if break_out:
        result.append(e)
        capacity.append(min(min(capacity), cap))
        stack.append(next)
        break

    if break_out: continue
    stack.pop()
    result.pop()
    capacity.pop()

  return min(capacity), [r for r in reversed(result)]


def labeling(G, l, u, f, s, t):
  m = {}
  for v in G.V:
    m[v] = False
  m[s] = True

  stack = [s]
  came_from = {s: None}
  while len(stack) > 0:
    i = stack[-1]
    if i == t: return build_augmenting_path(came_from, t)

    break_out = False
    for e in list(i.inbound.values()) + list(i.outbound.values()):
      if m[e.u] and not m[e.v] and f[e] < u[e]:
        break_out = True
        new_capacity = float("inf") if u[e] == float("inf") else abs(f[e] - u[e])
        m[e.v] = True
        stack.append(e.v)
        if e.v not in came_from: came_from[e.v] = []
        came_from[e.v].append({"node": e.u, "capacity": new_capacity})
      elif not m[e.u] and m[e.v] and f[e] > l[e]:
        break_out = True
        new_capacity = abs(f[e] - l[e])
        m[e.u] = True
        stack.append(e.v)
        if e.v not in came_from: came_from[e.v] = []
        came_from[e.v].append({"node": e.u, "capacity": new_capacity})
    if break_out: continue
    stack.pop()

  return 0, []

def ford_fulkerson(G, l, u, f, s, t):
  feasible = False
  while True:
    capacity, path = labeling(G, l, u, f, s, t)
    # print([e.id for e in path])
    if capacity > 0:
      feasible = True
      for e in path:
        if e.is_forward_arc():
          f[e] = f[e] + capacity
        else:
          f[e] = f[e] - capacity
    elif not feasible:
      return -1
    else: break

  return f

def bellman_ford(G, s, c):
  d = {}
  for v in G.V:
    d[v] = float("inf")
  d[s] = 0

  p = {}
  for i in range(1, len(G.V) + 1):
    for e in G.E:
      v, w = e.u, e.v
      if d[w] > d[v] + c[e]:
        # print(("BELLMANFORD", e.id, d[v] + c[e], d[w]))
        d[w] = d[v] + c[e]
        p[w] = v

  return d, p

def contains_negative_cycle(G, d, c):
  for e in G.E:
    v, t = e.u, e.v
    if d[t] > d[v] + c[e]:
      return True
  return False

def negative_cycle(G, u, p, c):
  color = {}
  # print([(v.id, p[v].id) for v in p.keys()])
  for w in p.keys():
    # Loop start.
    # Loop broken, didn't find cycle. Reset colors.
    C = set()
    for t in p.keys():
      color[t] = "w"
  
    while w in p.keys():
      if color[w] == "b":
        cap = float("inf")
        for e in C:
          cap = min(cap, u[e])
        return cap, C
      elif color[w] == "g":
        color[w] = "b"
        C.add(Edge("%s->%s" % (p[w].id, w.id), p[w], w))
      else: color[w] = "g"
      w = p[w]

  return 0, set()


def build_residual_graph(G, l, u, c, f):
  # b = {}
  # for x in G.V:
  #   b[x] = sum([f[e] for e in x.inbound.values()]) - sum([f[e] for e in x.outbound.values()])

  Gf = Graph()
  cf = {}
  uf = {}
  for e in G.E:
    eij = Edge("%s->%s" % (e.u.id, e.v.id), e.u, e.v)
    cf[eij] = c[e]
    uf[eij] = u[e] - f[e]
    Gf.add_edge(eij)

    eji = Edge("%s->%s" % (e.v.id, e.u.id), e.v, e.u)
    f[eji] = -f[e]
    cf[eji] = -c[e]
    uf[eji] = f[e] - l[e]
    Gf.add_edge(eji)

  return Gf, uf, cf

def cycle_canceling(G, l, u, c, f):
  from pprint import pprint

  Gf, uf, cf = build_residual_graph(G, l, u, c, f)
  out = []
  while True:
    Gf_copy = deepcopy(Gf)
    for e in Gf_copy.E:
      out.append(e.id + ' ' + str(round(cf[e], 2)) + ' ' + str(round(uf[e], 2)) + '\r\n')
    # print('\r\n'.join(sorted(out)))
    # input()
    for e in Gf.E:
      if uf[e] == 0:
        Gf_copy.remove_edge(e)

    s_p = Node("s'")
    Gf_copy.add_node(s_p)
    for v in Gf_copy.V:
      if v == s_p: continue
      e = Edge("%s->%s" % (s_p.id, v.id), s_p, v)
      cf[e] = 0
      Gf_copy.add_edge(e)


    d, p = bellman_ford(Gf_copy, s_p, cf)
    # pprint([(v.id, d[v]) for v in d])
    # input()
    if not contains_negative_cycle(Gf_copy, d, cf): return f
    
    cap, C = negative_cycle(Gf, uf, p, cf)
    # print(cap)
    pprint([e.id for e in C])
    input()
    if cap > 0:
      for e in C:
        e_op = Edge("%s->%s" % (e.v.id, e.u.id), e.v, e.u)  
        if e in G.E:
          f[e] = f[e] + cap
        else:
          f[e_op] = f[e_op] - cap

        uf[e] = uf[e] - cap
        uf[e_op] = uf[e_op] + cap
    else: break

  return f

def euclid(p1, p2):
  return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def build_graph_for_frame_pair_assignment(f_k, f_k_1):
  l = {}
  u = {}
  f = {}
  c = {}
  G = Graph()
  G.s = Node("s")
  G.t = Node("t")
  G.add_node(G.s)
  G.add_node(G.t)
  V = []
  
  n = len(f_k)
  for i in range(n):
    v = Node(str(f_k[i].idx) + f_k[i].id)
    v.meta = f_k[i]
    V.append(v)
    e = Edge("%s->%s" % (G.s.id, v.id), G.s, v)
    l[e] = 0
    u[e] = 1
    f[e] = 0
    c[e] = 0
    G.add_edge(e)
  for i in range(n):
    v = Node(str(f_k_1[i].idx) + f_k_1[i].id)
    v.meta = f_k_1[i]
    V.append(v)
    e = Edge("%s->%s" % (v.id, G.t.id), v, G.t)
    l[e] = 0
    u[e] = 1
    f[e] = 0
    c[e] = 0
    G.add_edge(e)

  for i in range(n):
    v = V[i]
    for j in range(n):
      w = V[n + j]
      e = Edge("%s->%s" % (v.id, w.id), v, w)
      l[e] = 0
      u[e] = 1
      f[e] = 0
      c[e] = euclid(f_k[i], f_k_1[j])
      G.add_edge(e)

  return G, l, u, f, c

def get_assignments(f, n, s, t):
  results = [""]*n
  for e in f.keys():
    if f[e] is 1 and not e.u == s and not e.v == t:
      # print(e.id)

      results[e.u.meta.idx] = str(e.v.meta.idx + 1)

  return results

# Read input
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")
line_content = input_file.readline().split(" ")
n, p = int(line_content[0]), int(line_content[1])

frames = []
for i in range(0, p):
  line_content = [int(token) for token in input_file.readline().split(" ")]
  frames.append([])
  for j in range(0, 2*n, 2):
    xy = Point(line_content[j], line_content[j + 1], j // 2)
    frames[i].append(xy)

input_file.close()

# Solve for each pair of frames independently
assignments = []
for k in range(p - 1):
  G, l, u, f, c = build_graph_for_frame_pair_assignment(frames[k], frames[k + 1])
  G_copy = deepcopy(G)
  f = ford_fulkerson(G_copy, copy(l), copy(u), copy(f), G_copy.s, G_copy.t)
  f = cycle_canceling(G, l, u, c, f)
  # print("\r\n".join(sorted([str((e.id, val)) for e, val in f.items() if e not in c])))
  assignments.append(get_assignments(f, n, G.s, G.t))

out = ""
for ass in assignments:
  # print(" ".join(ass))
  out += " ".join(ass)
  out += "\r\n"

output_file.write(out[:-2])
output_file.close()
