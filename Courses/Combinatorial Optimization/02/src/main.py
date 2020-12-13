#!/usr/bin/env python2

import sys
from copy import copy, deepcopy

class Node:
  def __init__(self, id = None, outbound = None, inbound = None, parents = None, children = None):
    self.id = id
    self.outbound = outbound if outbound is not None else {}
    self.inbound = inbound if inbound is not None else {}
    self.parents = parents if parents is not None else set()
    self.children = children if children is not None else set()

  def __deepcopy__(self, memo):
    result = Node(self.id, self.outbound, self.inbound, self.parents, self.children)
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

class Customer:
  def __init__(self, id, l = None, u = None, products = None):
    self.id = id
    self.l = l
    self.u = u
    self.products = products

class Network:
  def __init__(self, G, l, u, f, s, t):
    self.G = G
    self.l = l
    self.u = u
    self.f = f
    self.s = s
    self.t = t

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
    for e in i.inbound.values() + i.outbound.values() :
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

def validate_kirchhoff(G, l, u, f, s, t):
  b = {}
  for x in G.V:
    b[x] = sum([f[e] for e in x.inbound.values()]) - sum([f[e] for e in x.outbound.values()])

  result = True
  for x in G.V:
    if x == s or x == t: continue
    result &= b[x] == 0

  result &= b[s] == -b[t]

  return result

def validate_feasibility(G, l, u, f, s, t):
  ok = {}
  for e in G.E:
    # print((e.id, l[e], f[e], u[e]))
    ok[e] = l[e] <= f[e] <= u[e]

  result = True
  for e in G.E:
    result &= ok[e]

  if result == False:
    for e in G.E:
      if l[e] > f[e]: print((e.id, l[e], f[e], u[e]))

  return result


# Read input
input_file = open(sys.argv[1], "r")
output_file = open(sys.argv[2], "w")
line_content = input_file.readline().split(" ")
size_C, size_P = int(line_content[0]), int(line_content[1])

customers = []
for i in range(size_C):
  line_content = input_file.readline().split(" ")
  l = int(line_content.pop(0))
  u = int(line_content.pop(0))
  products = [int(p) - 1 for p in line_content]
  customers.append(Customer(i, l, u, products))

demand = [int(d) for d in input_file.readline().split(" ")]
input_file.close()

# Build network
C = [Node("C%d" % int(i + 1)) for i in range(size_C)]
P = [Node("P%d" % int(i + 1)) for i in range(size_P)]

s = Node("source")
t = Node("sink")
G = Graph()
G.s = s
G.t = t
l = {}
f = {}
u = {}
for i in range(size_C):
  e = Edge("%s->%s" % (s.id, C[i].id), s, C[i])
  G.add_edge(e)
  l[e] = customers[i].l
  u[e] = customers[i].u
  f[e] = 0
  for j in customers[i].products:
    e = Edge("%s->%s" % (C[i].id, P[j].id), C[i], P[j])
    G.add_edge(e)
    l[e] = 0
    u[e] = 1
    f[e] = 0

    e = Edge("%s->%s" % (P[j].id, t.id), P[j], t)
    G.add_edge(e)
    l[e] = demand[j]
    u[e] = float("inf")
    f[e] = 0

# Calculate initial feasible flow
G_copy = deepcopy(G)
f = feasible_flow(G_copy, copy(l), copy(u), copy(f), G_copy.s, G_copy.t)
# print(validate_kirchhoff(G, l, u, f, s, t))
# print(validate_feasibility(G, l, u, f, s, t))
max_flow = ford_fulkerson(G, l, u, f, s, t)
# print(validate_kirchhoff(G, l, u, f, s, t))
# print(validate_feasibility(G, l, u, f, s, t))
# exit()

# exit()
total = 0
out = ""
if not validate_kirchhoff(G, l, u, f, s, t) or not validate_feasibility(G, l, u, f, s, t):
  out = "-1\r\n"
else:
  for i in range(size_C):
    line = []
    for p, val in C[i].outbound.items():
      # print(C[i].id, p.id, max_flow[val])
      if max_flow[val] == 1:
        total += 1
        line.append(int(p.id.replace("P", "")))

    out += " ".join([str(x) for x in sorted(line)])
    out += "\r\n"

# print(total)
output_file.write(out[:-2])
output_file.close()
