from main import Node, Edge, Graph


# From problems given only with balances and no source and sink
# Assuming all lower bounds are 0, but source and sink 's
# source can be added s->(positive balance nodes) with lower bound as balance
# sink can be added (negative balance nodes)->t with lower bound as balance
# u[a source edge] = sum of balance node outgoing edges
# u[a sink edge] = sum of balance node incoming edges
s = Node("s")
t = Node("t")
V = [Node("1"), Node("2"), Node("3"), Node("4"), Node("5"), s, t]
c = {}
l = {}
u = {}
f = {}
es1 = Edge("%s->%s" % (V[5].id, V[0].id), V[5], V[0])
c[es1] = 0
l[es1] = 2
f[es1] = 0
u[es1] = 6
e13 = Edge("%s->%s" % (V[0].id, V[2].id), V[0], V[2])
c[e13] = 2
l[e13] = 0
f[e13] = 0
u[e13] = 4
e14 = Edge("%s->%s" % (V[0].id, V[3].id), V[0], V[3])
c[e14] = 4
l[e14] = 0
f[e14] = 0
u[e14] = 2

es2 = Edge("%s->%s" % (V[5].id, V[1].id), V[5], V[1])
c[es2] = 0
l[es2] = 5
f[es2] = 0
u[es2] = 12
e21 = Edge("%s->%s" % (V[1].id, V[0].id), V[1], V[0])
c[e21] = 1
l[e21] = 0
f[e21] = 0
u[e21] = 4
e23 = Edge("%s->%s" % (V[1].id, V[2].id), V[1], V[2])
c[e23] = 5
l[e23] = 0
f[e23] = 0
u[e23] = 3
e25 = Edge("%s->%s" % (V[1].id, V[4].id), V[1], V[4])
c[e25] = 5
l[e25] = 0
f[e25] = 0
u[e25] = 5
e34 = Edge("%s->%s" % (V[2].id, V[3].id), V[2], V[3])
c[e34] = 4
l[e34] = 0
f[e34] = 0
u[e34] = 3
e35 = Edge("%s->%s" % (V[2].id, V[4].id), V[2], V[4])
c[e35] = 4
l[e35] = 0
f[e35] = 0
u[e35] = 1

e4t = Edge("%s->%s" % (V[3].id, V[6].id), V[3], V[6])
c[e4t] = 0
l[e4t] = 4
f[e4t] = 0
u[e4t] = 5
e45 = Edge("%s->%s" % (V[3].id, V[4].id), V[3], V[4])
c[e45] = 4
f[e45] = 0
l[e45] = 0
u[e45] = 2

e5t = Edge("%s->%s" % (V[4].id, V[6].id), V[4], V[6])
c[e5t] = 0
l[e5t] = 3
f[e5t] = 0
u[e5t] = 8
G = Graph()
G.V = set(V)
G.E = set([e13, e14, e21, e23, e25, e34, e35, e45, es1, es2, e4t, e5t])
G.s = s
G.t = t
G_copy = deepcopy(G)
f = feasible_flow(G_copy, copy(l), copy(u), copy(f), G_copy.s, G_copy.t)
# f = ford_fulkerson(G, l, u, f, G.s, G.t)
# print([(e.id, f[e]) for e in f.keys()])
# exit()
f = cycle_canceling(G, l, u, c, f)
min_cost =0
for e in G.E:
  min_cost += c[e]
  print((e.id, c[e]))

print(min_cost)
