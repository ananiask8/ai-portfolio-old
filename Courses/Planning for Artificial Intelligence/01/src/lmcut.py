#!/usr/bin/env python2

import sys
from problem import *
import time

class BinHeap:
    def __init__(self):
        self.nodes = [0]
        self.current_size = 0

    def __len__(self):
        return self.current_size + 1
    
    @staticmethod
    def _parent(i):
        return i // 2

    @staticmethod
    def _left(i):
        return 2*i

    @staticmethod
    def _right(i):
        return 2*i + 1
    
    def _swap(self, i, j):
        self.nodes[i], self.nodes[j] = self.nodes[j], self.nodes[i]
        self.nodes[i].index = i
        self.nodes[j].index = j

    def insert(self, v):
        self.current_size = self.current_size + 1
        v.index = self.current_size
        self.nodes.append(v)
        self.bubble_up(self.current_size)


    def decrease_key(self, v):
        i = v.index
        while self._parent(i) > 0:
            parent = self._parent(i)
            if self.nodes[parent] < self.nodes[i]: break
            self._swap(i, parent)
            i = parent


    def bubble_down(self, i):
        while i * 2 <= self.current_size:
            j = self.min_of_children(i)
            if self.nodes[i] > self.nodes[j]:
                self._swap(i, j)
            i = j


    def bubble_up(self, i):
        while i // 2 > 0:
            if self.nodes[i] < self.nodes[i // 2]:
                self.nodes[i], self.nodes[i // 2] = self.nodes[i // 2], self.nodes[i]
                self.nodes[i].index, self.nodes[i // 2].index = i, i // 2
            i = i // 2

    def extract_min(self):
        val = self.nodes[1]
        val.index = None

        self.nodes[1] = self.nodes[self.current_size]
        self.current_size = self.current_size - 1
        self.nodes[1].index = 1
        self.nodes.pop()
        self.bubble_down(1)

        return val

    def min_of_children(self, i):
        if (i * 2) + 1 > self.current_size:
            return i * 2
        else:
            if self.nodes[i * 2] < self.nodes[(i * 2) + 1]:
                return i * 2
            else:
                return (i * 2) + 1

    def build_heap(self, g):
        i = len(g) // 2
        self.current_size = len(g)
        self.nodes = [0] + g[:]
        for i in range(1, self.current_size + 1):
            self.nodes[i].index = i
        while i > 0:
            self.bubble_down(i)
            i -= 1

class PriorityQueue:
    def __init__(self, l = None):
        self.heap = BinHeap()
        if l is not None:
            self.heap.build_heap(l)

    def is_empty(self):
        return self.heap.current_size == 0

    def insert_with_priority(self, v):
        if self.in_queue(v):
            self.update_key(v)
        else:
            self.heap.insert(v)

    def update_key(self, v):
        self.heap.decrease_key(v)

    def pull_highest_priority_element(self):
        return self.heap.extract_min()

    def in_queue(self, v):
        # return v.index <= self.heap.size
        return v.index is not None

class Operator:
    def __init__(self, id, pre, add, delete, cost):
        self.id = id
        self.pre = pre
        for s in pre:
            s.insert_pre(self)
        self.add = add
        for s in add:
            s.insert_add(self)
        self.delete = delete
        self.cost = cost
        self.original_cost = cost
        self.u = len(pre)

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def supp(self):
        return self.supporter

    def set_supp(self):
        self.supporter = max(self.pre)

    def reset(self):
        self.cost = self.original_cost
        self.u = len(self.pre)

    def p(self):
        return (self.id, [s.id for s in self.pre], [s.id for s in self.add], [s.id for s in self.delete], self.cost)

class Fact:
    # Assume the value of id will be final from the beginning
    def __init__(self, id, pre_op = None, add_op = None, cost = float("inf"), h = float("inf")):
        self.id = id
        if pre_op is None:
            self.pre_op = set()
        else:
            self.pre_op = pre_op
        if add_op is None:
            self.add_op = set()
        else:
            self.add_op = add_op
        self.cost = cost
        self.index = None
        self.n0 = False
        self.nX = False

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    def insert_pre(self, op):
        self.pre_op.add(op)

    def insert_add(self, op):
        self.add_op.add(op)

    def del_pre(self, op):
        self.pre_op.remove(op)

    def del_add(self, op):
        self.add_op.remove(op)

    def reset(self):
        self.index = None
        self.n0 = False
        self.nX = False
        self.cost = float("inf")

    def p(self):
        return (self.index, self.n0, self.nX, self.cost)

class State():
    def __init__(self, F, O = None, priority = float("inf"), h = float("inf")):
        self.F = frozenset(F)
        if O is None:
            self.O = self.available_operations()
        else:
            self.O = O
        self.priority = priority
        self.h = h
        self.index = None

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.h <= other.h
        return self.priority < other.priority

    def __le__(self, other):
        if self.priority == other.priority:
            return self.h <= other.h
        return self.priority <= other.priority

    def __gt__(self, other):
        if self.priority == other.priority:
            return self.h >= other.h
        return self.priority > other.priority

    def __ge__(self, other):
        if self.priority == other.priority:
            return self.h >= other.h
        return self.priority >= other.priority

    def __eq__(self, other):
        return other.F.issubset(self.F)

    def __ne__(self, other):
        return self.F != other.F

    def __hash__(self):
        return hash(self.F)

    def available_operations(self):
        O = set()
        for s in self.F:
            for o in s.pre_op:
                if o.pre.issubset(self.F):
                    O.add(o)
        return O

class PlanningTask:
    def __init__(self, F, O, s_init, s_goal):
        self.F = F
        self.O = O
        self.s_init = s_init
        self.s_goal = s_goal

    def reset(self):
        for f in self.F:
            f.reset()
        for o in self.O:
            o.reset()

    def remove_o(self, op):
        for f in op.pre:
            f.del_pre(op)
        for f in op.add:
            f.del_add(op)

    def remove_f(self, f):
        for op in f.pre_op:
            op.pre.remove(f)

        for op in f.add_op:
            op.add.remove(f)

def load_planning_task(fin):
    data = fin.read().strip().split('\n')

    num_facts = int(data[0])
    F = [Fact(f) for f in data[1:num_facts + 1]]

    data = data[num_facts+1:]
    init = [F[int(i)] for i in data[0].split()]
    s_init = init[1:]

    data = data[1:]
    goal = [F[int(i)] for i in data[0].split()]
    s_goal = goal[1:]

    data = data[1:]
    num_ops = int(data[0])

    O = []
    ops = [data[x:x+5] for x in range(1, len(data), 5)]
    for name, pre, add, delete, cost in ops:
        pre = set([F[int(x)] for x in pre.split()][1:])
        add = set([F[int(x)] for x in add.split()][1:])
        delete = set([F[int(x)] for x in delete.split()][1:])
        op = Operator(name, pre, add, delete, int(cost))
        O.append(op)

    return set(F), set(O), set(s_init), set(s_goal)

def hmax(II, s, reset = False):
    if reset:
        II.reset()

    for f in II.F:
        f.cost = float("inf")
    for f in s:
        f.cost = 0
    for o in II.O:
        o.u = len(o.pre)

    C = set()
    q = PriorityQueue(list(II.F))
    while not II.s_goal.issubset(C):
        # cx = min(II.F - C)
        # ori = cx.index
        c = q.pull_highest_priority_element()
        C.add(c)
        for o in II.O:
            if c in o.pre:
                o.u -= 1
                if o.u == 0:
                    for f in o.add:
                        prev = f.cost
                        f.cost = min(f.cost, o.cost + c.cost)
                        if q.in_queue(f) and f.cost < prev:
                            q.update_key(f)

    # print([(f.id, f.cost) for f in II.s_goal])
    return max(II.s_goal).cost

def set_supporters(II):
    for o in II.O:
        o.set_supp()

def set_n0(II):
    open = []

    for t in II.s_goal:
        open.append(t)
        t.n0 = True
        t.nX = False
        t.visited_n0 = True

    while len(open) > 0:
        t = open.pop(0)
        for o in t.add_op:
            s = o.supp()

            if not s.visited_n0 and o.cost == 0 and t.n0:
                s.visited_n0 = True
                open.append(s)
                s.n0 = True
                s.nX = False

def set_nX(II):
    open = []

    for s in II.s_init:
        s.nX = True
        s.n0 = False

    for t in II.s_goal:
        open.append(t)
        t.visited_nX = True

    while len(open) > 0:
        t = open[-1]
        break_out = False
        for o in t.add_op:
            s = o.supp()
            if s.visited_nX:
                continue

            open.append(s)
            break_out = True
            s.visited_nX = True

        if not break_out:
            for o in t.add_op:
                s = o.supp()
                if s.nX and not t.n0:
                    t.n0 = False
                    t.nX = True
            open.pop()

def find_landmarks(II):
    L = set()
    for o in II.O:
        s = o.supp()
        for t in o.add:
            if s.nX and t.n0:
                L.add(o)
    return L

def reset_nodes_for_landmarks(II):
    for f in II.F:
        f.n0 = False
        f.nX = False
        f.visited_n0 = False
        f.visited_nX = False

def h_lmcut(II, s, reset = True):
    if reset:
        II.reset()

    if II.s_goal.issubset(s):
        return 0

    h = 0
    I, G = Fact("INIT"), Fact("GOAL")
    oI, oG = Operator('oI', set([I]), s, set(), 0), Operator('oG', II.s_goal, set([G]), set([]), 0)
    II_i = PlanningTask(II.F.union(set([I, G])), II.O.union(set([oI, oG])), set([I]), set([G]))
    while hmax(II_i, II_i.s_init) is not 0:
        reset_nodes_for_landmarks(II_i)
        set_supporters(II_i)
        set_n0(II_i)
        set_nX(II_i)
        L = find_landmarks(II_i)
        print([l.id for l in L])
        if len(L) == 0:
            h = float("inf")
            break
        m = min(L).cost
        if m == 0:
            break
        if m is not None:
            h += m
            for o in L:
                o.cost -= m

    II.reset()
    II.remove_o(oI)
    II.remove_o(oG)
    II.remove_f(I)
    II.remove_f(G)
    return h

def a_star(II, h_function):
    s_init = State(II.s_init)
    s_init.h = 0
    frontier = PriorityQueue()
    # frontier = [s_init]
    frontier.insert_with_priority(s_init)
    path = {}
    optimal_cost = {}
    path[s_init] = None
    optimal_cost[s_init] = 0
    
    best = float("inf")
    closed = []

    # while len(frontier) > 0:
    while not frontier.is_empty():
        current = frontier.pull_highest_priority_element()
        # current = min(frontier)
        # frontier.remove(current)
        # print(current.h)
        # print((current.priority, current.h, optimal_cost[current]))
        # 
        # if II.s_goal.issubset(current.F):
        #     print((current.h, optimal_cost[current]))

        # if current.h > best: # Can do because heuristics are admissible; ie, h' <= h*
            # if (II.s_goal.issubset(current.F)):
            # print(current.h)
            # continue
        # print(len(closed))

        if II.s_goal.issubset(current.F):
            break
            # if best > optimal_cost[current]:
            #     # print((current.h, optimal_cost[current]))
            #     best = optimal_cost[current]
            #     break
            # continue
        
        closed.append(current)
        # print([f.id for f in current.F])

        for o in current.O: # Operations available, for which the set of facts active in this node fulfill their preconditions
            start = time.time()
            next = State(current.F.union(o.add) - o.delete) 
            end = time.time()
            # print("State: %f" % (end - start))
            # next.O = available_operations(next.F)
            new_cost = optimal_cost[current] + o.cost
            # Calculating successor nodes heuristic value is important because 
            # LMCut is consistent, therefore h(current) >= cost(current -> succ) + h(succ)
            break_out = False
            start = time.time()

            if next not in optimal_cost:
                # if h_function == hmax:
                #     for state in closed:
                #         if next.F.issubset(state.F):
                #             break_out = True
                #             break
                if not break_out:
                    optimal_cost[next] = new_cost
                    h = h_function(II, next.F)
                    next.priority = new_cost + h
                    next.h = h
                    frontier.insert_with_priority(next)
                    # frontier.append(next)
                    path[next] = {"o": o, "prev": current}
            elif new_cost < optimal_cost[next]:
                optimal_cost[next] = new_cost
                h = h_function(II, next.F)
                next.priority = new_cost + h
                next.h = h
                frontier.insert_with_priority(next)
                # frontier.append(next)
                path[next] = {"o": o, "prev": current}
            end = time.time()
            # print("Rest: %f" % (end - start))

    print(len(closed))
    return (path, optimal_cost) + get_final_state_and_cost(II, path, optimal_cost)

def get_final_state_and_cost(II, path, result):
    candidates = [s for s in path.keys() if II.s_goal.issubset(s.F)]
    candidates_costs = [result[s] for s in candidates]
    if len(candidates) == 0:
        return None, None

    index_min = min(range(len(candidates_costs)), key=candidates_costs.__getitem__)

    return candidates[index_min], candidates_costs[index_min]

def main(fn_strips, fn_fdr):
    with open(fn_strips, 'r') as fin:
        F, O, s_init, s_goal = load_planning_task(fin)

    II = PlanningTask(F, O, s_init, s_goal)
    # print("h^max for init: " + str(hmax(II, II.s_init, True)))
    # print("h^lmcut for init: " + str(h_lmcut(II, II.s_init)))
    zeros = 0
    ones = 0
    for o in II.O:
        if o.cost == 0:
            zeros += 1
        elif o.cost == 1:
            ones += 1

    h_function = hmax if zeros > 0 and zeros + ones == len(II.O) else h_lmcut
    # print("hmax" if zeros > 0 and zeros + ones == len(II.O) else "h_lmcut")
    # print(sorted([len(o.add) for o in II.O]))
    # print(sorted([o.cost for o in II.O]))
    path, result, final, optimal_cost = a_star(II, h_function)
    sequence = list()
    current = final
    while True and current is not None:
        if path[current] is None:
            break
        sequence.append(path[current]['o'].id)
        current = path[current]['prev']
    print(";; Cost: " + str(optimal_cost))
    print(";; Init: " + str(hmax(II, II.s_init, True)))
    print("")
    print("\r\n".join(["(" + action + ")" for action in reversed(sequence)]))

 
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {0} problem.strips problem.fdr'.format(sys.argv[0]))
        sys.exit(-1)
 
    main(sys.argv[1], sys.argv[2])
