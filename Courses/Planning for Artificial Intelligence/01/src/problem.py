
class StripsOperator(object):
    def __init__(self, name, pre, add_eff, del_eff, cost):
        self.name = '(' + name + ')'
        self.pre = set(pre)
        self.add_eff = set(add_eff)
        self.del_eff = set(del_eff)
        self.cost = cost

class Strips(object):
    def __init__(self, path):
        with open(path, 'r') as fin:
            self._load(fin)

    def _load(self, fin):
        data = fin.read().strip().split('\n')

        num_facts = int(data[0])
        self.facts = data[1:num_facts+1]

        data = data[num_facts+1:]
        init = [int(x) for x in data[0].split()]
        self.init = init[1:]

        data = data[1:]
        goal = [int(x) for x in data[0].split()]
        self.goal = goal[1:]

        data = data[1:]
        num_ops = int(data[0])

        self.operators = []
        ops = [data[x:x+5] for x in range(1, len(data), 5)]
        for name, pre, add_eff, del_eff, cost in ops:
            pre = [int(x) for x in pre.split()][1:]
            add_eff = [int(x) for x in add_eff.split()][1:]
            del_eff = [int(x) for x in del_eff.split()][1:]
            op = StripsOperator(name, pre, add_eff, del_eff, int(cost))
            self.operators += [op]


class FDRVar(object):
    def __init__(self, names):
        self.range = len(names)
        self.names = names

class FDRPartState(object):
    def __init__(self, pairs):
        self.facts = []
        for i in range(0, len(pairs), 2):
            self.facts += [(pairs[i], pairs[i+1])]

class FDROperator(object):
    def __init__(self, name, pre, eff, cost):
        self.name = name
        self.pre = pre
        self.eff = eff
        self.cost = cost

class FDR(object):
    def __init__(self, path):
        with open(path, 'r') as fin:
            self._load(fin)

    def _load(self, fin):
        data = fin.read().strip().split('\n')

        self.vars = []
        num_vars = int(data[0])
        data = data[1:]
        for i in range(num_vars):
            num_values = int(data[0])
            names = data[1:num_values+1]
            self.vars += [FDRVar(names)]
            data = data[num_values+1:]

        self.init = [int(x) for x in data[0].split()]
        self.goal = FDRPartState([int(x) for x in data[1].split()[1:]])

        num_ops = int(data[2])
        data = data[3:]

        self.operators = []
        ops = [data[x:x+4] for x in range(0, len(data), 4)]
        for name, pre, eff, cost in ops:
            pre = FDRPartState([int(x) for x in pre.split()[1:]])
            eff = FDRPartState([int(x) for x in eff.split()[1:]])
            cost = int(cost)
            self.operators += [FDROperator(name, pre, eff, cost)]
