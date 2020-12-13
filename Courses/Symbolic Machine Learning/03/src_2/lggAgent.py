from logic import *
from bridge import subsume
from dataset import Sample
from copy import copy

class LGGResolver:
    ''' 
    Your task is to implement the first-order generalization agent based on LGG algorithm here, as discussed in the 
    lecture (see 4.2 Generalization of Clauses in SMU textbook version of year 201/).

    The class represents an generalization agent of first-order clauses based on the LGG algorithm. He initially starts
    with no hypothesis at all. Each time he gets an observation (in form of Sample, consisting of a class and a clause, 
    by calling his method, seeObservation(Sample)), he should change his hypothesis accordingly; i.e. in the case that
    prediction of the observation by the agent's hypothesis differs from the sample class. Recall that the agent predict
    the observation as positive iff the observed clause is theta subsumed by the agent hypothesis. Also recall that we 
    assume that there is no noise in the data. 
    
    One can obtain current hypothesis of the agent (Clause) by calling getHypothesis().

    Your first task is to implement the agent/LGG algorithm here in the method seeObservation(Sample). 
    Your second task is to implement lgg with the clause reduction step, which is called by seeObservation(Sample,reduceClause=True).
    Your third task is to implement lgg with taxonomical extension. Taxonomical information about constants are given to 
    the agent by the class constructor, e.g. LGGResolver(taxonomical=info) where info is a set of literals of a form 
     'isa(dog,mammal)'. It is ensured that from this set a forest can be formed, i.e. set of rooted oriented trees. 
    '''

    def __init__(self, taxonomical: Set[Literal] = None):
        '''
        Constructs new LGGResolver.
        
        Parameter taxonomical contains set of literals describing taxonomical information about the domain. It either
        may be None, i.e. no taxonomy provided, or it consists of literal of pairs isa/2 hierarchy, e.g. isa(car, vehicle).
        It is always ensured that literals in the set describes a forest, i.e. set of rooted oriented trees.
        
        :type taxonomical : Set of Literal
        :rtype: LGGResolver
        '''
        self.taxonomical = {}
        for l in taxonomical:
            self.taxonomical[l.atom.terms[0]] = l.atom.terms[1]
        self.hypothesis: Clause = None

    def getHypothesis(self) -> Clause:
        '''
        Returns current hypothesis of the agent.
        
        :rtype: Clause 
        '''
        return self.hypothesis

    def getSelection(self, clauseA: Clause, clauseB: Clause):
        selection = set()
        for a_lit in clauseA.getPositiveLiterals():
            for b_lit in clauseB.getPositiveLiterals():
                if a_lit.atom.arity == b_lit.atom.arity and a_lit.getPredicate() == b_lit.getPredicate():
                    selection.add((a_lit, b_lit))

        for a_lit in clauseA.getNegativeLiterals():
            for b_lit in clauseB.getNegativeLiterals():
                if a_lit.atom.arity == b_lit.atom.arity and a_lit.getPredicate() == b_lit.getPredicate():
                    selection.add((a_lit, b_lit))

        return selection

    def compoundEquals(self, t1, t2):
        return isinstance(t1, CompoundTerm) and isinstance(t2, CompoundTerm) and \
            [f for f in t1.getFunctors()][0] == [f for f in t2.getFunctors()][0]

    def lgg(self, a, b):
        # print("h", a)
        # print("o", b)
        # input()
        subs_a = {}
        subs_b = {}
        if isinstance(a, CompoundTerm) and isinstance(b, CompoundTerm):
            compatible = set()
            compatible.add((a,b))
        else:
            compatible = self.getSelection(a, b)

        new_clause = Clause.parse(" ")
        for x, y in compatible:
            new_x = Clause.parse(str(x).replace("|", ",")).literals[0]
            new_y = Clause.parse(str(y).replace("|", ",")).literals[0]
            if isinstance(a, CompoundTerm) and isinstance(b, CompoundTerm):
                terms = list(zip(x.terms, y.terms))
            else:
                terms = list(zip(x.atom.terms, y.atom.terms))

            for idx in range(len(terms)):
                t1, t2 = terms[idx]
                if t2 in self.taxonomical and self.taxonomical[t2] == t1:
                    new_x.atom.terms = new_x.atom.terms[0:idx] + (t1,) + new_x.atom.terms[idx + 1:]
                    new_y.atom.terms = new_y.atom.terms[0:idx] + (t1,) + new_y.atom.terms[idx + 1:]
                elif t1 in self.taxonomical and self.taxonomical[t1] == t2:
                    new_x.atom.terms = new_x.atom.terms[0:idx] + (t2,) + new_x.atom.terms[idx + 1:]
                    new_y.atom.terms = new_y.atom.terms[0:idx] + (t2,) + new_y.atom.terms[idx + 1:]
                elif t1 in self.taxonomical and t2 in self.taxonomical and self.taxonomical[t1] == self.taxonomical[t2]:
                    new_x.atom.terms = new_x.atom.terms[0:idx] + (self.taxonomical[t1],) + new_x.atom.terms[idx + 1:]
                    new_y.atom.terms = new_y.atom.terms[0:idx] + (self.taxonomical[t1],) + new_y.atom.terms[idx + 1:]
                elif self.compoundEquals(t1, t2):
                    lgg_r = self.lgg(t1, t2).literals[0]
                    new_x.atom.terms = new_x.atom.terms[0:idx] + (lgg_r,) + new_x.atom.terms[idx + 1:]
                    new_y.atom.terms = new_y.atom.terms[0:idx] + (lgg_r,) + new_y.atom.terms[idx + 1:]
                elif isinstance(t1, CompoundTerm) and isinstance(t2, CompoundTerm) or \
                    isinstance(t1, Variable) and isinstance(t2, CompoundTerm) or \
                    isinstance(t1, CompoundTerm) and isinstance(t2, Variable) or \
                    isinstance(t1, Variable) and isinstance(t2, Variable):
                        found = False
                        for j in range(len(self.v)):
                            if self.v[j] in self.theta and self.v[j] in self.sigma and \
                                ((isinstance(t1, CompoundTerm) and self.compoundEquals(t1, self.theta[self.v[j]])) or \
                                    (not isinstance(t1, CompoundTerm) and self.theta[self.v[j]] == t1)) and \
                                ((isinstance(t2, CompoundTerm) and self.compoundEquals(t2. self.sigma[self.v[j]])) or \
                                    (not isinstance(t2, CompoundTerm) and self.sigma[self.v[j]] == t2)):
                                found = True
                                new_x.atom.terms = new_x.atom.terms[0:idx] + (self.v[j],) + new_x.atom.terms[idx + 1:]
                                new_y.atom.terms = new_y.atom.terms[0:idx] + (self.v[j],) + new_y.atom.terms[idx + 1:]
                                break
                        if not found:
                            self.v.append(Variable(self.baseVar + str(len(self.v))))
                            self.theta[self.v[-1]] = t1
                            self.sigma[self.v[-1]] = t2
                            new_x.atom.terms = new_x.atom.terms[0:idx] + (self.v[-1],) + new_x.atom.terms[idx + 1:]
                            new_y.atom.terms = new_y.atom.terms[0:idx] + (self.v[-1],) + new_y.atom.terms[idx + 1:]
                else:
                    found = False
                    for j in range(len(self.v)):
                        if self.v[j] in self.theta and self.v[j] in self.sigma and \
                            self.theta[self.v[j]] == t1 and self.sigma[self.v[j]] == t2:
                            new_x.atom.terms = new_x.atom.terms[0:idx] + (self.v[j],) + new_x.atom.terms[idx + 1:]
                            new_y.atom.terms = new_y.atom.terms[0:idx] + (self.v[j],) + new_y.atom.terms[idx + 1:]
                            found = True
                            break
                    if not found:
                        self.v.append(Variable(self.baseVar + str(len(self.v))))
                        self.theta[self.v[-1]] = t1
                        self.sigma[self.v[-1]] = t2
                        new_x.atom.terms = new_x.atom.terms[0:idx] + (self.v[-1],) + new_x.atom.terms[idx + 1:]
                        new_y.atom.terms = new_y.atom.terms[0:idx] + (self.v[-1],) + new_y.atom.terms[idx + 1:]

            if new_x == new_y: new_clause = new_clause.extend(new_x)

        return new_clause

    def reduceHypothesis(self, h):
        delta = Clause([lit for lit in h.literals])
        for l in h.getPositiveLiterals() | h.getNegativeLiterals():
            delta_literals = list(delta.literals)
            if len(delta_literals) == 1: break
            delta_literals.remove(l)
            temp = Clause([lit for lit in delta_literals])
            if subsume(h, temp): delta = temp
        return delta

    def seeObservation(self, sample: Sample, reduceClause: bool = False) -> None:
        '''
        Performs LGG with the current hypothesis stored in the agent iff the the sample has positive label but the agent does predict the opposite class for the sample given.
        
        If reduction is set to True, then the agent process also the reduction step. You do not have to implement the 
        whole functionality, i.e. subsumption engine. To test whether one clause subsumes another one, 
        e.g. \alpha \subseq_{\theta} \beta, use library method subsume from package logic, e.g. subsume(\alpha,\beta).   

        
        :type sample: Sample
        :type reduceClause : bool
        :rtype: None 
        '''
        if not sample.positiveClass: return

        h = self.getHypothesis()
        if h is None:
            h = sample.data
            self.hypothesis = h if not reduceClause else self.reduceHypothesis(h)
            return

        self.baseVar = "V" if str(h).find("V") == -1 else "W"
        self.v = [Variable(self.baseVar + str(0))]
        if not subsume(h, sample.data):
            a, b = h, sample.data
            i, self.theta, self.sigma = 0, dict(), dict()
            h = self.lgg(a, b)
            self.hypothesis = h if not reduceClause else self.reduceHypothesis(h)
