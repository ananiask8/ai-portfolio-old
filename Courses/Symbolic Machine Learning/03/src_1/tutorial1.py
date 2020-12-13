import bridge
from dataset import InterpretationDataset
from logic import *


def withoutBidirectionalEdge(clause: Clause) -> bool:
    '''
    Returns true if the clause contains two literals among which there are bidirectional edges, e.g. {edge(1,2) | edge(2,1)}; false otherwise.

    :type clause: Clause
    :rtype: bool
    '''
    return not any(
        Literal(Atom(literal.atom.predicate, literal.atom.terms[::-1]), literal.positive) in clause.literals for literal
        in
        clause.literals if literal.atom.predicate.arity == 2)


def session(pathToData: str, s: int, t: int, showMostSpecificCNF=False, showOnlyDirectedGraphs=True):
    '''
    Runs the session given path to dataset, s and t constrains, and agent class.

    Prints the most specific CNF if showMostSpecificCNF is set to True.
    Prints only clauses corresponding to directed graphs if showOnlyDirectedGraphs is set to True.

    :type pathToData: string
    :type s: int,>=0
    :type t: int,>=0
    :type showMostSpecificCNF: bool
    :type showOnlyDirectedGraphs: bool
    :rtype: None
    '''
    dataset = InterpretationDataset(pathToData)
    cnf = bridge.generateMostSpecificSTRangeRestrictedClauses(dataset.getPredicates(), s, t)
    # if you want to generate the most specific rr-st-clause with constants as well, add the following parameter to the previous call
    # constants=dataset.getConstants()

    if showMostSpecificCNF:
        print('most specific CNF consists of the following {} literals'.format(len(cnf.clauses)))
        print('clause by clause')
        for c in cnf.clauses:
            print('\t{}'.format(c))

    print('agent initialization')
    agent = GeneralizingAgent(cnf)

    for sample in dataset:
        print('feeding agent with {}'.format(sample))
        agent.feedSample(sample)

    print('end of observation feeding, agent has hypothesis of {} disjunctions:\n{}'.format(
        len(agent.getHypothesis().clauses), agent.getHypothesis()))

    hypothesis = agent.getHypothesis()
    if showOnlyDirectedGraphs:
        print('-', 'showing only directed graphs')
        hypothesis = CNF(filter(withoutBidirectionalEdge, hypothesis.clauses))

    print('\nfinal learned hypothesis is')
    print(hypothesis)

    print('\ndisjunction by disjunction ')
    for clause in hypothesis:
        print("\t{}".format(clause))


if __name__ == "__main__":
    #pathToDataset = os.sep.join([".", "data", "threeChain"]) # negative class si generated from e(X,Y),e(Y,Z),e(Z,W), , s=t=3 is sufficient
    pathToDataset = os.sep.join([".", "data", "lectureGraphs"])  # negative class is generated from triangles, s=t=3 is sufficient
    s = 3  # maximal number of literals in a clause
    t = 3  # maximal number of symbols in a literal

    from agent import GeneralizingAgent


    session(pathToDataset, s, t, showMostSpecificCNF=True, showOnlyDirectedGraphs=True)
