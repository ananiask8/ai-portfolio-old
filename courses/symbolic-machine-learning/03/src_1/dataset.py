from logic import *


class ClausesDataset:
    '''
    Storage of clauses; use .clauses to get list of clauses (:type: Clause).
    '''

    def __init__(self, pathToFile: str):
        '''
        Creates new dataset of clauses from path to the file. The clauses can be accessed by .clauses.
        
        :param pathToFile: str 
        '''
        jClauses = JSMU.parseClauses(pathToFile.encode("UTF-8"))
        self.clauses = tuple(toPython(jClauses.get(idx)) for idx in range(0, jClauses.size()))

    def __iter__(self) -> Iterable[Clause]:
        return iter(self.clauses)

    def __len__(self):
        return len(self.clauses)


class Sample:
    '''
    Stores a sample which consist of a class label and an interpretation.

    Class is either positive '+' or negative '-'.
    Delimiter ':' splits class and the interpretation which is ended by a dot.

    use .positiveClass to get T if the sample has positive label 
    use .interpretation to get interpretation stored in the sample (:type: Interpretation)
    '''

    def __init__(self, interpretation: Interpretation, positiveClass: bool):
        self.interpretation: Interpretation = interpretation
        self.positiveClass: bool = positiveClass

    def __str__(self):
        return "{} : {}".format("+" if self.positiveClass else "-", self.interpretation)

    def __iter__(self) -> Iterable[Atom]:
        return iter(self.interpretation)

    def __len__(self):
        return len(self.interpretation)

    def getPredicates(self) -> Set[Predicate]:
        '''
        Returns set of predicates.

        :rtype: set of Predicate
        '''
        return self.interpretation.getPredicates()

    def getConstants(self) -> Set[Constant]:
        '''
        Returns set of constants.

        :rtype: set of Constant
        '''
        return self.interpretation.getConstants()

    def getFunctors(self) -> Set[Functor]:
        '''
        Returns set of functors.

        :rtype: set of Functor
        '''
        return self.interpretation.getFunctors()


class InterpretationDataset:
    '''
    Storage of samples. Use .samples to get list of samples (:type: Sample).
    '''

    def __init__(self, path: str):
        med = JSMU.loadMED(path.encode('utf-8'), JMatching.THETA_SUBSUMPTION)

        targets = med.getTargets()
        examples = med.getExamples().toArray()

        self.med: JMED = med
        self.samples: List[Sample] = [Sample(Interpretation(toPython(examples[idx]).literals), targets[idx] > 0.5) for
                                      idx in
                                      range(0, med.size())]

    def __iter__(self) -> Iterable[Sample]:
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def getPredicates(self) -> Set[Predicate]:
        '''
        Returns set of all predicates used in the dataset.

        :rtype: set of Predicate
        '''
        return tools.unionSets(map(lambda sample: sample.getPredicates(), self.samples))

    def getConstants(self) -> Set[Constant]:
        '''
        Returns all constants in the dataset.

        :rtype: set of Constant
        '''
        return tools.unionSets(map(lambda sample: sample.getConstants(), self.samples))

    def getFunctors(self) -> Set[Functor]:
        '''
        Returns all functors in the dataset.
        
        :rtype: set of Functor 
        '''
        return tools.unionSets(map(lambda sample: sample.getFunctors(), self.samples))
