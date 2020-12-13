from logic import *
from dataset import Sample


class GeneralizingAgent:
    '''
    Your task is to implement this class which should behave as generalizing algorithm from the lecture (see section 4.1 
    Generalizing Agent in SMU textbook version of year 2017).

    Firstly, the agent is initialized with the most specific hypothesis (CNF) generated from given st-range 
    restriction constrains; that is his hypothesis. When the agent gets a sample (by calling method feedSample(Sample)),
    he should get his hypothesis consistent with the sample using only polynomial number of steps.

    The agent may be asked (using the method getHypothesis()) to return current hypothesis.
    '''

    def __init__(self, hypothesis: CNF):
        '''
        Construct the agent with a hypothesis.

        :type hypothesis:
        :rtype: GeneralizingAgent
        '''
        self.cnf = hypothesis

    def feedSample(self, sample: Sample) -> None:
        '''
        After receiving the sample, agent should get his hypothesis consistent to given sample/observation.

        :type sample: Sample
        :rtype: None
        '''
        ...
        # your code here
        pass

    def getHypothesis(self) -> CNF:
        '''
        Returns agent hypothesis (a CNF).

        :rtype: CNF
        '''
        return self.cnf
