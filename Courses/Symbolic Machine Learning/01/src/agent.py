import numpy as np
from math import factorial
from itertools import combinations, filterfalse, chain

class Agent:
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon 
        self.delta = delta

    def map_s_interpretation(self, interpretation):
        s_interpretation = len(self.propositions) * [False]
        for i in range(0, len(self.propositions)):
            for j in range(0, len(self.propositions[i])):
                neg = '~' in self.propositions[i][j]
                idx = int(self.propositions[i][j].replace('~', ''))
                s_interpretation[i] |= ~interpretation[idx] if neg else interpretation[idx]

        return s_interpretation

    def compute_required_training_dataset_size(self):
        # TODO - compute your value here.
        # It should depend on self.n_variables and self.s, self.epsilon and self.delta.
        
        # Generate all propositions
        self.literals = [chr(ord('0') + i) for i in range(self.n_variables)] + ['~' + chr(ord('0') + i) for i in range(self.n_variables)]
        self.propositions = self.s * [0]
        for i in range(1, self.s + 1):
            self.propositions[i - 1] = list(combinations(self.literals, i))
            self.propositions[i - 1] = list(filterfalse(lambda t: len(set([elem.replace('~', '') for elem in t])) != len(t), self.propositions[i - 1]))
        self.propositions = list(chain(*self.propositions))
        self.active_propositions = len(self.propositions) * [1]
        # propositions_length = len(self.propositions)
        print(self.propositions)


        # Calculation of amount of propositions, to get required dataset size
        propositions_length = 0
        for i in range(1, self.s + 1):
            propositions_length += (2 ** i) * factorial(self.n_variables) / (factorial(self.n_variables - i) * factorial(i))
        return int(propositions_length * np.log(propositions_length / self.delta) / self.epsilon)

    def process_first_observation(self, interpretation):
        # TODO - do something with interpretation and return 
        # a prediction
        self.prev_s_interpretation = self.map_s_interpretation(interpretation)

        return False

    def predict(self, interpretation, reward):
        if reward is not None:
            # We are in training branch
            #
            # Use the reward and the previous interpretation and 
            # the previous prediction to update your model.
            # Then make a prediction for the given interpretation.
            if reward == 0: # == wrong -> make all false propositions inactive
                for i in range(0, len(self.propositions)):
                    if ~self.prev_s_interpretation[i]:
                        self.active_propositions[i] = 0

        self.prev_s_interpretation = self.map_s_interpretation(interpretation)
        prediction = True
        for i in range(0, len(self.propositions)):
            if self.active_propositions[i]:
                prediction &= self.prev_s_interpretation[i]

        return prediction

    def learned_concept(self):
        concept = ''
        for i in range(0, len(self.propositions)):
            if self.active_propositions[i]:
                if len(concept) == 0:
                    concept += '(' + ' | '.join(map(str, self.propositions[i])) + ')'
                else:
                    concept += ' & (' + ' | '.join(map(str, self.propositions[i])) + ')'

        print(concept)

    def interact_with_oracle(self, oracle_session):
        # You may alter this method as you desire,
        # but it is not required.

        self.n_variables, self.s = oracle_session.request_parameters()
        print(self.n_variables, self.s)
        m = self.compute_required_training_dataset_size()
        first_sample = oracle_session.request_dataset(m)
        prediction = self.process_first_observation(first_sample)
        interpretation, reward = oracle_session.predict(prediction)
        while oracle_session.has_more_samples():
            interpretation, reward = oracle_session.predict(prediction)
            prediction = self.predict(interpretation, reward)

        # self.learned_concept()
