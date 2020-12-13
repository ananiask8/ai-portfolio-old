from carddeck import *
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TDAgent:
    '''
    Implementaion of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    Your goal is to modify train() method to learn the state utility function.
    I.e. you need to change this agent to a passive reinforcement learning
    agent that learns utility estimates using temporal diffrence method.
    '''
    def __init__(self, env, number_of_epochs):
        self.env = env
        self.number_of_epochs = number_of_epochs

    def train(self):
        U = [0]*30
        Ns = [0]*30
        U_10 = [0]*self.number_of_epochs
        U_11 = [0]*self.number_of_epochs
        U_16 = [0]*self.number_of_epochs
        U_18 = [0]*self.number_of_epochs
        for i in range(self.number_of_epochs):
            # print(i)
            observation = self.env.reset()
            terminal = False
            reward = 0
            alpha = 0.8
            discount_factor = 0.8
            while not terminal:
                if terminal: break
                old_feature = observation.player_hand.value()
                Ns[old_feature] += 1
                action = self.make_step(observation, reward, terminal)
                observation, reward, terminal, _ = self.env.step(action)
                new_feature = observation.player_hand.value()

                alpha_s = alpha/(alpha - 1 + Ns[old_feature])
                U[old_feature] = U[old_feature] + alpha_s*(reward + discount_factor*U[new_feature] - U[old_feature])
                U_10[i] = U[10]
                U_11[i] = U[11]
                U_16[i] = U[16]
                U_18[i] = U[18]
            #self.env.render()
        self.plot_series(U_10, "U_10_TD.png") 
        self.plot_series(U_11, "U_11_TD.png") 
        self.plot_series(U_16, "U_16_TD.png") 
        self.plot_series(U_18, "U_18_TD.png") 
        for i in range(len(U)):
            print(str((i, U[i])))

    def make_step(self, observation, reward, terminal):
        return 1 if observation.player_hand.value() < 17 else 0

    def plot_series(self, arr, fileName = None):
        plt.clf()
        plt.plot(arr)
        if fileName is not None:
            plt.savefig(fileName)