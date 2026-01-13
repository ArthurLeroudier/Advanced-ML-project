import numpy as np
import numpy.random as rd
import pandas as pd
#arms k between 1 and K
#experts n between 1 and N
#steps t between 1 and T

class Bandit(n_experts, k_arms, t_steps):
    def __init__(self,
                 n_experts,
                 k_arms,
                 t_steps):
        
        self.N = n_experts
        self.K = k_arms
        self.T = t_steps
        self.chosen_arms = []
        self.eta = np.ones(self.T) #temperature parameter, default to ones

    def experts(self, data, t):
        #function to get expert advice at time t, must return expert array of size (N x K)
        #first dim is expert at given time, 2nd is distribution over arms for given expert at given time

    def loss(self, k, t):
        #return l_kt loss for chosing the arm k at time t

    def step_exp4(self, t, Y_cum, q, data):
        #q distribution over experts, represent how much we trust them
        xi = self.experts(t, data) #array N x K of experts advice on arms
        p = [np.dot(xi[:,k],q) for k in range(self.K)]/self.N #get a distribution over arms from advices xi weighted by trust q
        It = rd.multinomial(1,p)

        #estimated losses for arms
        Lt = np.zeros(self.K)
        Lt[It] =self.loss(It,t)/p[It]

        #estimated losses for experts
        yt = [np.dot(xi[n], Lt) for n in range(N)]/self.K
        #cumulative losses for experts
        Y_cum = np.a(Y_cum,yt)

        #update trust q
        q = np.exp(-self.eta[t] * Y_cum)/np.sum(np.exp(-self.eta[t] * Y_cum))
        return It, Y_cum, q

    def exp4(self):
        q = np.ones(self.K)/self.K #trust distribution for experts, starts as uniform
        Y_cum = np.zeros(self.K) #cumulative loss for experts

        for t in range(self.T):
            It, Y_cum, q = self.step_exp4(t, Y_cum, q)
            self.chosen_arms.append(It)

        return(chosen_arms, Y_cum, q)