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
        self.experts = np.empty((self.T, self.N, self.K))
        self.eta = np.ones(self.T) #temperature parameter, default to ones

    def experts(self, data):
        #function to get expert advice, must set values in self.expert (T x K x N array)
        #first dim is time, 2nd is expert at given time, 3rd is distribution over arms for given expert at given time

    def loss(self, k, t):
        #return l_kt loss for chosing the arm k at time t

    def step_exp4(self, t, Y_cum, q):
        #q distribution over experts, represent how much we trust them
        xi = self.experts()[t] #array N x K of experts advice on arms
        p = [np.dot(xi[:,k],q) for k in range(self.K)]/self.N #get a distribution over arms from advices xi weighted by trust q
        It = rd.multinomial(1,p)

        #estimated losses for arms
        Lt = [self.loss(k,t) for k in range(K)]
        Lt[It] = Lt[It]/p[It]

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
        chosen_arms = []

        for t in range(self.T):
            It, Y_cum, q = self.step_exp4(t, Y_cum, q)
            chosen_arms.append(It)
            
        return(It, Y_cum, q)