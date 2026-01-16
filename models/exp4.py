import numpy as np
import numpy.random as rd
import pandas as pd
from collections import Counter
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
        self.q = np.ones((self.T,self.K))/self.K #trust distribution for experts K at time T, starts as uniform
        self.loss_arms = np.zeros(self.T)

    def experts(self, data, t):
        '''function to get expert advice at time t, must return expert array of size (N x K)
        first dim is expert at given time, 2nd is distribution over arms for given expert at given time'''
        expert = np.zeros((self.N, self.K))
        #expert 0 advise uniformly the stores not visited yet (or uniform if all stores have been visited)
        unknown = np.arange(self.K)
        for tau in range(len(self.chosen_arms)):
            unknown[self.chosen_arms[tau]] = 0
        if np.sum(unknown) != 0:
            expert[0] = unknown / np.sum(unknown)
        else:
            expert[0] = np.ones(self.K) / self.K

        #expert 1 advise the frequence each store has already been visited (or uniform at t=0)
        arms = np.arange(self.K)
        for tau in range(len(self.chosen_arms)):
            arms[self.chosen_arms[tau]] += 1
        if np.sum(arms) != 0:
            expert[1] = arms / np.sum(arms)
        else:
            expert[1] = np.ones(self.K) / self.K

        #expert 2 advise according to advertisements for the price of an item of the basket
        data[]

        #expert 3 for the store with smallest loss yet
        t_min = np.argmin(self.loss_arms)
        expert[3,self.chosen_arms[t_min]] = 1

        #expert 4 at random (uniform distribution)
        expert = np.ones(self.K) / self.K




    def loss(self, k, t):
        #return l_kt loss for chosing the arm k at time t

    def step_exp4(self, t, Y_cum, data):
        xi = self.experts(t, data) #array N x K of experts advice on arms
        p = [np.dot(xi[:,k], self.q) for k in range(self.K)]/self.N #get a distribution over arms from advices xi weighted by trust q
        #choose an arm according to p
        It = rd.choice(a = np.arange(self.K), p = p)

        #estimated losses for arms
        Lt = np.zeros(self.K)
        Lt[It] =self.loss(It,t)/p[It]
        self.loss_arms[t] = Lt[It] 
        #estimated losses for experts
        yt = [np.dot(xi[n], Lt) for n in range(N)]/self.K
        #cumulative losses for experts
        Y_cum = np.a(Y_cum,yt)

        #update trust q
        self.q[t] = np.exp(-self.eta[t] * Y_cum)/np.sum(np.exp(-self.eta[t] * Y_cum))
        return It, Y_cum

    def exp4(self):
        
        Y_cum = np.zeros(self.K) #cumulative loss for experts

        for t in range(self.T):
            It, Y_cum = self.step_exp4(t, Y_cum)
            self.chosen_arms.append(It)

        return(self.chosen_arms, Y_cum)