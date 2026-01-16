import numpy as np
import numpy.random as rd
import pandas as pd
from collections import Counter
#arms k between 1 and K
#experts n between 1 and N
#steps t between 1 and T

class Bandit():
    def __init__(self,
                 n_experts,
                 k_arms,
                 t_steps):
        
        self.N = n_experts
        self.K = k_arms
        self.T = t_steps
        self.chosen_arms = []
        self.eta = np.ones(self.T) #temperature parameter, default to ones
        self.q = np.ones((self.T,self.N))/self.N #trust distribution for experts N at time T, starts as uniform
        self.loss_arms = []
        #basket components depends on the mean price of each product
        self.basket = [2, 1 , 1, 1, 3, 5, 4]

    def experts(self, data, t):
        '''function to get expert advice at time t, must return expert array of size (N x K)
        first dim is expert at given time, 2nd is distribution over arms for given expert at given time'''
        expert = np.zeros((self.N, self.K))
        data_t = data[data['SHOP_WEEK'] == t]
        uniform = np.ones(self.K) / self.K

        #expert 0 advises uniformly the stores not visited yet (or uniform if all stores have been visited)
        
        if t>0:
            unknown = np.arange(self.K)
            for tau in range(len(self.chosen_arms)):
                unknown[self.chosen_arms[tau]] = 0
            if np.sum(unknown) != 0:
                expert[0] = unknown / np.sum(unknown)
            else: expert[0] = uniform
        else:
            expert[0] = uniform

        #expert 1 advises the frequency each store has already been visited (or uniform at t=0)
        arms = np.arange(self.K)
        
        if np.sum(arms) != 0 and t>0:
            for tau in range(len(self.chosen_arms)):
                arms[self.chosen_arms[tau]] += 1
            expert[1] = arms / np.sum(arms)
        else:
            expert[1] = uniform

        #expert 2 advises according to advertisements for the price of an item of the basket
        prices = np.array(data_t['PRD0900173'])
        expert[2] = prices/np.sum(prices)

        #expert 3 advises for the store with smallest loss yet
        if t > 0:
            t_min = np.argmin(self.loss_arms)
            expert[3,self.chosen_arms[t_min]] = 1
        else:
            expert[3] = uniform

        #expert 4 advises at random (uniform distribution)
        expert[4] = uniform

        return expert


    def loss(self, data, k, t):
        #return l_kt loss for chosing the arm k at time t
        data_t = data[data['SHOP_WEEK'] == t]

        price_all_stores = np.array(data_t[['PRD0900173', 'PRD0900531', 'PRD0900679', 'PRD0901265', 'PRD0901878', 'PRD0902540', 'PRD0903052']])
        price_all_baskets = np.dot(price_all_stores,self.basket)

        price_k = price_all_baskets[k]
        best_price = np.min(price_all_baskets)
        loss = price_k - best_price
        return loss


    def step_exp4(self, t, Y_cum, data):
        xi = self.experts(data, t) #array N x K of experts advice on arms
        print(xi)
        p = [np.dot(xi[:,k], self.q[t]) for k in range(self.K)] #get a distribution over arms from advices xi weighted by trust q
        #choose an arm according to p
        It = rd.choice(a = np.arange(self.K), p = p)

        #estimated losses for arms
        Lt = np.zeros(self.K)
        Lt[It] =self.loss(data, It,t)/p[It]
        self.loss_arms.append(Lt[It])

        #estimated losses for experts
        yt = [np.dot(xi[n], Lt) for n in range(self.N)]
        #cumulative losses for experts
        Y_cum = np.add(Y_cum,yt)

        #update trust q
        self.q[t] = np.exp(-self.eta[t] * Y_cum)/np.sum(np.exp(-self.eta[t] * Y_cum))
        return It, Y_cum

    def exp4(self, data):
        
        Y_cum = np.zeros(self.N) #cumulative loss for experts

        for t in range(self.T):
            It, Y_cum = self.step_exp4(t, Y_cum, data)
            self.chosen_arms.append(It)

        return self.chosen_arms