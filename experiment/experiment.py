import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data.load_data import load_data
from models.exp4 import Bandit

np.random.seed(0)

df = load_data()

nb_stores = len(pd.unique(df['STORE_CODE']))
nb_weeks = len(pd.unique(df['SHOP_WEEK']))
bandit = Bandit(n_experts=5, k_arms= nb_stores, t_steps=nb_weeks)
path_taken = bandit.exp4(df)

fig, axs = plt.subplots(3,1)

t_axis = range(nb_weeks)
#plot loss over time
axs[0].plot(t_axis, bandit.loss_arms)
axs[0].set_title("Estimated loss for arms over time")

#plot trust in experts over time
for i in range(5):
    axs[1].plot(t_axis, bandit.q[:,i], label = 'expert ' + str(i))
axs[1].legend()
axs[1].set_title("Trust in experts q over time")

#plot number of time store has been chosen at time t over t
stores_freq = np.zeros((nb_stores, nb_weeks))
for i in range(1,len(path_taken)):
    indic = np.zeros(nb_stores)
    indic[path_taken[i]] +=1
    stores_freq[:,i] = stores_freq[:, i-1] + indic
for i in range(nb_stores):
    axs[2].plot(t_axis, stores_freq[i], label = 'store ' + str(i))
axs[2].legend()
axs[2].set_title("Number of time each store has been chosen by time t")

plt.subplots_adjust(hspace = 0.3)
plt.show()