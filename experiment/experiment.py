import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data.load_data import load_data
from models.exp4 import Bandit

np.random(seed)

df = load_data()

nb_stores = len(pd.unique(df['STORE_CODE']))
nb_weeks = len(pd.unique(df['SHOP_WEEK']))
bandit = Bandit(n_experts=5, k_arms= nb_stores, t_steps=nb_weeks)
path_taken = bandit.exp4(df)

fig, axs = plt.subplots(3,1)

t = range(nb_weeks)
axs[0].plot(t, bandit.loss_arms)



for i in range(5):
    axs[1].plot(t, bandit.q[:,i], label = i)
    axs[1].legend()

axs[2].plot(t)
plt.show()