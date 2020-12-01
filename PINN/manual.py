# Data is (N, M) where M is features, N is number of rows or datapoints

import numpy as np
data = np.load('superclean.npy')

charge_steps = 1
profits = [[(data[:,i:i+3].sum(1) - data[:,j:j+charge_steps].sum(1)).mean() for j in range(i+1+charge_steps, data.shape[1])] for i in range(data.shape[1])]

min_i = -1
min_j = -1
min_neg_profit = 10e10

for i in range(len(profits)):
    for j in range(len(profits[i])):
        if profits[i][j] < min_neg_profit:
            min_neg_profit = profits[i][j]
            min_i = i
            min_j = i + j + 1 + charge_steps
            
# Always buy at time i, always sell at time j
# Let D be the maximum amount of energy our battery can store
D = 14/1000
daily_profits = D * (data[:,min_j:min_j+charge_steps].mean(1) - data[:,min_i:min_i+charge_steps].mean(1))

# Graphing
import util
util.profit_hist(daily_profits)

