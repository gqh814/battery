"""
to do:
- perfect foresight 
- copy action space from Zheng (2022)
- add parameters (charge/discharge rate, efficiency)
- add lifetime ( 1 year ? * lifetime)

"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
#%%
dk1_p = pd.read_csv('../data/entsoe_price_DK_1_20150101_20240101.csv') #, sep=';', decimal=','
dk1_p
#%% 
dk1_p['date'] = pd.to_datetime(dk1_p['date'], utc=True)  # Handles timezones too
dk1_p['year'] = dk1_p['date'].dt.year

dk1_p_training =  dk1_p[dk1_p['year'] == 2017]
dk1_p_test =  dk1_p[dk1_p['year'] == 2018]

prices_training =  dk1_p_training.DK_1_day_ahead_price.to_numpy()
prices_test =  dk1_p_test.DK_1_day_ahead_price.to_numpy()
assert len(prices_training) == len(prices_test), "Training and test data lengths do not match."

#%%

# Set parameters
num_periods = prices_training.shape[0]
battery_capacity = 10  # Max storage capacity
initial_storage = 0  # Start at half capacity

# Generate continuous price data (random walk for realistic price behavior)
# np.random.seed(42)
# prices = np.cumsum(np.random.randn(num_periods))
# prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices)) * 10  # Scale between 0 and 10

# Define parameters
num_storage_levels = 11  # Battery levels from 0 to 10
num_price_levels = 100  # Discretize continuous prices into 20 levels
gamma = 0.99  # Discount factor
iterations = 1000  # Iterations for Value Function Iteration
tolerance = 1e-4  # Convergence criteria

# Discretize price space
price_min, price_max = np.min(prices_test), np.max(prices_test)
price_grid = np.linspace(price_min, price_max, num_price_levels)

# Initialize value function and policy
V = np.zeros((num_storage_levels, num_price_levels))  # Value function
policy = np.zeros((num_storage_levels, num_price_levels), dtype=int)  # Actions: -1 (sell), 0 (hold), +1 (buy)

# Transition function: probability of next price given current price
price_transitions = np.zeros((num_price_levels, num_price_levels))
for i in range(num_price_levels):
    for j in range(num_price_levels):
        price_transitions[i, j] = np.exp(-abs(price_grid[i] - price_grid[j]))  # Smooth transitions
    price_transitions[i, :] /= np.sum(price_transitions[i, :])  # Normalize to sum to 1

#%%
# Value Function Iteration
for _ in range(iterations):
    V_new = np.copy(V)

    for s in range(num_storage_levels):
        for p in range(num_price_levels):
            current_price = price_grid[p]

            # Possible actions: hold, buy, sell
            actions = []
            if s > 0:  # Can sell
                actions.append((current_price + gamma * np.sum(price_transitions[p, :] * V[s - 1, :]), -1))
            actions.append((gamma * np.sum(price_transitions[p, :] * V[s, :]), 0))  # Hold
            if s < num_storage_levels - 1:  # Can buy
                actions.append((-current_price + gamma * np.sum(price_transitions[p, :] * V[s + 1, :]), +1))

            # Choose best action
            best_value, best_action = max(actions)
            V_new[s, p] = best_value
            policy[s, p] = best_action

    # Convergence check
    if np.max(np.abs(V_new - V)) < tolerance:
        print('converged after ', _ + 1, 'iterations')
        break
    V = V_new

#%% Simulate battery operation using optimal policy

# initialize variables
battery_storage_sim = np.zeros(num_periods)
profit_sim = np.zeros(num_periods)
storage = 0  # Start at half capacity

for t in range(num_periods):
    # Find closest price index in grid
    price_idx = np.argmin(np.abs(price_grid - prices_test[t]))

    # Take optimal action
    action = policy[storage, price_idx]
    if action == 1 and storage < battery_capacity:  # Buy
        storage += 1
        profit_sim[t] = profit_sim[t - 1] - prices_test[t] if t > 0 else -prices_test[t]
    elif action == -1 and storage > 0:  # Sell
        storage -= 1
        profit_sim[t] = profit_sim[t - 1] + prices_test[t] if t > 0 else prices_test[t]
    else:  # Hold
        profit_sim[t] = profit_sim[t - 1] if t > 0 else 0

    battery_storage_sim[t] = storage



# Plot Battery Storage Evolution
plt.figure(figsize=(12, 5))
plt.scatter(range(num_periods), battery_storage_sim, c=prices_test, cmap="coolwarm", edgecolors="k", label="Battery Storage")
plt.plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")  # Light line for trend
plt.colorbar(label="Price Level (Continuous)")
plt.xlabel("Time Periods")
plt.ylabel("Battery Storage Level")
plt.title("Battery Storage Evolution Over Time (Value Function Iteration)")
plt.grid()
plt.show()

# Plot Price Evolution
plt.figure(figsize=(12, 5))
plt.plot(prices_test, color="orange", linestyle="-", markersize=4, label="test", alpha=0.5)
plt.plot(prices_training, color="red", linestyle="-", markersize=4, label="train", alpha=0.5)
# plt.plot(moving_avg, color="blue", linestyle="-", marker="o", markersize=4, label="MA")
plt.xlabel("Time Periods")
plt.ylabel("Profit")
plt.title("Prices")
plt.legend()
plt.grid()
plt.show()


# Plot Cumulative Profit Evolution
plt.figure(figsize=(12, 5))
plt.plot(profit_sim, color="green", linestyle="-", marker="o", markersize=4, label="Cumulative Profit")
plt.xlabel("Time Periods")
plt.title("Cumulative Profit Evolution Over Time")
plt.legend()
plt.grid()
plt.show()

# # Plot Cumulative Profit Evolution
# plt.figure(figsize=(12, 5))
# plt.plot(profit, color="green", linestyle="-", marker="o", markersize=4, label="Cumulative Profit")
# plt.xlabel("Time Periods")
# plt.ylabel("Profit")
# plt.title("Prices Evolution Over Time")
# plt.legend()
# plt.grid()
# plt.show()


#%#
# montecarlo, træk parameter ud.
# 
# 
# %%
profit_sim.any() < 0

# %%
battery_storage_sim[battery_storage_sim > 0]
# %%
