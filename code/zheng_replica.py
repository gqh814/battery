#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
dk1_p = pd.read_csv('../data/entsoe_price_DK_1_20150101_20240101.csv')
dk1_p['date'] = pd.to_datetime(dk1_p['date'], utc=True)
dk1_p['year'] = dk1_p['date'].dt.year
# dk1_p_training = dk1_p[dk1_p['year'] == 2017]
dk1_p_test = dk1_p[dk1_p['year'] == 2019]

# prices_training = dk1_p_training.DK_1_day_ahead_price.to_numpy()
prices_test = dk1_p_test.DK_1_day_ahead_price.to_numpy()

# Parameters
battery_capacity = 10
initial_storage = 0
num_storage_levels = battery_capacity + 1
num_price_levels = 100
gamma = 0.99
iterations = 1000
tolerance = 1e-4
eta_charge = 0.95
eta_discharge = 0.9

# Discretize prices
price_min, price_max = np.min(prices_test), np.max(prices_test)
price_grid = np.linspace(price_min, price_max, num_price_levels)

# Initialize VFI
V = np.zeros((num_storage_levels, num_price_levels))
policy = np.zeros((num_storage_levels, num_price_levels), dtype=int)

# Price transition probabilities
price_transitions = np.zeros((num_price_levels, num_price_levels))
for i in range(num_price_levels):
    for j in range(num_price_levels):
        price_transitions[i, j] = np.exp(-abs(price_grid[i] - price_grid[j]))
    price_transitions[i, :] /= np.sum(price_transitions[i, :])

# Value Function Iteration
for it in range(iterations):
    V_new = np.copy(V)
    for s in range(num_storage_levels):
        for p in range(num_price_levels):
            price = price_grid[p]
            actions = []
            if s > 0:
                reward = price * eta_discharge
                future = gamma * np.sum(price_transitions[p, :] * V[s - 1, :])
                actions.append((reward + future, -1))
            actions.append((gamma * np.sum(price_transitions[p, :] * V[s, :]), 0))
            if s < battery_capacity:
                reward = -price / eta_charge
                future = gamma * np.sum(price_transitions[p, :] * V[s + 1, :])
                actions.append((reward + future, 1))
            V_new[s, p], policy[s, p] = max(actions)
    if np.max(np.abs(V_new - V)) < tolerance:
        print(f'Converged in {it+1} iterations.')
        break
    V = V_new

# Simulation
num_periods = len(prices_test)
storage = initial_storage
battery_storage_sim = np.zeros(num_periods)
profit_sim = np.zeros(num_periods)

for t in range(num_periods):
    price = prices_test[t]
    p_idx = np.argmin(np.abs(price_grid - price))
    action = policy[storage, p_idx]
    if action == 1 and storage < battery_capacity:
        storage += 1
        profit_sim[t] = profit_sim[t-1] - price if t > 0 else -price
    elif action == -1 and storage > 0:
        storage -= 1
        profit_sim[t] = profit_sim[t-1] + price if t > 0 else price
    else:
        profit_sim[t] = profit_sim[t-1] if t > 0 else 0
    battery_storage_sim[t] = storage

# Plots
plt.figure(figsize=(12, 5))
plt.scatter(range(num_periods), battery_storage_sim, c=prices_test, cmap="coolwarm", edgecolors="k")
plt.plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
plt.colorbar(label="Price Level (DKK/MWh)")
plt.xlabel("Time Periods")
plt.ylabel("Battery Storage Level")
plt.title("Battery Storage Evolution")
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(prices_test, color="orange", label="Test Prices")
plt.xlabel("Time Periods")
plt.ylabel("Price (DKK/MWh)")
plt.title("Electricity Prices")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(profit_sim, color="green", label="Cumulative Profit")
plt.xlabel("Time Periods")
plt.title("Profit Over Time")
plt.grid()
plt.legend()
plt.show()

# %%
