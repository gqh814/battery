#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# Load data
dk1_p = pd.read_csv('../data/entsoe_price_DK_1_20150101_20240101.csv')
dk1_p['date'] = pd.to_datetime(dk1_p['date'], utc=True)
dk1_p['year'] = dk1_p['date'].dt.year
dk1_p['month'] = dk1_p['date'].dt.month
dk1_p['day'] = dk1_p['date'].dt.day

# training data
dk1_p_train = dk1_p[dk1_p['year'] == 2018]
# dk1_p_train = dk1_p_train[dk1_p_train['month'] == 1]
# test data 
# dk1_p_test = dk1_p[dk1_p['DK_1_day_ahead_price'] > 10][:200]
dk1_p_test = dk1_p[dk1_p['year'] == 2019]
# dk1_p_test = dk1_p_test[dk1_p_test['month'] == 1]
# dk1_p_test = dk1_p_test[dk1_p_test['day'] <= 10]

prices_train = dk1_p_train.DK_1_day_ahead_price.to_numpy()
prices_test = dk1_p_test.DK_1_day_ahead_price.to_numpy()
# assert len(prices_train) == len(prices_test), "Training and test data lengths do not match."
dk1_p_test
#%%
# Parameters
battery_capacity = 10
initial_storage = 0
num_storage_levels = battery_capacity + 1
num_price_levels = 13 # 100 gridpoints seems sufficient (no change to 1000)
gamma = 0.99
max_iteration = 2000
tolerance = 1e-4
eta_charge = 0.95
eta_discharge = 0.95

# Discretize prices
price_min, price_max = np.min(prices_train), np.max(prices_train)
price_grid = np.linspace(price_min, price_max, num_price_levels)

# Initialize VFI
V = np.zeros((num_storage_levels, num_price_levels))
policy = np.zeros((num_storage_levels, num_price_levels), dtype=int)


#%% Price transition probabilities
# problem some prices do not fall into category. 
# price_transitions = np.zeros((num_price_levels, num_price_levels))
# for i in range(num_price_levels):
#     for j in range(num_price_levels):
#         price_transitions[i, j] = np.exp(-abs(price_grid[i] - price_grid[j]))
#     price_transitions[i, :] /= np.sum(price_transitions[i, :])
np.set_printoptions(precision=3, suppress=True)
# Rebuild empirical transition matrix using training data
price_transitions = np.zeros((num_price_levels, num_price_levels))
price_indices_training = np.array([np.argmin(np.abs(price_grid - p)) for p in prices_train])

assert max(price_indices_training) == num_price_levels-1, "Price index exceeds number of price levels."

for t in range(len(price_indices_training) - 1):
    i = price_indices_training[t]
    j = price_indices_training[t + 1]
    price_transitions[i, j] += 1
print(price_transitions)
# Normalize rows
row_sums = price_transitions.sum(axis=1, keepdims=True)

price_transitions = np.divide(price_transitions, row_sums, where=row_sums != 0)
assert np.allclose(price_transitions.sum(axis=1), 1), "Not all rows sum to 1."

#%%
# Plot histogram for price_grid[56] using empirical transitions
selected_index = 3
selected_price = price_grid[selected_index]
next_probs = price_transitions[selected_index, :]
print(next_probs)
print(np.sum(next_probs))
assert np.isclose(np.sum(next_probs), 1), "Probabilities do not sum to 1."

plt.figure(figsize=(10, 6))
plt.bar(price_grid, next_probs, width=(price_grid[1] - price_grid[0]), align='center', color='salmon', edgecolor='k')
plt.vlines(selected_price, ymin=-0.2,ymax=0.2)
plt.xlabel("Next Period Price (DKK/MWh)")
plt.ylabel("Empirical Probability")
plt.title(f"Empirical Transition Probabilities for Current Price ≈ {selected_price:.2f} DKK/MWh")
plt.grid(True)
plt.show()

#%%
# Plot empirical transition probabilities for 5 selected price indices
selected_indices = [1, 3, 5, 7, 9, 11]
colors = ['blue', 'green', 'red', 'purple', 'orange']

plt.figure(figsize=(12, 6))

for idx, color in zip(selected_indices, colors):
    selected_price = price_grid[idx]
    next_probs = price_transitions[idx, :]
    plt.plot(price_grid, next_probs, label=f"Price ≈ {selected_price:.2f}", color=color)
    plt.axvline(x=selected_price, color=color, linestyle="--", alpha=0.5)

plt.xlabel("Next Period Price (DKK/MWh)")
plt.ylabel("Empirical Probability")
plt.title("Empirical Transition Probabilities for Multiple Current Prices")
plt.legend(title="Current Price Level")
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Value Function Iteration
for it in range(max_iteration):
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

#%% Simulation
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

#import matplotlib.pyplot as plt
#%%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, figsize=(10, 30), sharex=True)

# Plot Battery Storage Evolution (without legend)
axs[0].scatter(range(num_periods), battery_storage_sim, c=prices_test, cmap="coolwarm", edgecolors="k")
axs[0].plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
axs[0].set_ylabel("Battery Storage Level")
axs[0].set_title("Battery Storage, Prices, and Profit Over Time")
axs[0].grid(True)

# Plot Prices
# Identify charge and discharge times
charge_times = np.where(np.diff(battery_storage_sim, prepend=0) > 0)[0]
discharge_times = np.where(np.diff(battery_storage_sim, prepend=0) < 0)[0]

# Price Plot with action markers
axs[1].plot(prices_test, color="orange", label="Test Prices", alpha=0.5)
# Overlay markers
# plot hline with mean of prices_test
axs[1].axhline(y=np.mean(prices_test), color='gray', linestyle='--', label='Mean Price')
axs[1].scatter(charge_times, prices_test[charge_times], color="blue", label="Charge", s=20)
axs[1].scatter(discharge_times, prices_test[discharge_times], color="red", label="Discharge", s=20)
axs[1].set_ylabel("Price (EUR/MWh)")
axs[1].legend()
axs[1].grid(True)

# Plot Cumulative Profit
axs[2].plot(profit_sim, color="green", label="Cumulative Profit")
axs[2].set_xlabel("Time Periods")
axs[2].set_ylabel("Profit")
axs[2].legend()
axs[2].grid(True)

# show prices 
axs[3].plot(prices_train, color="red", label="Train Prices", alpha=0.5)
axs[3].plot(prices_test, color="orange", label="Test Prices", alpha=0.5)
axs[3].set_ylabel("Price (EUR/MWh)")
axs[3].legend()
axs[3].grid(True)
# Add colorbar to the first subplot
# cbar = fig.colorbar(sc, ax=axs[0], orientation="vertical", label="Price Level (DKK/MWh)")
plt.tight_layout()
plt.show()


# %%
