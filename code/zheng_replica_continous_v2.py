"""
3) be able to add gridpoints to prices
1) be able to add gridpoints to battery storage
2) EGM?
- project description. 
"""


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
np.set_printoptions(precision=2, suppress=True)

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
# dk1_p_train = dk1_p_train[dk1_p_train['day'] == 1]
# test data 
# dk1_p_test = dk1_p[dk1_p['DK_1_day_ahead_price'] > 10][:200]
dk1_p_test = dk1_p[dk1_p['year'] == 2019]
# dk1_p_test = dk1_p_test[dk1_p_test['month'] == 1]
# dk1_p_test = dk1_p_test[dk1_p_test['day'] == 1]

prices_train = dk1_p_train.DK_1_day_ahead_price.to_numpy()
prices_test = dk1_p_test.DK_1_day_ahead_price.to_numpy()
# assert len(prices_train) == len(prices_test), "Training and test data lengths do not match."
dk1_p_test
#%%
# State variables 
battery_capacity_min = 3
battery_capacity = 11.2

num_storage_levels = 12
battery_grid = np.linspace(battery_capacity_min, battery_capacity, num_storage_levels)

num_price_levels = 12 # 100 gridpoints seems sufficient (no change to 1000)
price_min, price_max = np.min(prices_train), np.max(prices_train)
price_grid = np.linspace(price_min, price_max, num_price_levels)

# action space 
num_actions = 25  # Number of discrete points in continuous space
action_grid = np.linspace(-7.2, 7.2, num_actions)

# Parameters
gamma = 0.99
eta_charge = 0.9
eta_discharge = 0.9

# Convergence parameters
max_iteration = 2000
tolerance = 1e-4


#%% 
# Price transition probabilities
price_transitions = np.zeros((num_price_levels, num_price_levels))
price_indices_training = np.array([np.argmin(np.abs(price_grid - p)) for p in prices_train])

# check
assert np.unique(price_indices_training).size == num_price_levels, "Not all price indices are observed in training data."

for t in range(len(price_indices_training) - 1):
    i = price_indices_training[t]
    j = price_indices_training[t + 1]
    price_transitions[i, j] += 1

# Normalize rows
row_sums = price_transitions.sum(axis=1, keepdims=True)
price_transitions = np.divide(price_transitions, row_sums, where=row_sums != 0)

assert np.allclose(price_transitions.sum(axis=1), 1), "Not all rows sum to 1."

print(price_transitions)

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
# Initialize VFI
V = np.zeros((num_storage_levels, num_price_levels))
policy = np.zeros((num_storage_levels, num_price_levels), dtype=float)



# Value Function Iteration with continuous actions
for it in range(max_iteration):
    V_new = np.copy(V)

    interp = interpolate.interp1d(
                                battery_grid,    # x-values
                                V_new,           # y-values (shape: s x p)
                                axis=0,          # axis=0 : returns p dimension
                                bounds_error=False,
                                fill_value='extrapolate')

    for i_s, s in enumerate(battery_grid):
        for p in range(num_price_levels):
            price = price_grid[p]

            best_value = -np.inf
            best_action = 0

            for a in action_grid:

                storage_next = s + a
                if storage_next < battery_capacity_min or storage_next > battery_capacity:
                    continue  # skip infeasible storage transitions

                # Compute adjusted reward
                if a > 0: # charge
                    reward = -a * price / eta_charge  
                elif a < 0: #discharge
                    reward = -a * price * eta_discharge  
                else: # hold
                    reward = 0

                # Interpolate value at fractional storage level
                future_V = interp(storage_next)  
                future_value = np.dot(price_transitions[p, :], future_V)

                total_value = reward + gamma * future_value

                if total_value > best_value:
                    best_value = total_value
                    best_action = a
            
            V_new[i_s, p] = best_value
            policy[i_s, p] = best_action
    
    if np.max(np.abs(V_new - V)) < tolerance:
        print(f'Converged in {it+1} iterations.')
        break
    V = V_new

print(policy)

#%% Simulation with continuous action policy
num_periods = len(prices_test)
battery_storage_sim = np.zeros(num_periods)
profit_sim = np.zeros(num_periods)

interp_final = interpolate.interp1d(battery_grid,    # x-values
                                    V,           # y-values (shape: s x p)
                                    axis=0,          # axis=0 : returns p dimension
                                    bounds_error=False,
                                    fill_value='extrapolate')

for t in range(num_periods):
    storage = battery_storage_sim[t]

    price = prices_test[t]
    p_idx = np.argmin(np.abs(price_grid - price))

    s = battery_storage_sim[t]
    s_idx = np.argmin(np.abs(battery_grid - s))

    # # Get continuous action (charge/discharge level)
    # s_int = int(np.floor(storage))
    # s_frac = storage - s_int
    # s_int = min(s_int, battery_capacity - 1)  # keep within bounds
    # s_next_int = min(s_int + 1, battery_capacity)
    
    # Interpolate action from policy
    action = policy[s_idx, p_idx]
    storage_next = np.clip(storage + action, 0, battery_capacity)
    
    # Calculate profit
    if action > 0:
        cost = action * price / eta_charge
        profit = -cost
    elif action < 0:
        revenue = -action * price * eta_discharge
        profit = revenue
    else:
        profit = 0

    profit_sim[t] = profit_sim[t - 1] + profit if t > 0 else profit
    if t != num_periods - 1: battery_storage_sim[t+1] = storage_next 

#%% plots 

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
axs[1].scatter(discharge_times, prices_test[discharge_times], color="red", label="Discharge", s=20)
axs[1].scatter(charge_times, prices_test[charge_times], color="blue", label="Charge", s=20)

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
