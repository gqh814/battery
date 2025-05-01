#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
np.set_printoptions(precision=2, suppress=True)
#%%
# Set parameters
num_periods = 1000  # how many time steps to simulate
num_price_levels = 11
price_min, price_max = 0, 100
price_grid = np.linspace(price_min, price_max, num_price_levels)
price_avg = np.mean(price_grid)

# action space 
num_actions = 5  # Number of discrete points in continuous space
load = 1.0
action_grid = np.linspace(-load, load, num_actions)

battery_capacity_min = 0
battery_capacity = 20

num_storage_levels = 41
battery_grid = np.linspace(battery_capacity_min, battery_capacity, num_storage_levels)

# Parameters
gamma = 0.95
eta_charge = 0.9
eta_discharge = 0.9
marginal_cost = 1.0

# Convergence parameters
max_iteration = 2000
tolerance = 1e-6
#%%
# Initialize empty transition matrix
price_transitions = np.zeros((num_price_levels, num_price_levels))

def mean_reverting_probs_extended(price, price_avg, scale=0.1):
    """Return probabilities for [big down, small down, stay, small up, big up]"""
    distance = price - price_avg

    # Weights based on smooth mean reversion logic
    weight_big_down  = 0.1 * (1 / (1 + np.exp(-scale * distance)))
    weight_down      = 0.3 * (1 / (1 + np.exp(-scale * distance)))
    weight_stay      = 0.2 + 0.2 * np.exp(-abs(distance) / 10)  # more stay near avg
    weight_up        = 0.3 * (1 / (1 + np.exp(scale * distance)))
    weight_big_up    = 0.1 * (1 / (1 + np.exp(scale * distance)))

    weights = np.array([
        weight_big_down,
        weight_down,
        weight_stay,
        weight_up,
        weight_big_up
    ])
    return weights / np.sum(weights)

# Fill the matrix
for i, price in enumerate(price_grid):
    probs = mean_reverting_probs_extended(price, price_avg)

    # Assign to positions i-2, i-1, i, i+1, i+2 with boundary checks
    if i >= 2:
        price_transitions[i, i-2] += probs[0]
    else:
        price_transitions[i, i] += probs[0]  # reflect

    if i >= 1:
        price_transitions[i, i-1] += probs[1]
    else:
        price_transitions[i, i] += probs[1]  # reflect

    price_transitions[i, i] += probs[2]

    if i < num_price_levels - 1:
        price_transitions[i, i+1] += probs[3]
    else:
        price_transitions[i, i] += probs[3]  # reflect

    if i < num_price_levels - 2:
        price_transitions[i, i+2] += probs[4]
    else:
        price_transitions[i, i] += probs[4]  # reflect

# Normalize rows
price_transitions /= price_transitions.sum(axis=1, keepdims=True)

# Check
assert np.allclose(price_transitions.sum(axis=1), 1), "Rows must sum to 1."

# Display
np.set_printoptions(precision=3, suppress=True)
print("Price Grid:", price_grid)
print("\nTransition Matrix:")
print(price_transitions)

selected_indices = [0 ,3 , 5, 8, 10]
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



# %%
# Assume you already have:

# Simulation parameters
price_series = np.zeros(num_periods)

# Start at the middle price
current_idx = len(price_grid) // 2
price_series[0] = price_grid[current_idx]

# Simulate
np.random.seed(1)
for t in range(1, num_periods):
    next_idx = np.random.choice(
        np.arange(len(price_grid)),
        p=price_transitions[current_idx]
    )
    price_series[t] = price_grid[next_idx]
    current_idx = next_idx



# %%
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
                    reward = -a * price / eta_charge - marginal_cost*a  
                elif a < 0: #discharge
                    reward = -a * price * eta_discharge + marginal_cost*a  
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

#%%
num_periods = len(price_series)
battery_storage_sim = np.zeros(num_periods)
battery_storage_sim += battery_capacity_min
profit_sim = np.zeros(num_periods)
action_sim = np.zeros(num_periods)

# interp_final = interpolate.interp1d(battery_grid,    # x-values
#                                     V,           # y-values (shape: s x p)
#                                     axis=0,          # axis=0 : returns p dimension
#                                     bounds_error=False,
#                                     fill_value='extrapolate')

for t in range(num_periods):
    storage = battery_storage_sim[t]

    price = price_series[t]
    p_idx = np.argmin(np.abs(price_grid - price))

    s = battery_storage_sim[t]
    s_idx = np.argmin(np.abs(battery_grid - s))

    action = policy[s_idx, p_idx]
    storage_next = storage + action
    matched_price = price_grid[p_idx]
    action_sim[t] = action

    # Calculate profit
    if action > 0:
        cost = action * price / eta_charge + marginal_cost*action
        profit = -cost
    elif action < 0:
        revenue = -action * price * eta_discharge + marginal_cost*action 
        profit = revenue
    else:
        profit = 0


    profit_sim[t] = profit_sim[t - 1] + profit if t > 0 else profit
    if t != num_periods - 1: battery_storage_sim[t+1] = storage_next 
#%% plots 

fig, axs = plt.subplots(3, 1, figsize=(10, 30), sharex=True)

# Plot Battery Storage Evolution (without legend)
axs[0].scatter(range(num_periods), battery_storage_sim, c=price_series, cmap="coolwarm", edgecolors="k")
axs[0].plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
axs[0].set_ylabel("Battery Storage Level")
axs[0].set_title("Battery Storage, Prices, and Profit Over Time")
axs[0].grid(True)

# Plot Prices
# Identify charge and discharge times
charge_times = np.where(action_sim > 0.01)[0]
discharge_times = np.where(action_sim < -0.01)[0]

# Price Plot with action markers
axs[1].plot(price_series, color="orange", label="Test Prices", alpha=0.5)
# Overlay markers
# plot hline with mean of prices_test
axs[1].axhline(y=np.mean(price_series), color='gray', linestyle='--', label='Mean Price')
axs[1].scatter(discharge_times, price_series[discharge_times], color="red", label="Discharge", s=20)
axs[1].scatter(charge_times, price_series[charge_times], color="blue", label="Charge", s=20)

axs[1].set_ylabel("Price (EUR/MWh)")
axs[1].legend()
axs[1].grid(True)

# Plot Cumulative Profit
axs[2].plot(profit_sim, color="green", label="Cumulative Profit")
axs[2].set_xlabel("Time Periods")
axs[2].set_ylabel("Profit")
axs[2].legend()
axs[2].grid(True)

#%%
#%% Sensitivity analysis on eta

eta_list = [0.7, 0.8, 0.9, 0.95]
results = {}
sensitivity_par = "eta_discharge"


for eta in eta_list:
    V = np.zeros((num_storage_levels, num_price_levels))
    policy = np.zeros((num_storage_levels, num_price_levels), dtype=float)

    # --- Value Function Iteration ---
    for it in range(max_iteration):
        V_new = np.copy(V)
        interp = interpolate.interp1d(battery_grid, V_new, axis=0, bounds_error=False, fill_value='extrapolate')

        for i_s, s in enumerate(battery_grid):
            for p in range(num_price_levels):
                price = price_grid[p]
                best_value = -np.inf
                best_action = 0

                for a in action_grid:
                    storage_next = s + a
                    if storage_next < battery_capacity_min or storage_next > battery_capacity:
                        continue

                    if a > 0:
                        if sensitivity_par == "eta_charge":
                            reward = -a * price / eta - marginal_cost*a 
                        else:
                            reward =-a * price / eta_charge - marginal_cost*a 
                    elif a < 0:
                        if sensitivity_par == "eta_discharge":
                            reward = -a * price * eta + marginal_cost*a 
                        else:
                            reward = -a * price * eta_discharge + marginal_cost*a 
                    else:
                        reward = 0

                    future_V = interp(storage_next)
                    future_value = np.dot(price_transitions[p, :], future_V)
                    total_value = reward + gamma * future_value

                    if total_value > best_value:
                        best_value = total_value
                        best_action = a

                V_new[i_s, p] = best_value
                policy[i_s, p] = best_action


        if np.max(np.abs(V_new - V)) < tolerance:
            if sensitivity_par == "eta_charge":
                print(f"[eta_charge={eta}] Converged in {it+1} iterations.")
            else:
                print(f"[eta_discharge={eta}] Converged in {it+1} iterations.")
            break
        V = V_new

    # --- Simulation ---
    battery_storage_sim = np.zeros(num_periods)
    battery_storage_sim[0] = battery_capacity_min
    profit_sim = np.zeros(num_periods)

    for t in range(num_periods):
        price = price_series[t]
        p_idx = np.argmin(np.abs(price_grid - price))
        s_idx = np.argmin(np.abs(battery_grid - battery_storage_sim[t]))

        action = policy[s_idx, p_idx]
        storage_next = battery_storage_sim[t] + action

        if action > 0:
            if sensitivity_par == "eta_charge":
                profit = -action * price / eta - marginal_cost*action 
            else:
                profit = -action * price / eta_charge - marginal_cost*action 
        elif action < 0:
            if sensitivity_par == "eta_discharge":
                profit = -action * price * eta + marginal_cost*action 
            else:
                profit = -action * price * eta_discharge + marginal_cost*action 
        else:
            profit = 0

        profit_sim[t] = profit_sim[t-1] + profit if t > 0 else profit
        if t < num_periods - 1:
            battery_storage_sim[t+1] = storage_next
    results[eta] = profit_sim

# --- Plot ---
plt.figure(figsize=(12, 6))
for eta, profit_sim in results.items():
    if sensitivity_par == "eta_charge":
        plt.plot(profit_sim, label=f"η_charge = {eta}")
    else:
        plt.plot(profit_sim, label=f"η_discharge = {eta}")

plt.xlabel("Time Period")
plt.ylabel("Cumulative Profit")
plt.title("Cumulative Profit for Different Charge Efficiencies")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
