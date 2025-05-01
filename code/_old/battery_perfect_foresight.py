"""
backwards induction with perfect foresight 
"""
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# Load data
dk1_p = pd.read_csv('../data/entsoe_price_DK_1_20150101_20240101.csv')
dk1_p['date'] = pd.to_datetime(dk1_p['date'], utc=True)
dk1_p['year'] = dk1_p['date'].dt.year

# Split data
dk1_p_training = dk1_p[dk1_p['year'] == 2017]
dk1_p_test = dk1_p[dk1_p['year'] == 2018]

prices_training = dk1_p_training.DK_1_day_ahead_price.to_numpy()
prices_test = dk1_p_test.DK_1_day_ahead_price.to_numpy()
assert len(prices_training) == len(prices_test), "Training and test data lengths do not match."

#%%
# Parameters
num_periods = prices_test.shape[0]
battery_capacity = 10
initial_storage = 0

#%% Perfect foresight dynamic programming (backward induction)
storage_levels = np.arange(0, battery_capacity + 1)  # 0 to 10

# Value function (time x storage)
V = np.zeros((num_periods + 1, battery_capacity + 1))
policy = np.zeros((num_periods, battery_capacity + 1), dtype=int)  # -1=sell, 0=hold, +1=buy

for t in reversed(range(num_periods)):
    today_price = prices_test[t]
    tomorrow_price = prices_test[t + 1] if t + 1 < num_periods else today_price

    for s in storage_levels:
        actions = []
        values = []

        # Hold
        actions.append(0)
        values.append(V[t + 1, s])

        # Sell
        if s > 0:
            actions.append(-1)
            values.append(today_price + V[t + 1, s - 1])

        # Buy
        if s < battery_capacity:
            actions.append(1)
            values.append(-today_price + V[t + 1, s + 1])

        # Choose best action
        best_idx = np.argmax(values)
        V[t, s] = values[best_idx]
        policy[t, s] = actions[best_idx]

#%% Simulate battery operation using optimal policy
battery_storage_sim = np.zeros(num_periods)
profit_sim = np.zeros(num_periods)
storage = initial_storage

for t in range(num_periods):
    action = policy[t, storage]

    if action == 1 and storage < battery_capacity:  # Buy
        storage += 1
        profit_sim[t] = profit_sim[t - 1] - prices_test[t] if t > 0 else -prices_test[t]
    elif action == -1 and storage > 0:  # Sell
        storage -= 1
        profit_sim[t] = profit_sim[t - 1] + prices_test[t] if t > 0 else prices_test[t]
    else:  # Hold
        profit_sim[t] = profit_sim[t - 1] if t > 0 else 0

    battery_storage_sim[t] = storage

#%% Plot Battery Storage Evolution
plt.figure(figsize=(12, 5))
plt.scatter(range(num_periods), battery_storage_sim, c=prices_test, cmap="coolwarm", edgecolors="k", label="Battery Storage")
plt.plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
plt.colorbar(label="Price Level (Continuous)")
plt.xlabel("Time Periods")
plt.ylabel("Battery Storage Level")
plt.title("Battery Storage Evolution Over Time (Perfect Foresight)")
plt.grid()
plt.show()

#%% Plot Prices
plt.figure(figsize=(12, 5))
plt.plot(prices_test, color="orange", label="Test", alpha=0.7)
plt.plot(prices_training, color="red", label="Train", alpha=0.5)
plt.xlabel("Time Periods")
plt.ylabel("Price")
plt.title("Day-Ahead Prices (DK1)")
plt.legend()
plt.grid()
plt.show()

#%% Plot Profit
plt.figure(figsize=(12, 5))
plt.plot(profit_sim, color="green", linestyle="-", marker="o", markersize=3, label="Cumulative Profit")
plt.xlabel("Time Periods")
plt.title("Cumulative Profit Over Time (Perfect Foresight)")
plt.legend()
plt.grid()
plt.show()

# %%
profit_sim
# %%
