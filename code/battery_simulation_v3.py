#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
time_horizon = 50  # Time steps
price_states = 100  # Possible price levels

# Generate a random walk for prices
np.random.seed(42)
prices = np.cumsum(np.random.randn(time_horizon)) + 50  # Start at 50
prices = np.maximum(prices, 1)  # Ensure prices are positive

# Value function table
V = np.zeros((time_horizon, price_states))

# Battery storage and profits tracking
battery_storage = np.zeros(time_horizon)
profits = np.zeros(time_horizon)

# Bellman equation backward iteration
for t in range(time_horizon - 2, -1, -1):
    for p in range(price_states):
        price = p + 1  # Convert index to price
        wait = V[t + 1, min(price_states - 1, p + np.random.choice([-1, 0, 1]))]
        sell = price
        V[t, p] = max(wait, sell)

# Simulate battery storage and profits over time
for t in range(1, time_horizon):
    battery_storage[t] = max(0, battery_storage[t - 1] + np.random.choice([-1, 1]))
    profits[t] = profits[t - 1] + (battery_storage[t] * prices[t])

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot price evolution
axs[0].plot(prices, label='Prices', color='blue')
axs[0].set_title('Price Evolution')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Price')
axs[0].legend()

# Plot battery storage evolution
axs[1].plot(battery_storage, label='Battery Storage', color='green')
axs[1].set_title('Battery Storage Evolution')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Storage Level')
axs[1].legend()

# Plot profit evolution
axs[2].plot(profits, label='Profits', color='red')
axs[2].set_title('Profit Evolution')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Profit')
axs[2].legend()

plt.tight_layout()
plt.show()

# Plot 3D optimal path
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.arange(time_horizon), prices, battery_storage, label='Optimal Path', color='purple', linewidth=2, marker='o', markersize=4)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Price', fontsize=12)
ax.set_zlabel('battery_storage', fontsize=12)
ax.set_title('Optimal Path in 3D Space', fontsize=14)
ax.legend()

# Enhance visibility with grid and ticks
ax.grid(True, linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=10)

# Annotate start and end points
ax.scatter(0, prices[0], battery_storage[0], color='green', s=100, label='Start')
ax.scatter(time_horizon-1, prices[-1], battery_storage[-1], color='red', s=100, label='End')
ax.legend()

plt.show()

# %%
