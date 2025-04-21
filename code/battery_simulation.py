#%%
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
num_periods = 100
battery_capacity = 10  # Max storage capacity
initial_storage = 5  # Start at half capacity

# Generate continuous price data (random walk for realistic price behavior)
np.random.seed(42)
prices = np.cumsum(np.random.randn(num_periods))
prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices)) * 10  # Scale between 0 and 10

# Compute a moving average for decision-making
moving_avg = np.convolve(prices, np.ones(10)/10, mode='same')

# Initialize storage and profit tracking
battery_storage = np.zeros(num_periods)
profit = np.zeros(num_periods)
total_profit = 0
storage = initial_storage

# Simulate battery operation based on price thresholds
for t in range(num_periods):
    if prices[t] < moving_avg[t] and storage < battery_capacity:  # Buy when price is low
        total_profit -= prices[t]  # Subtract cost
        storage += 1
    elif prices[t] > moving_avg[t] and storage > 0:  # Sell when price is high
        total_profit += prices[t]  # Add revenue
        storage -= 1
    battery_storage[t] = storage
    profit[t] = total_profit  # Track cumulative profit

# Plot Battery Storage Evolution
plt.figure(figsize=(12, 5))
plt.scatter(range(num_periods), battery_storage, c=prices, cmap="coolwarm", edgecolors="k", label="Battery Storage")
plt.plot(battery_storage, linestyle="-", alpha=0.5, color="gray")  # Light line for trend
plt.colorbar(label="Price Level (Continuous)")
plt.xlabel("Time Periods")
plt.ylabel("Battery Storage Level")
plt.title("Battery Storage Evolution Over Time (Colored by Continuous Price)")
plt.grid()
plt.show()

# Plot Cumulative Profit Evolution
plt.figure(figsize=(12, 5))
plt.plot(profit, color="green", linestyle="-", marker="o", markersize=4, label="Cumulative Profit")
plt.xlabel("Time Periods")
plt.ylabel("Profit")
plt.title("Prices Evolution Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot Cumulative Profit Evolution
plt.figure(figsize=(12, 5))
plt.plot(prices, color="orange", linestyle="-", marker="o", markersize=4, label="Cumulative Profit")
plt.plot(moving_avg, color="blue", linestyle="-", marker="o", markersize=4, label="MA")
plt.xlabel("Time Periods")
plt.ylabel("Profit")
plt.title("Cumulative Profit Evolution Over Time")
plt.legend()
plt.grid()
plt.show()
#%%