import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class EnergyStorageModel:
    def __init__(self, price_data_path, battery_capacity_min=3, battery_capacity=11.2, num_storage_levels=10, num_price_levels=12, num_actions=5, gamma=0.99, eta_charge=0.89, eta_discharge=0.89, max_iteration=2000, tolerance=1e-6):
        # Load price data
        self.price_data_path = price_data_path
        self.load_data()
        
        # Battery and pricing grid
        self.battery_capacity_min = battery_capacity_min
        self.battery_capacity = battery_capacity
        self.num_storage_levels = num_storage_levels
        self.num_price_levels = num_price_levels
        self.num_actions = num_actions
        self.gamma = gamma
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        
        # Gridpoints
        self.battery_grid = np.linspace(self.battery_capacity_min, self.battery_capacity, self.num_storage_levels)
        self.price_grid = np.linspace(np.min(self.prices_train), np.max(self.prices_train), self.num_price_levels)
        self.action_grid = np.linspace(-7.2, 7.2, self.num_actions)
        
        # Value function and policy
        self.V = np.zeros((self.num_storage_levels, self.num_price_levels))
        self.policy = np.zeros((self.num_storage_levels, self.num_price_levels), dtype=float)

    def load_data(self):
        # Load and process price data
        dk1_p = pd.read_csv(self.price_data_path)
        dk1_p['date'] = pd.to_datetime(dk1_p['date'], utc=True)
        dk1_p['year'] = dk1_p['date'].dt.year
        dk1_p['month'] = dk1_p['date'].dt.month
        dk1_p['day'] = dk1_p['date'].dt.day
        
        # Training and test data
        self.dk1_p_train = dk1_p[dk1_p['year'] == 2018]
        self.dk1_p_test = dk1_p[dk1_p['year'] == 2019]
        
        self.prices_train = self.dk1_p_train.DK_1_day_ahead_price.to_numpy()
        self.prices_test = self.dk1_p_test.DK_1_day_ahead_price.to_numpy()

    def add_gridpoints(self, grid_type='battery', num_points=10):
        if grid_type == 'battery':
            self.battery_grid = np.linspace(self.battery_capacity_min, self.battery_capacity, num_points)
        elif grid_type == 'price':
            self.price_grid = np.linspace(np.min(self.prices_train), np.max(self.prices_train), num_points)
        else:
            raise ValueError("Invalid grid type. Use 'battery' or 'price'.")

    def compute_price_transitions(self):
        price_transitions = np.zeros((self.num_price_levels, self.num_price_levels))
        price_indices_training = np.array([np.argmin(np.abs(self.price_grid - p)) for p in self.prices_train])

        for t in range(len(price_indices_training) - 1):
            i = price_indices_training[t]
            j = price_indices_training[t + 1]
            price_transitions[i, j] += 1

        # Normalize rows
        row_sums = price_transitions.sum(axis=1, keepdims=True)
        price_transitions = np.divide(price_transitions, row_sums, where=row_sums != 0)
        self.price_transitions = price_transitions

    def value_function_iteration(self):
        for it in range(self.max_iteration):
            V_new = np.copy(self.V)

            interp = interpolate.interp1d(self.battery_grid, V_new, axis=0, bounds_error=False, fill_value='extrapolate')

            for i_s, s in enumerate(self.battery_grid):
                for p in range(self.num_price_levels):
                    price = self.price_grid[p]

                    best_value = -np.inf
                    best_action = 0

                    for a in self.action_grid:
                        storage_next = s + a
                        if storage_next < self.battery_capacity_min or storage_next > self.battery_capacity:
                            continue

                        # Compute reward and future value
                        if a > 0:
                            reward = -a * price / self.eta_charge
                        elif a < 0:
                            reward = -a * price * self.eta_discharge
                        else:
                            reward = 0

                        future_V = interp(storage_next)
                        future_value = np.dot(self.price_transitions[p, :], future_V)

                        total_value = reward + self.gamma * future_value

                        if total_value > best_value:
                            best_value = total_value
                            best_action = a

                    V_new[i_s, p] = best_value
                    self.policy[i_s, p] = best_action

            if np.max(np.abs(V_new - self.V)) < self.tolerance:
                print(f'Converged in {it+1} iterations.')
                break

            self.V = V_new
        return self.V, self.policy

    def vfi_vec(self, _print=False):

        for it in range(self.max_iteration):
            V_new = np.copy(self.V)

            interp = interpolate.interp1d(self.battery_grid, V_new, axis=0, bounds_error=False, fill_value='extrapolate')

            # possible actions for each state: OBS action can be between the actiongrid.  
            pos_actions = self.battery_grid[:, np.newaxis] + self.action_grid[np.newaxis, :] 
            pos_actions = np.clip(pos_actions, self.battery_capacity_min, self.battery_capacity)
            if _print: print('pos_actions = ', pos_actions.shape) # num_storage_levels, num_actions

            # Calculate reward for each action and price level
            reward = np.where(pos_actions > 0, -pos_actions / self.eta_charge, -pos_actions * self.eta_discharge)
            reward = reward[:, :, np.newaxis] * self.price_grid[ np.newaxis, :]
            if _print: print('reward = ', reward.shape) # (num_storage_levels, num_actions, num_price_levels)

            # OBS action can be between the actiongrid. 
            s_next = self.battery_grid[:, np.newaxis] + self.action_grid[np.newaxis, :] 
            s_next = np.clip(s_next, self.battery_capacity_min, self.battery_capacity)
            if _print: print('s_next = ', s_next.shape) # (num_storage_levels, num_actions)

            future_V = interp(s_next) 
            if _print: print('future_V = ', future_V.shape)  # (num_gridpoints, num_actions, num_price_levels)

            future_value = np.dot(future_V, self.price_transitions)
            future_value.reshape
            if _print: print('future_value = ',future_value.shape) # (num_gridpoints, num_price_levels)

            total_value = reward + self.gamma * future_value 
            if _print: print('total_value = ', total_value.shape) # (num_gridpoints, num_actions, num_price_levels)
            # Find the best action for each state and price level
            best_value = np.max(total_value, axis=1) 
            best_action = np.argmax(total_value, axis=1) 

            if _print: print('best_value = ', best_value.shape)

            # Update V_new and policy
            V_new = best_value
            self.policy = best_action

            # Check for convergence
            if np.max(np.abs(V_new - self.V)) < self.tolerance:
                print(f'Converged in {it+1} iterations.')
                print(f'reward = {reward}')
                break

            self.V = V_new
        return self.V, self.policy


    def simulate(self):
        num_periods = len(self.prices_test)
        battery_storage_sim = np.zeros(num_periods)
        battery_storage_sim += self.battery_capacity_min
        profit_sim = np.zeros(num_periods)

        for t in range(num_periods):
            storage = battery_storage_sim[t]
            price = self.prices_test[t]
            p_idx = np.argmin(np.abs(self.price_grid - price))

            s = battery_storage_sim[t]
            s_idx = np.argmin(np.abs(self.battery_grid - s))

            action = self.policy[s_idx, p_idx]
            storage_next = storage + action

            # Calculate profit
            if action > 0:
                cost = action * price / self.eta_charge
                profit = -cost
            elif action < 0:
                revenue = -action * price * self.eta_discharge
                profit = revenue
            else:
                profit = 0

            profit_sim[t] = profit_sim[t - 1] + profit if t > 0 else profit
            if t != num_periods - 1:
                battery_storage_sim[t+1] = storage_next
        
        return battery_storage_sim, profit_sim

    def plot_results(self, battery_storage_sim, profit_sim):
        fig, axs = plt.subplots(4, 1, figsize=(10, 30), sharex=True)

        axs[0].scatter(range(len(battery_storage_sim)), battery_storage_sim, c=self.prices_test, cmap="coolwarm", edgecolors="k")
        axs[0].plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
        axs[0].set_ylabel("Battery Storage Level")
        axs[0].set_title("Battery Storage, Prices, and Profit Over Time")
        axs[0].grid(True)

        axs[1].plot(self.prices_test, color="orange", label="Test Prices", alpha=0.5)
        axs[1].set_ylabel("Price (EUR/MWh)")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(profit_sim, color="green", label="Cumulative Profit")
        axs[2].set_xlabel("Time Periods")
        axs[2].set_ylabel("Profit")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(self.prices_train, color="red", label="Train Prices", alpha=0.5)
        axs[3].plot(self.prices_test, color="orange", label="Test Prices", alpha=0.5)
        axs[3].set_ylabel("Price (EUR/MWh)")
        axs[3].legend()
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()