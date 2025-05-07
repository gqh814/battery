import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from price_simulator import PriceSimulator

"""
added is fixed prices for 2015
"""

class EnergyStorageModel:
    def __init__(self,  
                 min_battery_capacity=0, 
                 max_battery_capacity=10, 
                 num_storage_levels=11, 
                 num_price_levels=12, 
                 num_actions=7, 
                 eta_charge=0.98, 
                 eta_discharge=0.97, 
                 max_iteration=2000, 
                 tolerance=1e-6,
                 beta = 0.99,
                 sigma = 0.1/100 / 24, 
                 variable_cost=2.1/1000,
                 battery_capacity_price=(1.288 + 0.308 + 0.11)*1000,
                 power_capacity_price=0.29*1000,
                 annual_fixed_cost=0.54,
                 a_bar=7.2,
                 simulate_prices=True,
                 mean_reversion = 0.3,
                 p_variance = 100,
                 risk_averse = False,
                 risk_parameter = 0.001,
                 price_data_path = '../data/dk1price_20000101_20191231.csv'
                 ):
        
        # Battery and pricing grid
        self.min_battery_capacity = min_battery_capacity
        self.max_battery_capacity = max_battery_capacity
        self.num_storage_levels = num_storage_levels
        self.num_price_levels = num_price_levels
        self.num_actions = num_actions
        self.beta = beta
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.mean_reversion = mean_reversion
        self.p_variance = p_variance
        self.risk_averse = risk_averse
        self.risk_parameter = risk_parameter

        # added 
        self.sigma = sigma # percentage per hour # approximated 
        self.variable_cost = variable_cost # EUR/kWh
        self.battery_capacity_price = battery_capacity_price # EUR/kWh
        self.power_capacity_price = power_capacity_price # EUR/kW
        self.annual_fixed_cost = annual_fixed_cost # EUR/kW per year.
        self.a_bar = a_bar

        #
        self.price_data_path = price_data_path

        # v2 
        self._price_generator(simulate=simulate_prices)
        
        # Gridpoints
        self.battery_grid = np.linspace(self.min_battery_capacity, self.max_battery_capacity, self.num_storage_levels)
        self.action_grid = np.linspace(-self.a_bar, self.a_bar, self.num_actions)
        
        # Value function and policy
        self.V = np.zeros((self.num_storage_levels, self.num_price_levels))
        self.policy = np.zeros((self.num_storage_levels, self.num_price_levels), dtype=float)

        # prices 

    def _price_generator(self, simulate=True):

        if simulate:
            print('simulating')
            self.price_grid = np.linspace(0, 100, self.num_price_levels)
            price_avg = 50
            num_periods = 1000

            simulator = PriceSimulator(self.price_grid, price_avg, num_periods, alpha=self.mean_reversion, sigma2=self.p_variance)
            self.prices_test = simulator.simulate_prices()
            self.price_transitions = simulator.price_transitions

        else:
            print('loading data')
            self._load_data()
            self._compute_price_transitions()

    def _load_data(self):
        
        # Read the data
        price_data_path = '../data/dk2price_20000101_20191231.csv'
        dk2_p = pd.read_csv(price_data_path, sep=';')

        # Drop observations where SpotPriceEUR is NA
        dropped_obs = dk2_p[dk2_p.SpotPriceEUR.isna()]
        df = dk2_p.dropna(subset=['SpotPriceEUR']).copy()  # Make a copy to avoid warnings

        # Convert HourDK to datetime and create a 'year' column
        df['date'] = pd.to_datetime(df['HourDK'], utc=True)
        df['year'] = df['date'].dt.year  # No need for .loc here
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        df['SpotPriceEUR'] = df['SpotPriceEUR'].str.replace(',', '.').astype(float)
        df = df[['SpotPriceEUR']].resample('D').mean()

        # Filter prices
        max_price = 120
        min_price = -10
        cond = (df.SpotPriceEUR > min_price) & (df.SpotPriceEUR < max_price)
        self.df = df[cond]

        # plt.figure(figsize=(12, 4))
        # self.df.SpotPriceEUR.plot(title='Hourly Spot Prices (Before Filtering)', alpha=0.5)
        # plt.ylabel('EUR/MWh')
        # plt.xlabel('Date')
        # plt.tight_layout()
        # plt.show()


    def _compute_price_transitions(self):
        self.prices_test = self.df["SpotPriceEUR"].values
        self.price_grid = np.linspace(self.prices_test.min(), self.prices_test.max(), self.num_price_levels)

        price_transitions = np.zeros((self.num_price_levels, self.num_price_levels))
        price_indices = np.array([np.argmin(np.abs(self.price_grid - p)) for p in self.prices_test])

        assert np.unique(price_indices).size == self.num_price_levels, "Not all price indices are observed in training data."

        for t in range(len(price_indices) - 1):
            i = price_indices[t]
            j = price_indices[t + 1]
            price_transitions[i, j] += 1

        
        # Normalize rows
        row_sums = price_transitions.sum(axis=1, keepdims=True)
        price_transitions = np.divide(price_transitions, row_sums, where=row_sums != 0)
        assert np.allclose(price_transitions.sum(axis=1), 1), "Not all rows sum to 1."

        self.price_transitions = price_transitions

    def plot_price_transition_distributions(self, indices=None):
        price_grid = self.price_grid
        T = self.price_transitions

        if indices is None:
            i_first = 0
            i_mid = np.argmin(np.abs(price_grid - np.mean(price_grid)))
            i_last = len(price_grid) - 1
            indices = [i_first, i_mid, i_last]

        plt.figure(figsize=(10, 6))
        for i in indices:
            plt.plot(price_grid, T[i], label=f'From {price_grid[i]:.1f}')

        plt.xlabel('Target Price (EUR/MWh)')
        plt.ylabel('Transition Probability')
        plt.title('Transition Distributions from Selected Price Levels')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def utility_function(self, action, price):
        # Compute deterministic profit
        variable_cost = self.variable_cost * abs(action)

        if action > 0:
            profit = -action * price / self.eta_charge - variable_cost
        elif action < 0:
            profit = -action * price * self.eta_discharge - variable_cost
        else:
            profit = 0

        # Apply CARA utility if risk aversion is enabled
        if self.risk_averse:
            return -np.exp(-self.risk_parameter * profit)
        else:
            return profit

    def profit_function(self, action, price):

        variable_cost = self.variable_cost*np.abs(action)  

        if action > 0:
            return -action * price / self.eta_charge - variable_cost
        elif action < 0:
            return -action * price * self.eta_discharge - variable_cost
        else:
            return 0
        
    def value_function_iteration(self):

        for it in range(self.max_iteration):
            V_new = np.copy(self.V)

            interp = interpolate.interp1d(self.battery_grid, 
                                          V_new, 
                                          axis=0, 
                                          bounds_error=True) # True = no extrapolation outside battery_grid

            for i_s, s in enumerate(self.battery_grid):
                for p in range(self.num_price_levels):
                    price = self.price_grid[p]

                    best_value = -np.inf
                    best_action = 0

                    for a in self.action_grid:
                        
                        storage_next = s * (1-self.sigma) + a

                        # skip invalid actions
                        if storage_next < self.min_battery_capacity or storage_next > self.max_battery_capacity:
                            continue 

                        reward = self.utility_function(a, price)

                        future_V = interp(storage_next)
                        future_value = np.dot(self.price_transitions[p, :], future_V)

                        total_value = reward + self.beta * future_value

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

    def vfi_vec(self):
        print('Starting Value Function Iteration...')

        # Storage level in next period after action and leakage
        storage_next = self.battery_grid[:, np.newaxis] * (1 - self.sigma) + self.action_grid[np.newaxis, :]
        mask = (storage_next < self.min_battery_capacity) | (storage_next > self.max_battery_capacity)
        storage_next = np.where(mask, np.nan, storage_next)

        # Corresponding action matrix
        action = storage_next - self.battery_grid[:, np.newaxis]  # shape (S, A)

        # Expand dimensions to broadcast with price
        action_broadcast = action[:, :, np.newaxis]  # shape (S, A, 1)
        price_broadcast = self.price_grid[np.newaxis, np.newaxis, :]  # shape (1, 1, P)

        # Variable cost
        variable_cost = self.variable_cost * np.abs(action_broadcast)

        # Compute deterministic profit
        profit = np.where(
            action_broadcast > 0,
            -action_broadcast * price_broadcast / self.eta_charge - variable_cost,
            np.where(
                action_broadcast < 0,
                -action_broadcast * price_broadcast * self.eta_discharge - variable_cost,
                0
            )
        )  # shape (S, A, P)

        # Apply risk-averse utility if enabled
        if self.risk_averse:
            gamma = self.risk_parameter
            V_now = -np.exp(-gamma * profit)
        else:
            V_now = profit

        # Main value function iteration loop
        for it in range(self.max_iteration):
            # Interpolate V across battery grid
            interp = interpolate.interp1d(
                self.battery_grid,
                np.copy(self.V),
                axis=0,
                bounds_error=True  # no extrapolation
            )

            V_next = interp(storage_next)  # shape (S, A, P)

            # Expected value over future prices using transition matrix
            EV = np.einsum("ij,abj->abi", self.price_transitions, V_next)  # shape (S, A, P) → (S, A, P) × (P, P)

            # Total value for each (state, action)
            total_value = V_now + self.beta * EV  # shape (S, A, P)

            # Max over actions
            V_new = np.nanmax(total_value, axis=1)  # shape (S, P)

            # Check for convergence
            if np.max(np.abs(V_new - self.V)) < self.tolerance:
                print(f'Converged in {it + 1} iterations.')
                break

            # Update value and policy
            self.V = V_new
            self.policy = self.action_grid[np.nanargmax(total_value, axis=1)]

        if it == self.max_iteration - 1:
            print(f'Max iterations reached: {self.max_iteration}')

        return self.V, self.policy



    def simulate(self, policy=None):

        # initialize 
        num_periods = len(self.prices_test)
        battery_storage_sim = np.nan + np.zeros(num_periods)
        battery_storage_sim[0] = self.min_battery_capacity
        action_sim = np.nan + np.zeros(num_periods)
        
        profit_sim = np.zeros(num_periods)
        profit_sim[0] = -(self.max_battery_capacity*self.battery_capacity_price + self.power_capacity_price*self.a_bar + self.annual_fixed_cost)

        # interpolate policy function
        interp = RegularGridInterpolator((self.battery_grid, self.price_grid),
                                         policy,
                                         bounds_error=False, # False: allow for extrapolation in prices AND BATTERY (assert not extrapolated)
                                         fill_value=None) # None: extrapolation

        # simulate 
        for t in range(num_periods):
            
            storage = battery_storage_sim[t]
            price = self.prices_test[t]

            action = interp((storage, price))
            storage_next = storage*(1-self.sigma) + action

            assert storage_next >= self.min_battery_capacity and storage_next <= self.max_battery_capacity , f"Storage next is being extrapolated: {storage_next}."

            profit = self.profit_function(action, price)

            profit_sim[t] = profit_sim[t - 1] + profit if t > 0 else profit + profit_sim[0]
            action_sim[t] = action
            if t != num_periods - 1:
                battery_storage_sim[t+1] = storage_next
        
        return battery_storage_sim, profit_sim, action_sim

    def plot_results(self, battery_storage_sim, profit_sim, action_sim):

        # # --- Plot 1: Battery Storage with Prices ---
        # plt.figure(figsize=(10, 6))
        # sc = plt.scatter(range(len(battery_storage_sim)), battery_storage_sim, c=self.prices_test, cmap="coolwarm", edgecolors="k")
        # plt.plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
        # plt.ylabel("Battery Storage Level")
        # plt.title("Battery Storage and Prices Over Time")
        # plt.colorbar(sc, label="Price (EUR/MWh)")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        # --- Plot 2: Prices and Actions ---
        plt.figure(figsize=(10, 6))
        plt.plot(self.prices_test, color="orange", label="Test Prices", alpha=0.5)
        plt.axhline(np.mean(self.prices_test), color='gray', linestyle='--', label='Mean Price')
        plt.scatter(np.where(action_sim > self.a_bar - 0.5)[0], self.prices_test[action_sim > self.a_bar - 0.5], color="blue", label="Charge", s=20)
        plt.scatter(np.where(action_sim < -self.a_bar + 0.5)[0], self.prices_test[action_sim < -self.a_bar + 0.5], color="red", label="Discharge", s=20)
        plt.ylabel("Price (EUR/MWh)")
        plt.title("Prices and Charge/Discharge Actions")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 3: Profit Over Time ---
        plt.figure(figsize=(10, 6))
        plt.plot(profit_sim, color="green", label="Cumulative Profit")
        plt.xlabel("Time Periods")
        plt.ylabel("Profit")
        plt.title("Profit Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 4: Policy Heatmap ---
        non_zero = self.policy != 0
        norm = mcolors.TwoSlopeNorm(vmin=self.policy[non_zero].min(), vcenter=0, vmax=self.policy[non_zero].max())
        cmap = plt.get_cmap("seismic_r")
        plt.figure(figsize=(10, 6))
        im = plt.imshow(cmap(norm(self.policy)), origin='lower', aspect='auto')
        plt.title("Policy Visualization")
        plt.xlabel("Prices (X)")
        plt.ylabel("Battery Storage (Y)")
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="Policy Value")
        plt.tight_layout()
        plt.show()

        # --- Plot 5: Value Function ---
        plt.figure(figsize=(10, 6))
        selected_price_indices = np.linspace(0, self.num_price_levels - 1, 8, dtype=int)

        for idx in selected_price_indices:
            price_level = self.price_grid[idx]
            plt.plot(self.battery_grid, self.V[:, idx], label=f'Price ≈ {price_level:.2f}')

        plt.xlabel('Battery Storage Level')
        plt.ylabel('Value Function')
        plt.title('Value Function vs Storage Capacity for Different Price Points')
        plt.legend(title="Price Level", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 6: First Day Simulation ---
        hours = np.arange(72)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Plot price
        ax1.plot(hours, self.prices_test[:72], color='orange', label='Price (EUR/MWh)')
        ax1.set_ylabel('Price (EUR/MWh)', color='orange')
        ax1.tick_params(axis='y', labelcolor='orange')

        # Plot battery storage
        ax2.step(hours, battery_storage_sim[:72], color='blue', label='Battery Storage', where='mid')
        ax2.set_ylabel('Battery Storage Level', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Titles and grid
        ax1.set_xlabel('Hour of Day')
        plt.title('Simulated Battery Storage and Prices – First Day (72 Hours)')
        ax1.grid(True)
        plt.tight_layout()
        plt.show()



