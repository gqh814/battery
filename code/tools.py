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
                 p_variance = 100 
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

        # added 
        self.sigma = sigma # percentage per hour # approximated 
        self.variable_cost = variable_cost # EUR/kWh
        self.battery_capacity_price = battery_capacity_price # EUR/kWh
        self.power_capacity_price = power_capacity_price # EUR/kW
        self.annual_fixed_cost = annual_fixed_cost # EUR/kW per year.
        self.a_bar = a_bar

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
            self.price_grid = np.linspace(0, 100, self.num_price_levels)
            price_avg = 50
            num_periods = 1000

            simulator = PriceSimulator(self.price_grid, price_avg, num_periods, alpha=self.mean_reversion, sigma2=self.p_variance)
            self.prices_test = simulator.simulate_prices()
            self.price_transitions = simulator.price_transitions

        else:
            self._load_data()
            self._compute_price_transitions()

    def _load_data(self,price_data_path='../data/entsoe_price_DK_1_20150101_20240101.csv'):

        # Load and process price data
        dk1_p = pd.read_csv(price_data_path)
        dk1_p['date'] = pd.to_datetime(dk1_p['date'], utc=True)
        dk1_p['year'] = dk1_p['date'].dt.year
        # dk1_p['month'] = dk1_p['date'].dt.month
        # dk1_p['day'] = dk1_p['date'].dt.day
        
        # Training and test data
        dk1_p_train = dk1_p[dk1_p['year'] == 2018]
        dk1_p_test = dk1_p[dk1_p['year'] == 2019]
        
        self.prices_train = dk1_p_train.DK_1_day_ahead_price.to_numpy()
        self.prices_test = dk1_p_test.DK_1_day_ahead_price.to_numpy()
        self.price_grid = np.linspace(np.min(self.prices_train), np.max(self.prices_train), self.num_price_levels)

    def _compute_price_transitions(self):
        price_transitions = np.zeros((self.num_price_levels, self.num_price_levels))
        price_indices_training = np.array([np.argmin(np.abs(self.price_grid - p)) for p in self.prices_train])

        assert np.unique(price_indices_training).size == self.num_price_levels, "Not all price indices are observed in training data."

        for t in range(len(price_indices_training) - 1):
            i = price_indices_training[t]
            j = price_indices_training[t + 1]
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
        print('heyo')
        
        # storage in next period and actions.  
        storage_next = self.battery_grid[:, np.newaxis]*(1-self.sigma) + self.action_grid[np.newaxis, :]
        mask = (storage_next < self.min_battery_capacity) | (storage_next > self.max_battery_capacity)
        storage_next = np.where(mask, np.nan, storage_next)
        action = storage_next - self.battery_grid[:, np.newaxis]

        # utility now
        V_now = np.where(action > 0, -action / self.eta_charge, -action * self.eta_discharge)
        V_now = V_now[:, :, np.newaxis] * self.price_grid[np.newaxis, np.newaxis, :]

        for it in range(self.max_iteration):

            interp = interpolate.interp1d(self.battery_grid, 
                                          np.copy(self.V), 
                                          axis=0, # location of battery grid in V_new
                                          bounds_error=True) # True = no extrapolation

            V_next = interp(storage_next) 

            # EV_loop = np.zeros((self.num_storage_levels, self.num_actions, self.num_price_levels))
            # for p in range(self.num_price_levels): # past 
            #     for f in range(self.num_price_levels): # future
            #         EV_loop[:,:,p] +=  self.price_transitions[p, f] * V_next[:,:,f]
            
            EV = np.einsum("ij,abj->abi", self.price_transitions, V_next)
            
            total_value = V_now + self.beta * EV

            V_new = np.nanmax(total_value, axis=1) 
            
            # Check for convergence
            if np.max(np.abs(V_new - self.V)) < self.tolerance:
                print(f'Converged in {it+1} iterations.')
                break

            self.V = V_new
            self.policy = self.action_grid[np.nanargmax(total_value, axis=1)]

        if it == self.max_iteration - 1:
            print(f'Max iterations reached: {self.max_iteration}')
            
        return self.V, self.policy


    def simulate(self):

        # initialize 
        num_periods = len(self.prices_test)
        battery_storage_sim = np.nan + np.zeros(num_periods)
        battery_storage_sim[0] = self.min_battery_capacity
        action_sim = np.nan + np.zeros(num_periods)
        
        profit_sim = np.zeros(num_periods)
        profit_sim[0] = -(self.max_battery_capacity*self.battery_capacity_price + self.power_capacity_price*self.a_bar + self.annual_fixed_cost)

        # interpolate policy function
        interp = RegularGridInterpolator((self.battery_grid, self.price_grid),
                                         self.policy,
                                         bounds_error=False, # False : allow for extrapolation in prices AND BATTERY
                                         fill_value=None) # None: extrapolation

        # simulate 
        for t in range(num_periods):
            
            storage = battery_storage_sim[t]
            price = self.prices_test[t]

            action = interp((storage, price))
            storage_next = storage*(1-self.sigma) + action

            assert storage_next >= self.min_battery_capacity and storage_next <= self.max_battery_capacity , f"Storage next is being extrapolated: {storage_next}."

            profit = self.utility_function(action, price)

            profit_sim[t] = profit_sim[t - 1] + profit if t > 0 else profit + profit_sim[0]
            action_sim[t] = action
            if t != num_periods - 1:
                battery_storage_sim[t+1] = storage_next
        
        return battery_storage_sim, profit_sim, action_sim

    def plot_results(self, battery_storage_sim, profit_sim, action_sim):
        fig, axs = plt.subplots(4, 1, figsize=(10, 35), sharex=False)

        axs[0].scatter(range(len(battery_storage_sim)), battery_storage_sim, c=self.prices_test, cmap="coolwarm", edgecolors="k")
        axs[0].plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
        axs[0].set_ylabel("Battery Storage Level")
        axs[0].set_title("Battery Storage, Prices, and Profit Over Time")
        axs[0].grid(True)

        # Price Plot with action markers
        axs[1].plot(self.prices_test, color="orange", label="Test Prices", alpha=0.5)
        # Overlay markers
        # plot hline with mean of prices_test
        axs[1].axhline(y=np.mean(self.prices_test), color='gray', linestyle='--', label='Mean Price')
        charge_times = np.where(action_sim > 0.5)[0]
        discharge_times = np.where(action_sim < -0.5)[0]
        axs[1].scatter(discharge_times, self.prices_test[discharge_times], color="red", label="Discharge", s=20)
        axs[1].scatter(charge_times, self.prices_test[charge_times], color="blue", label="Charge", s=20)

        axs[1].set_ylabel("Price (EUR/MWh)")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(profit_sim, color="green", label="Cumulative Profit")
        axs[2].set_xlabel("Time Periods")
        axs[2].set_ylabel("Profit")
        axs[2].legend()
        axs[2].grid(True)

        # Normalize data, exclude zeros from normalization
        non_zero_mask = self.policy != 0
        vmin = self.policy[non_zero_mask].min()
        vmax = self.policy[non_zero_mask].max()
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = plt.get_cmap("seismic_r")

        # Create RGB image from colormap, then set zeros to black
        rgba_img = cmap(norm(self.policy))

        # Plot
        im = axs[3].imshow(rgba_img, origin='lower', aspect='auto')
        axs[3].set_title("Policy Visualization")
        axs[3].set_xlabel("Prices (X)")
        axs[3].set_ylabel("Battery Storage (Y)")

        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[3], orientation='vertical')
        cbar.set_label("Policy Value")

        plt.tight_layout()
        plt.show()