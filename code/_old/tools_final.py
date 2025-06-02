import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from price_simulator import PriceSimulator

from scipy import interpolate
from scipy.sparse.linalg import LinearOperator, gmres
import numpy as np

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
                 max_iteration=10_000, 
                 tolerance=1e-5,
                 beta = 0.99,
                 sigma = 0.1/100 / 24, 
                 variable_cost=2.1,
                 a_bar=7.2,
                 risk_averse=True,
                 risk_parameter=0.01,
                 simulate_prices=True,
                 mean_reversion = 0.3,
                 p_variance = 100,
                 ):
        
        # assign all parameters as attributes using locals()
        for k, v in locals().items():
            if k != 'self': setattr(self, k, v)

        # prices
        self.prices, self.price_grid, self.price_transitions = self._price_generator(simulate=self.simulate_prices)

        # grids
        self.battery_grid = np.linspace(self.min_battery_capacity, self.max_battery_capacity, self.num_storage_levels)
        self.action_grid = np.linspace(-self.a_bar, self.a_bar, self.num_actions)

        # value and policy function
        self.V = np.zeros((self.num_storage_levels, self.num_price_levels))
        self.policy = np.zeros((self.num_storage_levels, self.num_price_levels), dtype=float)

    def _price_generator(self, simulate=True):

        if simulate:
            print('... simulating price data')
            price_grid = np.linspace(0, 100, self.num_price_levels)

            price_avg = 50
            num_periods = 72

            simulator = PriceSimulator(price_grid, price_avg, num_periods, alpha=self.mean_reversion, sigma2=self.p_variance)
            prices = simulator.simulate_prices()
            price_transitions = simulator.price_transitions

            return prices, price_grid, price_transitions

        else:
            print('... loading price data')
            df = self._load_data()
            prices, price_grid, price_transitions = self._compute_price_transitions(df)

            return prices, price_grid, price_transitions

    def _load_data(self):
        
        # Read the data
        price_data_path = '../data/dk2price_20000101_20191231.csv'
        dk2_p = pd.read_csv(price_data_path, sep=';')

        # Drop observations where SpotPriceEUR is NA
        dropped_obs = dk2_p[dk2_p.SpotPriceEUR.isna()]
        df = dk2_p.dropna(subset=['SpotPriceEUR']).copy() 

        # Convert HourDK to datetime and create a 'year' column
        df['date'] = pd.to_datetime(df['HourDK'], utc=True)
        df['year'] = df['date'].dt.year  # No need for .loc here
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        df['SpotPriceEUR'] = df['SpotPriceEUR'].str.replace(',', '.').astype(float) # MwH
        # df = df[['SpotPriceEUR']].resample('D').mean() # turn data into daily

        # filter prices
        max_price = 120
        min_price = -10
        cond = (df.SpotPriceEUR > min_price) & (df.SpotPriceEUR < max_price)
        df = df[cond]

        return df

    def _compute_price_transitions(self, df):

        prices = df["SpotPriceEUR"].values
        price_grid = np.linspace(prices.min(), prices.max(), self.num_price_levels)

        price_transitions = np.zeros((self.num_price_levels, self.num_price_levels))
        price_indices = np.array([np.argmin(np.abs(price_grid - p)) for p in prices])

        assert np.unique(price_indices).size == self.num_price_levels, "Not all price indices are observed in training data."

        for t in range(len(price_indices) - 1):
            i = price_indices[t]
            j = price_indices[t + 1]
            price_transitions[i, j] += 1
        
        # Normalize rows
        row_sums = price_transitions.sum(axis=1, keepdims=True)
        price_transitions = np.divide(price_transitions, row_sums, where=row_sums != 0)
        assert np.allclose(price_transitions.sum(axis=1), 1), "Not all rows sum to 1."

        return prices, price_grid, price_transitions

    def _compute_profit(self, action, price):
        cost = self.variable_cost * np.abs(action)
        profit = np.where(
            action > 0,
            -action * price / self.eta_charge - cost,
            np.where(action < 0, -action * price * self.eta_discharge - cost, 0)
        )

        return profit 

    def _compute_utility(self, action, price):
        profit = self._compute_profit(action, price)
        V_now = -np.exp(-self.risk_parameter * profit) if self.risk_averse else profit
        return V_now
    
    def _construct_state_action_price_grid(self):
        """
        returns:
        - storage_next: (S, A) next storage levels after action
        - action_broadcast: (S, A, 1) action levels broadcasted for utility computation
        - price_broadcast: (1, 1, P) price levels broadcasted for utility computation
        """

        storage_next = self.battery_grid[:, np.newaxis] * (1 - self.sigma) + self.action_grid[np.newaxis, :]
        mask = (storage_next < self.min_battery_capacity) | (storage_next > self.max_battery_capacity)
        storage_next = np.where(mask, np.nan, storage_next) # (S, A)

        action = storage_next - self.battery_grid[:, np.newaxis] # (S, A)
        action_broadcast = action[:, :, np.newaxis]  # (S, A, 1)
        price_broadcast = self.price_grid[np.newaxis, np.newaxis, :]  # (1, 1, P)

        return storage_next, action_broadcast, price_broadcast

    def vfi_vec(self):
        print('Starting Vectorized Value Function Iteration...')

        storage_next, action_broadcast, price_broadcast = self._construct_state_action_price_grid()
        V_now = self._compute_utility(action_broadcast, price_broadcast)

        sum_change_P = 0
        for it in range(self.max_iteration): 

            # Utility next period
            interp = interpolate.interp1d(
                self.battery_grid,
                np.copy(self.V),
                axis=0,
                bounds_error=True  # True = no extrapolation
            )
            V_next = interp(storage_next)  # (S, A, P)
            EV = np.einsum("ij,abj->abi", self.price_transitions, V_next)  # Expected value given current price, i.

            total_value = V_now + self.beta * EV  # (S, A, P)
            
            # Optimal value and policy
            V_new = np.nanmax(total_value, axis=1)  # (S, P)
            check_V = np.max(np.abs(V_new - self.V))
            check_P = np.max(np.abs(self.policy - self.action_grid[np.nanargmax(total_value, axis=1)]))

            # Check for convergence
            if check_V < self.tolerance:
                print(f'Converged after {it + 1} iterations.')
                break
        
            sum_change_P += check_P
            if it % 1000 == 0 or it < 10:
                print(f"Iteration {it}: change in value = {check_V:.3e}, cumulative change in policy = {sum_change_P:.2f}")
                sum_change_P = 0 

            # Update 
            self.V = V_new
            self.policy = self.action_grid[np.nanargmax(total_value, axis=1)]

        if it == self.max_iteration - 1:
            print(f'Max iterations reached: {self.max_iteration}')

        return self.V, self.policy
    

    def make_Q_grid(self, P_new):
        """
        Constructs the transition matrix Q for the energy storage model.
        Parameters:
        - P_new: (S, P) array of new actions (power levels) to be applied to the current states.
        Returns:
        - Q: (S*P, S*P) transition matrix where Q[i, j] is the probability of transitioning from state i to state j.
        """

        n_storage, n_price = P_new.shape
        n_states = n_storage * n_price
        transition_price = self.price_transitions  # (n_price, n_price)

        curr_states = np.arange(n_states)
        curr_price = curr_states % n_price
        curr_storage = curr_states // n_price
        curr_storage_vals = self.battery_grid[curr_storage]

        # Flatten actions and compute new storage values
        next_storage_vals = np.clip(
            curr_storage_vals + P_new.flatten(),
            self.min_battery_capacity,
            self.max_battery_capacity,
        )

        # Interpolation indices and weights
        idx_upper = np.searchsorted(self.battery_grid, next_storage_vals, side='right')
        idx_upper = np.clip(idx_upper, 1, n_storage - 1)
        idx_lower = idx_upper - 1

        s_low = self.battery_grid[idx_lower]
        s_high = self.battery_grid[idx_upper]
        w_high = (next_storage_vals - s_low) / (s_high - s_low)
        w_low = 1.0 - w_high

        assert not np.isnan(w_high).any() and not np.isnan(w_low).any(), "NaN in interpolation weights"

        # Construct row (source), column (destination), and value arrays for Q
        all_price_indices = np.arange(n_price)

        # Next states for both lower and upper interpolated storage values
        next_states_lower = idx_lower[:, None] * n_price + all_price_indices[None, :]
        next_states_upper = idx_upper[:, None] * n_price + all_price_indices[None, :]

        # Price transition probabilities for current prices
        price_probs = transition_price[curr_price]  # (n_states, n_price)

        # Row indices repeated for each price
        rows = np.repeat(curr_states, n_price)  # (n_states * n_price,)

        # Flatten everything
        cols_lower = next_states_lower.flatten()
        cols_upper = next_states_upper.flatten()
        vals_lower = (price_probs * w_low[:, None]).flatten()
        vals_upper = (price_probs * w_high[:, None]).flatten()

        # Fill Q matrix using advanced indexing
        Q = np.zeros((n_states, n_states), dtype=np.float64)
        np.add.at(Q, (rows, cols_lower), vals_lower)
        np.add.at(Q, (rows, cols_upper), vals_upper)

        assert np.allclose(Q.sum(axis=1), 1), "Rows of Q must sum to 1"

        return Q

    def pfi_vec(self):
        print('... Starting Policy Function Iteration')

        self.policy = np.zeros((self.num_storage_levels, self.num_price_levels))  # (S, P)
        self.V = np.zeros((self.num_storage_levels, self.num_price_levels))       # (S, P)

        storage_next, action_broadcast, price_broadcast = self._construct_state_action_price_grid()
        V_now = self._compute_utility(action_broadcast, price_broadcast) # (S, A, P)
        
        for it in range(self.max_iteration):

            # === POLICY EVALUATION ===
            price = self.price_grid[np.newaxis, :]  # (1, P)
            V_cur_policy = self._compute_utility(self.policy, price)  # (S, P)

            # Solve for V: (I - βQ) V = V_cur_policy')
            Q = self.make_Q_grid(self.policy)  # (S*P, S*P)
            I_bQ = np.eye(Q.shape[0]) - self.beta * Q
            V_flat = np.linalg.solve(I_bQ, V_cur_policy.flatten(order='C'))
            V = V_flat.reshape(self.num_storage_levels, self.num_price_levels)

            # === POLICY IMPROVEMENT ===
            interp = interpolate.interp1d(self.battery_grid, V, axis=0, bounds_error=True)
            V_next = interp(storage_next)  # (S, A, P)
            EV = np.einsum("ij,abj->abi", self.price_transitions, V_next)  # shape (S, A, P)
            total_value = V_now + self.beta * EV  # (S, A, P)

            best_action_idx = np.nanargmax(total_value, axis=1)  # (S, P)
            P_new = self.action_grid[best_action_idx]  # (S, P)

            policy_change = np.max(np.abs(P_new - self.policy))
            value_change = np.max(np.abs(V - self.V))

            print(f"Iter {it:03d}: ΔV = {value_change:.3e}, Δπ = {policy_change:.4f}")

            # Check convergence
            if policy_change < self.tolerance:
                print(f"Converged after {it+1} iterations.")
                self.V = V
                self.policy = P_new
                
                return self.V, self.policy

            # Update
            self.V = V
            self.policy = P_new

        print("Did not fully converge.")
        return self.V, self.policy

    def simulate(self, policy=None):

        # initialize 
        num_periods = len(self.prices)
        battery_storage_sim = np.nan + np.zeros(num_periods)
        battery_storage_sim[0] = self.min_battery_capacity
        action_sim = np.nan + np.zeros(num_periods)  
        profit_sim = np.zeros(num_periods)

        # interpolate policy function
        interp = RegularGridInterpolator((self.battery_grid, self.price_grid),
                                         policy,
                                         bounds_error=False, # False: allow for extrapolation in prices AND BATTERY (assert not extrapolated)
                                         fill_value=None) # None: extrapolation

        # simulate 
        for t in range(num_periods):
            
            storage = battery_storage_sim[t]
            price = self.prices[t]

            action = interp((storage, price))
            storage_next = storage*(1-self.sigma) + action

            assert storage_next >= self.min_battery_capacity and storage_next <= self.max_battery_capacity , f"Storage next is being extrapolated: {storage_next}."

            profit = self._compute_profit(action, price)

            profit_sim[t] = profit_sim[t - 1] + profit if t > 0 else profit + profit_sim[0]
            action_sim[t] = action
            if t != num_periods - 1:
                battery_storage_sim[t+1] = storage_next
        
        return battery_storage_sim, profit_sim, action_sim



##### graveyard ######



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

    def plot_results(self, battery_storage_sim, profit_sim, action_sim):

        # --- Plot 1: Battery Storage with Prices ---
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(range(len(battery_storage_sim)), battery_storage_sim, c=self.prices, cmap="coolwarm", edgecolors="k")
        plt.plot(battery_storage_sim, linestyle="-", alpha=0.5, color="gray")
        plt.ylabel("Battery Storage Level")
        plt.title("Battery Storage and Prices Over Time")
        plt.colorbar(sc, label="Price (EUR/MWh)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Prices and Actions ---
        plt.figure(figsize=(10, 6))
        plt.plot(self.prices, color="orange", label="Test Prices", alpha=0.5)
        plt.axhline(np.mean(self.prices), color='gray', linestyle='--', label='Mean Price')
        plt.scatter(np.where(action_sim > self.a_bar - 0.5)[0], self.prices[action_sim > self.a_bar - 0.5], color="blue", label="Charge", s=20)
        plt.scatter(np.where(action_sim < -self.a_bar + 0.5)[0], self.prices[action_sim < -self.a_bar + 0.5], color="red", label="Discharge", s=20)
        plt.ylabel("Price (EUR/MWh)")
        plt.title("Prices and Charge/Discharge Actions")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 3: Profit Over Time ---
        plt.figure(figsize=(10, 6))
        plt.plot(profit_sim, color="blue", label="Cumulative Profit")
        plt.xlabel("Time Periods")
        plt.ylabel("Profit")
        plt.title("Profit Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f'you earned: {profit_sim[-1]-profit_sim[0]}')

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
        ax1.plot(hours, self.prices[:72], color='orange', label='Price (EUR/MWh)')
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



