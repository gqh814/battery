# standard packages
import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

# custom package
from price_generator import PriceGenerator

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
        
        # assign all parameters as attributes 
        for k, v in locals().items():
            if k != 'self': setattr(self, k, v)

        # setup price generator
        pg = PriceGenerator(simulate=self.simulate_prices,
                            num_price_levels=self.num_price_levels,
                            mean_reversion=self.mean_reversion,
                            p_variance=self.p_variance)

        self.prices = pg.prices 
        self.price_grid = pg.price_grid
        self.price_transitions = pg.price_transitions

        # grids
        self.battery_grid = np.linspace(self.min_battery_capacity, self.max_battery_capacity, self.num_storage_levels)
        self.action_grid = np.linspace(-self.a_bar, self.a_bar, self.num_actions)

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
    
    def _prepare_transition_inputs(self):
        """
        returns:
        - storage_next: (S, A) next storage levels after action
        - action_broadcast: (S, A, 1) action levels broadcasted for utility computation
        - price_broadcast: (1, 1, P) price levels broadcasted for utility computation
        """

        storage_next = self.battery_grid[:, np.newaxis] * (1 - self.sigma) + self.action_grid[np.newaxis, :]
        mask = (storage_next < self.min_battery_capacity) | (storage_next > self.max_battery_capacity)
        storage_next = np.where(mask, np.nan, storage_next) # (S, A), nan to invalid actions

        action = storage_next - self.battery_grid[:, np.newaxis] # (S, A)
        action_broadcast = action[:, :, np.newaxis]  # (S, A, 1)
        price_broadcast = self.price_grid[np.newaxis, np.newaxis, :]  # (1, 1, P)

        return storage_next, action_broadcast, price_broadcast

    def vfi_vec(self):
        print('Starting Vectorized Value Function Iteration')

        # initialize
        self.policy = np.zeros((self.num_storage_levels, self.num_price_levels))  # (S, P)
        self.V = np.zeros((self.num_storage_levels, self.num_price_levels))       # (S, P)

        # utility now 
        storage_next, action_broadcast, price_broadcast = self._prepare_transition_inputs()
        V_now = self._compute_utility(action_broadcast, price_broadcast)

        sum_change_P = 0
        for it in range(self.max_iteration): 

            # utility next period
            interp = interpolate.interp1d(
                self.battery_grid,
                np.copy(self.V),
                axis=0,
                bounds_error=True  # True = no extrapolation
            )
            V_next = interp(storage_next)  # (S, A, P)
            EV = np.einsum("ij,abj->abi", self.price_transitions, V_next)  # expected value given current price, i.

            total_value = V_now + self.beta * EV  # (S, A, P)
            
            # optimal value
            V_new = np.nanmax(total_value, axis=1)  # (S, P)

            # check for convergence
            check_V = np.max(np.abs(V_new - self.V))
            check_P = np.max(np.abs(self.policy - self.action_grid[np.nanargmax(total_value, axis=1)]))

            if check_V < self.tolerance:
                print(f'Converged after {it + 1} iterations.')
                break
        
            sum_change_P += check_P
            if it % 1000 == 0 or it < 10:
                #print(f"Iteration {it}: change in value = {check_V:.3e}, cumulative change in policy = {sum_change_P:.2f}")
                sum_change_P = 0 

            # update 
            self.V = V_new
            self.policy = self.action_grid[np.nanargmax(total_value, axis=1)]

        if it == self.max_iteration - 1:
            print(f'Max iterations reached: {self.max_iteration}')

        return self.V, self.policy
    
    def _make_Q(self, P_new): 
 
        n_storage, n_price = P_new.shape
        n_states = n_storage * n_price
        
        curr_states = np.arange(n_states)
        curr_price = curr_states % n_price
        curr_storage = curr_states // n_price

        # convert storage indices to actual storage levels
        curr_storage_vals = self.battery_grid[curr_storage]  # (n_states,)

        # next storage levels (continuous)
        next_storage_vals = curr_storage_vals*(1-self.sigma) + P_new.flatten()  # (n_states,)

        # clip to battery bounds
        next_storage_vals = np.clip(next_storage_vals, self.min_battery_capacity, self.max_battery_capacity)

        # find lower and upper indices for interpolation
        idx_upper = np.searchsorted(self.battery_grid, next_storage_vals, side='right')
        idx_upper = np.clip(idx_upper, 1, n_storage - 1)
        idx_lower = idx_upper - 1

        # compute weights for interpolation
        s_low = self.battery_grid[idx_lower]
        s_high = self.battery_grid[idx_upper]
        w_high = (next_storage_vals - s_low) / (s_high - s_low)
        w_low = 1.0 - w_high

        assert not np.isnan(w_high).any() and not np.isnan(w_low).any(), "Weights contain NaN values"
         
        # initialize Q
        Q = np.zeros((n_states, n_states), dtype=np.float64)

        # for each interpolation bin, calculate next states and fill Q
        for idxs, weights in zip([idx_lower, idx_upper], [w_low, w_high]):

            # next states = storage idx * n_price + next price idx
            next_states = idxs[:, None] * n_price + np.arange(n_price)[None, :]
            
            # +rice transition probs for current prices
            price_probs = self.price_transitions[curr_price]  # (n_states, n_price)

            # flatten arrays to fill Q matrix
            rows = np.repeat(curr_states, n_price)
            cols = next_states.flatten()
            vals = (price_probs * weights[:, None]).flatten()

            Q[rows, cols] += vals

        assert np.allclose(Q.sum(axis=1), 1), "Rows of Q must sum to 1"

        return Q

    def pfi_vec(self,dampening=False):
        print('Starting Vectorized Policy Function Iteration')

        # initialize
        self.policy = np.zeros((self.num_storage_levels, self.num_price_levels))  # (S, P)
        self.V = np.zeros((self.num_storage_levels, self.num_price_levels))       # (S, P)

        # utility now
        storage_next, action_broadcast, price_broadcast = self._prepare_transition_inputs()
        V_now = self._compute_utility(action_broadcast, price_broadcast) # (S, A, P)

        alpha = 1 # dampening parameter

        for it in range(self.max_iteration):

            # === POLICY EVALUATION ===
            price = self.price_grid[np.newaxis, :]  # (1, P)
            V_cur_policy = self._compute_utility(self.policy, price)  # (S, P)

            # solve for V: (I - βQ) V = V_cur_policy')
            Q = self._make_Q(self.policy)  # (S*P, S*P)
            I_bQ = np.eye(Q.shape[0]) - self.beta * Q
            V_flat = np.linalg.solve(I_bQ, V_cur_policy.flatten(order='C'))
            V = V_flat.reshape(self.num_storage_levels, self.num_price_levels)

            # === POLICY IMPROVEMENT ===
            interp = interpolate.interp1d(self.battery_grid, V, axis=0, bounds_error=True)
            V_next = interp(storage_next)  # (S, A, P)
            EV = np.einsum("ij,abj->abi", self.price_transitions, V_next)  # (S, A, P) expected value given current price, i.
            total_value = V_now + self.beta * EV  # (S, A, P)

            P_new = self.action_grid[np.nanargmax(total_value, axis=1) ]  # (S, P)

            # check convergence
            policy_change = np.max(np.abs(P_new - self.policy))
            value_change = np.max(np.abs(V - self.V))

            print(f"Iter {it:03d}: ΔV = {value_change:.3e}, Δπ = {policy_change:.10f}")

            if value_change < self.tolerance and it > 0:
                print(f"Converged after {it+1} iterations.")
                self.V = V
                self.policy = P_new
                
                return self.V, self.policy
            
            if dampening: # reduce alpha every 20 iterations
                if it > 0 and it % 20 == 0:
                    alpha = max(alpha - 0.1, 0.1)
                    print(f"Dampening: alpha={alpha}")

            # update
            self.V = V
            self.policy = alpha * P_new + (1 - alpha) * self.policy
        
        if it == self.max_iteration - 1:
            print(f'Max iterations reached: {self.max_iteration}')

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