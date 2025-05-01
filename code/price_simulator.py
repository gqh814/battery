import numpy as np

class PriceSimulator:
    def __init__(self, price_grid, price_avg, num_sim_periods, alpha=0.3, sigma2=100.0):
        self.price_grid = price_grid
        self.price_avg = price_avg
        self.num_price_levels = len(price_grid)
        self.num_sim_periods = num_sim_periods
        self.alpha = alpha
        self.sigma2 = sigma2
        self.price_series = np.zeros(num_sim_periods)
        self.price_transitions = self.build_price_transitions()

    def build_price_transitions(self):
        """Construct a full-grid mean-reverting Gaussian transition matrix"""
        N = self.num_price_levels
        T = np.zeros((N, N))
        for i, p_i in enumerate(self.price_grid):
            mu_i = (1 - self.alpha) * p_i + self.alpha * self.price_avg
            exponent = - (self.price_grid - mu_i) ** 2 / (2 * self.sigma2)
            raw = np.exp(exponent)
            T[i, :] = raw / raw.sum()
        assert np.allclose(T.sum(axis=1), 1), "Rows must sum to 1."
        return T

    def simulate_prices(self):
        print("Simulating price series...")
        np.random.seed(1)
        current_idx = len(self.price_grid) // 2
        self.price_series[0] = self.price_grid[current_idx]

        for t in range(1, self.num_sim_periods):
            next_idx = np.random.choice(
                np.arange(self.num_price_levels),
                p=self.price_transitions[current_idx]
            )
            self.price_series[t] = self.price_grid[next_idx]
            current_idx = next_idx

        return self.price_series
