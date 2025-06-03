import numpy as np
import pandas as pd

class PriceGenerator:
    def __init__(self, 
                 simulate=True,
                 num_price_levels=30,
                 num_sim_periods=5000,
                 price_avg=50,
                 mean_reversion=0.3,
                 p_variance=100.0,
                 data_path='./dk2price_20000101_20191231.csv'):

        self.simulate = simulate
        self.num_price_levels = num_price_levels
        self.num_sim_periods = num_sim_periods
        self.price_avg = price_avg
        self.alpha = mean_reversion
        self.sigma2 = p_variance
        self.data_path = data_path

        # compute and expose these
        self.prices = None
        self.price_grid = None
        self.price_transitions = None

        self._generate()

        # assert  
        assert all(x is not None for x in [self.prices, self.price_grid, self.price_transitions]), "Prices, price grid, or price transitions were not generated"

    def _generate(self):
        if self.simulate:
            print('... simulating price data')
            self._simulate_prices()
        else:
            print('... loading empirical price data')
            self._load_and_process_data()

    def _simulate_prices(self):
        self.price_grid = np.linspace(0, 100, self.num_price_levels)
        self.price_transitions = self._build_simulated_transition_matrix()
        self.prices = self._simulate_price_series()

    def _build_simulated_transition_matrix(self):
        N = self.num_price_levels
        T = np.zeros((N, N))
        for i, p_i in enumerate(self.price_grid):
            mu = (1 - self.alpha) * p_i + self.alpha * self.price_avg
            exponent = - (self.price_grid - mu) ** 2 / (2 * self.sigma2)
            probs = np.exp(exponent)
            T[i, :] = probs / probs.sum()
        assert np.allclose(T.sum(axis=1), 1), "Rows must sum to 1."
        return T

    def _simulate_price_series(self):
        np.random.seed(2)
        series = np.zeros(self.num_sim_periods)
        current_idx = len(self.price_grid) // 2
        series[0] = self.price_grid[current_idx]

        for t in range(1, self.num_sim_periods):
            next_idx = np.random.choice(
                np.arange(self.num_price_levels),
                p=self.price_transitions[current_idx]
            )
            series[t] = self.price_grid[next_idx]
            current_idx = next_idx

        return series

    def _load_and_process_data(self):
        df = pd.read_csv(self.data_path, sep=';')
        df = df.dropna(subset=['SpotPriceEUR']).copy()
        df['SpotPriceEUR'] = df['SpotPriceEUR'].str.replace(',', '.').astype(float)
        df['date'] = pd.to_datetime(df['HourDK'], utc=True)
        df.set_index('date', inplace=True)
        df = df[(df.SpotPriceEUR > -10) & (df.SpotPriceEUR < 120)]

        self.prices = df['SpotPriceEUR'].values
        self.price_grid = np.linspace(self.prices.min(), self.prices.max(), self.num_price_levels)
        self.price_transitions = self._compute_empirical_transitions()

    def _compute_empirical_transitions(self):
        price_indices = np.array([np.argmin(np.abs(self.price_grid - p)) for p in self.prices])
        T = np.zeros((self.num_price_levels, self.num_price_levels))
        assert np.unique(price_indices).size == self.num_price_levels, "Not all price indices are observed in data."
        
        for t in range(len(price_indices) - 1):
            i, j = price_indices[t], price_indices[t + 1]
            T[i, j] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        T = np.divide(T, row_sums, where=row_sums != 0)
        assert np.allclose(T.sum(axis=1), 1), "Transition rows must sum to 1."
        return T
