import numpy as np

class PriceSimulator:
    def __init__(self, price_grid, price_avg, num_sim_periods):
        self.price_grid = price_grid
        self.price_avg = price_avg
        self.num_price_levels = len(price_grid)
        self.num_sim_periods = num_sim_periods
        self.price_series = np.zeros(num_sim_periods)
        self.price_transitions = self.build_price_transitions()

    def mean_reverting_probs_extended(self, price, price_avg, scale=0.1):
        """Return probabilities for [big down, small down, stay, small up, big up]"""
        distance = price - price_avg

        weight_big_down  = 0.1 * (1 / (1 + np.exp(-scale * distance)))
        weight_down      = 0.3 * (1 / (1 + np.exp(-scale * distance)))
        weight_stay      = 0.2 + 0.2 * np.exp(-abs(distance) / 10)
        weight_up        = 0.3 * (1 / (1 + np.exp(scale * distance)))
        weight_big_up    = 0.1 * (1 / (1 + np.exp(scale * distance)))

        weights = np.array([
            weight_big_down,
            weight_down,
            weight_stay,
            weight_up,
            weight_big_up
        ])
        return weights / np.sum(weights)

    def build_price_transitions(self):
        """Construct the transition probability matrix using mean-reverting logic"""
        transitions = np.zeros((self.num_price_levels, self.num_price_levels))

        for i, price in enumerate(self.price_grid):
            probs = self.mean_reverting_probs_extended(price, self.price_avg)

            # Assign probabilities with reflecting boundaries
            if i >= 2:
                transitions[i, i - 2] += probs[0]
            else:
                transitions[i, i] += probs[0]

            if i >= 1:
                transitions[i, i - 1] += probs[1]
            else:
                transitions[i, i] += probs[1]

            transitions[i, i] += probs[2]

            if i < self.num_price_levels - 1:
                transitions[i, i + 1] += probs[3]
            else:
                transitions[i, i] += probs[3]

            if i < self.num_price_levels - 2:
                transitions[i, i + 2] += probs[4]
            else:
                transitions[i, i] += probs[4]

        # Normalize rows to sum to 1
        transitions /= transitions.sum(axis=1, keepdims=True)

        # Optional check
        assert np.allclose(transitions.sum(axis=1), 1), "Rows must sum to 1."
        return transitions

    def simulate_prices(self):
        print("Simulating price series...")
        """Simulate price series using the transition matrix"""
        np.random.seed(1)
        current_idx = len(self.price_grid) // 2  # or use a random start if preferred
        self.price_series[0] = self.price_avg

        for t in range(1, self.num_sim_periods):
            next_idx = np.random.choice(
                np.arange(self.num_price_levels),
                p=self.price_transitions[current_idx]
            )
            self.price_series[t] = self.price_grid[next_idx]
            current_idx = next_idx

        return self.price_series

