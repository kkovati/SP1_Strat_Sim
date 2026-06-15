import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

# CONSTANTS
SIM_LENGTH = 500  # Number of timesteps to simulate
N_COMPANIES = 10  # Number of companies in the simulation


class Company:
    """Represents a single company with a stochastic price evolution."""
    company_counter = 0

    def __init__(self, initial_market_cap: float, shares_outstanding: int, mu: float, sigma: float):
        self.company_id = Company.company_counter
        Company.company_counter += 1

        self.initial_market_cap = initial_market_cap
        self.shares_outstanding = shares_outstanding
        self.initial_price = initial_market_cap / shares_outstanding

        self.timestep = -1
        self.price_history = self.generate_random_timeseries(SIM_LENGTH, start_price=self.initial_price, mu=mu, sigma=sigma)
        assert np.all(self.price_history > 0), "Generated price history contains non-positive values"
        self.market_cap_history = self.price_history * shares_outstanding

    def step(self) -> None:
        """Advance price."""
        self.timestep += 1

    def get_current_price(self) -> float:
        """Return current price."""
        return self.price_history[self.timestep]

    def get_market_cap(self) -> float:
        """Calculate market capitalization."""
        return self.market_cap_history[self.timestep]

    @staticmethod
    def generate_random_timeseries(length, mu=.0001, sigma=.01, start_price=100.0):
        """Generate a random price series using geometric Brownian motion."""
        returns = np.random.normal(loc=mu, scale=sigma, size=length)
        timeseries = start_price * (1 + returns).cumprod()
        timeseries += (start_price - timeseries[0])
        return timeseries


class CompanyFactory:
    """Factory that creates Company instances with randomized parameters."""

    # Sensible default ranges
    MARKET_CAP_RANGE = (1_000_000, 500_000_000)
    SHARES_RANGE = (100_000, 10_000_000)
    MU_RANGE = (-0.001, 0.001)
    SIGMA_RANGE = (0.005, 0.02)

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def create(self) -> Company:
        """Create a single Company with random parameters."""
        initial_market_cap = self.rng.uniform(*self.MARKET_CAP_RANGE)
        shares_outstanding = int(self.rng.integers(*self.SHARES_RANGE))
        mu = self.rng.uniform(*self.MU_RANGE)
        sigma = self.rng.uniform(*self.SIGMA_RANGE)
        return Company(initial_market_cap=initial_market_cap, shares_outstanding=shares_outstanding, mu=mu, sigma=sigma)

    def create_batch(self, n: int) -> list[Company]:
        """Create n companies with random parameters."""
        return [self.create() for _ in range(n)]


class Trader:
    """Implements allocation strategies: max_cap, buy_and_hold, equal_weight."""

    STRATEGY_MAX_CAP = "max_cap"
    STRATEGY_BUY_AND_HOLD = "buy_and_hold"
    STRATEGY_EQUAL_WEIGHT = "equal_weight"

    def __init__(self, initial_cash: float, strategy: str = "max_cap"):
        self.cash = initial_cash
        self.shares_owned: dict[int, float] = {}
        self.net_worth = initial_cash
        self.strategy = strategy

        self.target_company = None
        self.net_worth_history = []
        self._initial_buy_done = False

    @staticmethod
    def select_target(companies: list[Company]) -> Company:
        """Return the company with the highest market cap."""
        return max(companies, key=lambda c: c.get_market_cap())

    def rebalance(self, companies: list[Company]) -> None:
        """Dispatch to the strategy-specific rebalance method."""
        if self.strategy == self.STRATEGY_MAX_CAP:
            self._rebalance_max_cap(companies)
        elif self.strategy == self.STRATEGY_BUY_AND_HOLD:
            self._rebalance_buy_and_hold(companies)
        elif self.strategy == self.STRATEGY_EQUAL_WEIGHT:
            self._rebalance_equal_weight(companies)

    def _rebalance_max_cap(self, companies: list[Company]) -> None:
        """Liquidate current holdings and invest all cash into the highest market cap company."""
        target_company = self.select_target(companies)
        if self.target_company == target_company:
            return  # No change in target, skip rebalancing
        self.target_company = target_company

        # Liquidate all holdings
        for cid, shares in self.shares_owned.items():
            company = next(c for c in companies if c.company_id == cid)
            self.cash += shares * company.get_current_price()
        self.shares_owned.clear()

        # Invest fully into highest market cap company
        shares_to_buy = self.cash / target_company.get_current_price()
        self.shares_owned[target_company.company_id] = shares_to_buy
        self.cash = 0.0

    def _rebalance_buy_and_hold(self, companies: list[Company]) -> None:
        """Buy the highest market cap stock once and hold forever."""
        if self._initial_buy_done:
            return
        target = self.select_target(companies)
        shares_to_buy = self.cash / target.get_current_price()
        self.shares_owned[target.company_id] = shares_to_buy
        self.cash = 0.0
        self._initial_buy_done = True

    def _rebalance_equal_weight(self, companies: list[Company]) -> None:
        """Buy equal cash amounts of all stocks once and hold forever."""
        if self._initial_buy_done:
            return
        cash_per_company = self.cash / len(companies)
        for c in companies:
            shares_to_buy = cash_per_company / c.get_current_price()
            self.shares_owned[c.company_id] = shares_to_buy
        self.cash = 0.0
        self._initial_buy_done = True

    def update_net_worth(self, companies: list[Company]) -> None:
        """Compute total value of holdings + cash."""
        holdings_value = 0.0
        for cid, shares in self.shares_owned.items():
            company = next(c for c in companies if c.company_id == cid)
            holdings_value += shares * company.get_current_price()
        self.net_worth = self.cash + holdings_value
        self.net_worth_history.append(self.net_worth)


class SimulationManager:
    """Coordinates the simulation."""

    def __init__(self, companies: list[Company], traders: list[Trader]):
        self.companies = companies
        self.traders = traders
        self.timestep = 0

    def step(self) -> None:
        """Execute one simulation timestep."""
        for company in self.companies:
            company.step()
        for trader in self.traders:
            trader.rebalance(self.companies)
            trader.update_net_worth(self.companies)
        self.timestep += 1

    def run(self, n_steps: int) -> None:
        """Run simulation for n_steps."""
        for _ in range(n_steps):
            self.step()


def plot_results(companies: list[Company], traders: list[Trader]) -> None:
    """Display a 2-row, 1-column chart: company market caps on top, all trader net worths on bottom."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Upper chart: company market cap histories ---
    for c in companies:
        ax_top.plot(c.market_cap_history[:c.timestep + 1], label=f"Company {c.company_id}")
    ax_top.set_title("Company Market Capitalizations")
    ax_top.set_ylabel("Market Cap")
    # ax_top.legend()
    ax_top.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=15))
    ax_top.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax_top.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax_top.grid(True, which="major", alpha=0.6)
    ax_top.grid(True, which="minor", alpha=0.15)

    # --- Lower chart: all trader net worth histories ---
    for trader in traders:
        ax_bot.plot(trader.net_worth_history, label=trader.strategy)
    ax_bot.set_title("Trader Net Worth Comparison")
    ax_bot.set_xlabel("Timestep")
    ax_bot.set_ylabel("Net Worth")
    ax_bot.legend()
    ax_bot.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=15))
    ax_bot.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax_bot.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax_bot.grid(True, which="major", alpha=0.6)
    ax_bot.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.show()


def main():
    seed = 42
    factory = CompanyFactory(seed=seed)
    companies = factory.create_batch(N_COMPANIES)

    traders = [
        Trader(initial_cash=100_000.0, strategy=Trader.STRATEGY_MAX_CAP),
        Trader(initial_cash=100_000.0, strategy=Trader.STRATEGY_BUY_AND_HOLD),
        Trader(initial_cash=100_000.0, strategy=Trader.STRATEGY_EQUAL_WEIGHT),
    ]

    sim = SimulationManager(companies, traders)
    sim.run(SIM_LENGTH)

    for trader in traders:
        print(f"[{trader.strategy}] Final net worth = {trader.net_worth:.2f}")

    for c in companies:
        print(f"Company {c.company_id}: price={c.get_current_price():.2f}, cap={c.get_market_cap():.0f}")

    plot_results(companies, traders)


if __name__ == "__main__":
    main()
