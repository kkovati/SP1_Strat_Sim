import numpy as np

# CONSTANTS
SIM_LENGTH = 10  # Number of timesteps to simulate


class Company:
    """Represents a single company with a stochastic price evolution."""
    company_counter = 0

    def __init__(self, initial_price: float, shares_outstanding: int, mu: float, sigma: float):
        self.company_id = Company.company_counter
        Company.company_counter += 1

        self.initial_price = initial_price
        self.shares_outstanding = shares_outstanding
        self.current_price = initial_price

        self.time = -1
        self.price_history = self.generate_random_timeseries(SIM_LENGTH, start_price=initial_price, mu=mu, sigma=sigma)
        assert np.all(self.price_history > 0), "Generated price history contains non-positive values"

    def step(self) -> None:
        """Advance price."""
        self.time += 1
        self.current_price = self.price_history[self.time]

    def market_cap(self) -> float:
        """Calculate market capitalization."""
        return self.current_price * self.shares_outstanding

    @staticmethod
    def generate_random_timeseries(length, mu=.0001, sigma=.01, start_price=100.0):
        """Generate a random price series using geometric Brownian motion."""
        returns = np.random.normal(loc=mu, scale=sigma, size=length)
        timeseries = start_price * (1 + returns).cumprod()
        timeseries += (start_price - timeseries[0])
        return timeseries


class Trader:
    """Implements the max-cap allocation strategy."""

    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.shares_owned: dict[str, float] = {}
        self.net_worth = initial_cash

    @staticmethod
    def select_target(companies: list[Company]) -> Company:
        """Return the company with the highest market cap."""
        return max(companies, key=lambda c: c.market_cap())

    def rebalance(self, companies: list[Company]) -> None:
        """Liquidate current holdings and invest all cash into the target company."""
        # Liquidate all holdings
        for cid, shares in self.shares_owned.items():
            company = next(c for c in companies if c.company_id == cid)
            self.cash += shares * company.current_price
        self.shares_owned.clear()

        # Invest fully into highest market cap company
        target = self.select_target(companies)
        shares_to_buy = self.cash / target.current_price
        self.shares_owned[target.company_id] = shares_to_buy
        self.cash = 0.0

    def update_net_worth(self, companies: list[Company]) -> None:
        """Compute total value of holdings + cash."""
        holdings_value = 0.0
        for cid, shares in self.shares_owned.items():
            company = next(c for c in companies if c.company_id == cid)
            holdings_value += shares * company.current_price
        self.net_worth = self.cash + holdings_value


class SimulationManager:
    """Coordinates the simulation."""

    def __init__(self, companies: list[Company], trader: Trader):
        self.companies = companies
        self.trader = trader
        self.timestep = 0

    def step(self) -> None:
        """Execute one simulation timestep."""
        for company in self.companies:
            company.step()
        self.trader.rebalance(self.companies)
        self.trader.update_net_worth(self.companies)
        self.timestep += 1

    def run(self, n_steps: int) -> None:
        """Run simulation for n_steps."""
        for _ in range(n_steps):
            self.step()


def main():
    seed = 42

    companies = [
        Company(initial_price=100.0, shares_outstanding=1_000_000, mu=.0001, sigma=.01),
        Company(initial_price=200.0, shares_outstanding=400_000, mu=.0001, sigma=.01),
    ]

    trader = Trader(initial_cash=100_000.0)
    sim = SimulationManager(companies, trader)

    print(f"Step {sim.timestep}: Net worth = {trader.net_worth:.2f}")
    sim.run(10)
    print(f"Step {sim.timestep}: Net worth = {trader.net_worth:.2f}")

    for c in companies:
        print(f"Company {c.company_id}: price={c.current_price:.2f}, cap={c.market_cap():.0f}")


if __name__ == "__main__":
    main()
