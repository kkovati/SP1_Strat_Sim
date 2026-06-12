import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
SIM_LENGTH = 1000  # Number of timesteps to simulate


class Company:
    """Represents a single company with a stochastic price evolution."""
    company_counter = 0

    def __init__(self, initial_price: float, shares_outstanding: int, mu: float, sigma: float):
        self.company_id = Company.company_counter
        Company.company_counter += 1

        self.initial_price = initial_price
        self.shares_outstanding = shares_outstanding

        self.timestep = -1
        self.price_history = self.generate_random_timeseries(SIM_LENGTH, start_price=initial_price, mu=mu, sigma=sigma)
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


class Trader:
    """Implements the max-cap allocation strategy."""

    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.shares_owned: dict[int, float] = {}
        self.net_worth = initial_cash

        self.target_company = None
        self.net_worth_history = []

    @staticmethod
    def select_target(companies: list[Company]) -> Company:
        """Return the company with the highest market cap."""
        return max(companies, key=lambda c: c.get_market_cap())

    def rebalance(self, companies: list[Company]) -> None:
        """Liquidate current holdings and invest all cash into the target company."""
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


def plot_results(companies: list[Company], trader: Trader) -> None:
    """Display a 2-row, 1-column chart: company prices on top, trader net worth on bottom."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Upper chart: company market cap histories ---
    for c in companies:
        ax_top.plot(c.market_cap_history[:c.timestep + 1], label=f"Company {c.company_id}")
    ax_top.set_title("Company Market Capitalizations")
    ax_top.set_ylabel("Market Cap")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # --- Lower chart: trader net worth history ---
    ax_bot.plot(trader.net_worth_history, color="tab:green", label="Trader Net Worth")
    ax_bot.set_title("Trader Net Worth")
    ax_bot.set_xlabel("Timestep")
    ax_bot.set_ylabel("Net Worth")
    ax_bot.legend()
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    seed = 42

    companies = [
        Company(initial_price=100.0, shares_outstanding=1_000_000, mu=.0001, sigma=.01),
        Company(initial_price=200.0, shares_outstanding=400_000, mu=.0001, sigma=.01),
    ]

    trader = Trader(initial_cash=100_000.0)
    sim = SimulationManager(companies, trader)

    print(f"Step {sim.timestep}: Net worth = {trader.net_worth:.2f}")
    sim.run(SIM_LENGTH)
    print(f"Step {sim.timestep}: Net worth = {trader.net_worth:.2f}")

    for c in companies:
        print(f"Company {c.company_id}: price={c.get_current_price():.2f}, cap={c.get_market_cap():.0f}")

    plot_results(companies, trader)


if __name__ == "__main__":
    main()
