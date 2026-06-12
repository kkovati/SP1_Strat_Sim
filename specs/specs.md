
# Module: Trading Simulation (Max-Cap Strategy)

## Purpose
Simulate a simplified investment strategy where a trader always allocates capital to the currently most valuable company (by market capitalization).

## Non-goals
- No real market data integration
- No portfolio diversification
- No transaction costs, slippage, or latency
- No multi-asset or multi-strategy modeling

---

## Public API

### class Company
Represents a single company with a stochastic price evolution.

Attributes:
- initial_price: float
- shares_outstanding: int
- current_price: float
- price_history: list[float]

Methods:
- step() -> None  
  Advances the price using a stochastic process

- market_cap() -> float  
  Returns current_price * shares_outstanding

---

### class Trader
Implements the max-cap allocation strategy.

Attributes:
- cash: float
- shares_owned: dict[str, float]   # company_id -> shares
- net_worth: float

Methods:
- select_target(companies: list[Company]) -> Company  
  MUST return the company with highest market cap

- rebalance(companies: list[Company]) -> None  
  MUST:
  - liquidate current holdings
  - invest all cash into selected company

- update_net_worth(companies: list[Company]) -> None  
  MUST compute total value of holdings + cash

---

### class SimulationManager
Coordinates the simulation.

Attributes:
- companies: list[Company]
- trader: Trader
- timestep: int

Methods:
- step() -> None  
  MUST:
  - update all company prices
  - trigger trader rebalance
  - update trader net worth
  - increment timestep

- run(n_steps: int) -> None  
  Runs simulation for n_steps

---

## Behavior

### Company.price evolution
- MUST generate stochastic price path
- MUST ensure price > 0 at all times
- SHOULD use simple model (e.g., geometric random walk)

### Trader strategy
- MUST always fully allocate capital to single highest market cap company
- MUST not hold multiple companies simultaneously
- MUST rebalance at every timestep

### Simulation flow
At each timestep:
1. All companies update price
2. Market caps recalculated
3. Trader selects highest market cap company
4. Trader reallocates fully
5. Net worth updated

---

## Examples

### Setup
- Company A: price=100, shares=1e6 → cap=100M
- Company B: price=200, shares=4e5 → cap=80M

→ Trader invests in Company A

---

### After price change
- Company A: price=90 → cap=90M
- Company B: price=210 → cap=84M

→ Trader switches to Company B

---

## Performance Requirements
- O(N_companies) per timestep
- No unnecessary allocations in inner loops
- Deterministic mode supported via fixed RNG seed

---

## Implementation Constraints
- Prefer vectorized operations for price updates
- Avoid branching inside per-company loops where possible
- Keep state minimal and explicit

---

## Extensions (out of scope, future work)
- Transaction costs
- Partial allocation strategies
- Multi-factor ranking (not just market cap)
- Real market data input
