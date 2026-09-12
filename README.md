# Backtesting a Systematic Trading Strategy

An **event-driven** (object-oriented) backtesting engine to evaluate systematic
trading strategies rigorously and reproducibly — from raw data to statistical
evaluation — with a focus on the pitfalls that make a backtest look better than it
really is: look-ahead bias, transaction costs, multiple testing and overfitting.

## Headline result

On the sample studied, **cross-sectional momentum** shows the best risk-adjusted
profile (Sharpe ≈ 1.32) — but with no claim of cross-regime robustness, since the
period contains no major crisis. The **ADI/MPWR** pair does not survive Sharpe
deflation (DSR ≈ 0.10; out-of-sample Sharpe ≈ 0.34; P[failing a Sharpe-1 target]
≈ 61%): it is therefore presented as a **diversification sleeve** (correlation to
SPY ≈ 0), not a standalone alpha source. The project emphasises the robustness of
the methodology over raw performance.

## Architecture

The engine separates four responsibilities, connected through an event queue. A
strategy knows nothing about the portfolio or execution, so it is fully
interchangeable (an ML alpha signal would plug in identically).

| Component          | Role                                                                            |
| ------------------ | ------------------------------------------------------------------------------- |
| `DataHandler`      | Replays history bar by bar → emits a `MarketEvent`                              |
| `Strategy`         | Applies the trading rule → emits a `SignalEvent`                                |
| `Portfolio`        | Sizes positions, marks to market, tracks cash/positions → emits an `OrderEvent` |
| `ExecutionHandler` | Simulates the market (commission + slippage) → emits a `FillEvent`              |
| `Backtest`         | Orchestrator: advances time and routes events                                   |

Cycle: `MarketEvent → SignalEvent → OrderEvent → FillEvent`.

**Anti look-ahead**: a bar is marked to market *before* that same bar's fill updates
positions — the structural equivalent of the vectorized backtest's `shift(1)`. The
event-driven engine is validated against a vectorized implementation (daily-return
correlation ≈ 0.9999).

## Strategies

* **Trend following** — SMA 50 / 200 crossover, long/flat.
* **Momentum** — cross-sectional (top-k over a 3-month lookback, monthly rebalancing).
* **Pairs trading** — core of the project: cointegration search (Engle–Granger, then Johansen to remove the OLS asymmetry), mean-reversion characterisation (Hurst exponent, variance ratio), spread modelled as an Ornstein–Uhlenbeck process (half-life), z-score / Bollinger-band rule with stop-loss.

## Evaluation rigour

* Transaction costs (commission + slippage in bps) tied to **turnover**.
* Position sizing: fixed fraction vs **volatility targeting**.
* Multiple-testing correction (**Benjamini–Hochberg** over 1006 candidate pairs).
* **Walk-forward** validation (3-year calibration / 1-year test).
* **Probabilistic** and **Deflated Sharpe Ratio** (deflated over the true number of screening trials).
* Strategy risk: concentration HHI, time under water, probability of failure (López de Prado ch. 15).

## Repository structure

```text
src/                     Engine modules (reusable)
  event.py               The 4 event types
  data.py                DataHandler (bar-by-bar replay)
  strategy.py            SMA, momentum, pairs
  portfolio.py           Sizing, mark-to-market, cash/positions
  execution.py           Market simulator (costs)
  backtest.py            Event loop
  screening.py           Pair screening + Deflated Sharpe (record-every-backtest)
  backtest_stats.py      Backtest statistics (ch.14) and strategy risk (ch.15)
notebooks/               Analysis notebooks (data pipeline → strategies → evaluation)
data/                    Frozen datasets (adjusted prices, universes, ranked pairs)
requirements.txt
```

## Setup & running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notebooks are meant to be read in prefix order (data pipeline → engine → strategies →
pair screening → final comparison). Market data is pulled from Yahoo Finance via
`yfinance`.

## Methodological references

* E. P. Chan — *Algorithmic Trading: Winning Strategies and Their Rationale*
* Y. Hilpisch (2020) — *Python for Algorithmic Trading*
* M. López de Prado (2018) — *Advances in Financial Machine Learning*, Wiley

---

*Built as part of a quantitative project portfolio. Results come from an experimental
study on a restricted universe and a limited period; they are not a general proof of
a strategy's effectiveness.*
