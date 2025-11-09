# hh_simulation

Two-period market simulation for studying matching intermediaries (clearinghouses/headhunters) with their own incentives in worker–firm matching.

## Model Overview

The simulation implements a discrete two-period framework:

- **Period 0 (Early Phase)**: Firms observe noisy signals of worker quality (`q' = q + ε`, where `ε ~ N(0, σ²)`)
- **Period 1 (Regular Phase)**: Firms observe true worker quality perfectly

### Agents

- **Workers**: Have true quality `q_j` (ranked `q₁ > q₂ > ... > qₙ`) and baseline utility `ū_j`
  - Utility: `U_w(f_i, t) = v(i) + δ_w · 1{t=0}`
  - Accept offers only if `U_w(f_i, t) ≥ ū_j`

- **Firms**: Ranked by prestige (`f₁ ≻ f₂ ≻ ... ≻ fₘ`)
  - Utility at t=0: `U_f(w_j, 0) = γ(q'_j) + δ_f`
  - Utility at t=1: `U_f(w_j, 1) = γ(q_j)`

- **Headhunters**: Represent subsets of firms and workers
  - Utility: `U_h(f_i, w_j, t) = P(offer accepted) · (β + η_{i,j}) + δ_h · 1{t=0}`
  - Facilitate matches between accessible workers and firms

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
cd "/Users/anandkrishnan/Desktop/Stanford/ECON 285/hh_simulation"
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

2. Run a simulation:

```bash
./.venv/bin/python -m hh_simulation --num-workers 50 --num-firms 10 --num-headhunters 3 --seed 42
```

You can also run via the convenience script:

```bash
./.venv/bin/python simulation.py --num-workers 50 --num-firms 10 --seed 42
```

## Command-Line Options

- `--num-workers`: Number of workers (default: 50)
- `--num-firms`: Number of firms (default: 10)
- `--num-headhunters`: Number of headhunters (default: 3)
- `--delta-w`: Worker early signing bonus δ_w (default: 0.1)
- `--delta-f`: Firm benefit from early hiring δ_f (default: 0.1)
- `--delta-h`: Headhunter benefit from early placement δ_h (default: 0.05)
- `--signal-noise-std`: Standard deviation of signal noise at t=0, σ (default: 0.2)
- `--seed`: Random seed for reproducibility (optional)

## Structure

- `hh_simulation/agents.py`: `Worker`, `Firm`, and `Headhunter` classes with utility functions
- `hh_simulation/market.py`: Two-period market simulation with matching logic
- `hh_simulation/__main__.py`: Typer CLI for running simulations
- `simulation.py`: Top-level shortcut to the CLI

## Output

The simulation displays:
- Matches in each period (early phase and regular phase)
- Worker and firm utilities for each match
- Headhunter utilities
- Observed quality (noisy at t=0, true at t=1)
- Summary statistics

## Next Steps

- Extend matching algorithms (e.g., deferred acceptance, stable matching)
- Add configuration files (YAML/JSON) for parameter sweeps
- Implement experiment logging and data export
- Add visualization and analysis tools
- Model more complex headhunter incentives and reputation dynamics
