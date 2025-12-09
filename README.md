# hh_simulation

Two-period market simulation for exploring how headhunters/matching intermediaries influence worker-firm match quality and market unraveling. Based off the Roth & Xing (1994) model with added matching intermediaries with their own incentives. Use experiments.py file to run an experiment and generate graphs by varying gamma, alpha, number of headhunters in market. Agents.py contain utility formulations for each type (worker, firm, headhunter), market.py contains game and setup.

## Quickstart
```bash
cd "/Users/anandkrishnan/Desktop/Stanford/ECON 285/hh_simulation"
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

Run a sample simulation:
```bash
./.venv/bin/python -m hh_simulation --num-workers 50 --num-firms 10 --num-headhunters 3 --gamma 0.5 --alpha 0.5 --matching-algorithm hungarian --seed 42
```

## CLI options
- `--num-workers` (default 50)
- `--num-firms` (default 10)
- `--num-headhunters` (default 3)
- `--gamma` outside option scaling factor γ (default 0.5)
- `--alpha` headhunter utility weight α (default 0.5)
- `--matching-algorithm` `hungarian` (O(n^3), default) or `enumerative` (slow)
- `--seed`

## Where things live
- `hh_simulation/agents.py`: worker/firm/headhunter objects and utility helpers
- `hh_simulation/market.py`: two-period market loop and matching logic
- `hh_simulation/__main__.py`: Typer CLI entrypoint
- `simulation.py`: tiny wrapper around the CLI

