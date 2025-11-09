# hh_simulation

Terminal-only simulation scaffolding for worker–intermediary–firm matching with configurable incentives.

## Quickstart

1. Create a virtual environment and install dependencies:

```bash
cd "/Users/anandkrishnan/Desktop/Stanford/ECON 285/hh_simulation"
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

2. Run a quick smoke test (prints a few matches):

```bash
./.venv/bin/python -m hh_simulation --steps 1 --seed 42
```

You can also run via the convenience script:

```bash
./.venv/bin/python simulation.py --steps 1 --seed 42
```

## What’s included

- `hh_simulation/agents.py`: basic `Worker`, `Firm`, `Intermediary` data models
- `hh_simulation/market.py`: simple market loop with matching and wage adjustments
- `hh_simulation/__main__.py`: Typer CLI (`python -m hh_simulation run ...`)
- `simulation.py`: top-level shortcut to the same CLI

## Next steps

- We can extend utilities, matching rules, and intermediary objectives as needed.
- Add configuration (YAML/JSON) and experiment logging to support your paper.
