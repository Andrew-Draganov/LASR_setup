# LASR_setup

Base template for LASR AI safety research projects.

## Directory Structure

- `src/` - Source code
- `scripts/` - Executable scripts
- `scratch/` - Experiment output files (use `tee` to capture tmux output here)
- `datasets/` - Data files

## Setup

### 1. System Dependencies

```bash
apt-get update
apt-get install -y tmux
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# You might then need to run `source ~/.local/bin/env` or otherwise restart your shell
```

### 3. Initialize Python Environment

```bash
uv sync
```

This installs dependencies from `pyproject.toml` including:
- torch
- transformers
- matplotlib
- numpy

### 4. Hugging Face Authentication

```bash
uv run hf auth login --token YOUR_TOKEN_HERE
```

Enter your HF token when prompted. Required for gated models.

## Running Experiments

Always run experiments inside tmux and capture output to `scratch/`:

```bash
# Start a tmux session
tmux new -s experiment

# Test the cli.py file
python src/lasr_setup/cli.py --num-vectors 2 --layer 7 2>&1 | tee scratch/my_experiment_output.txt
```

This ensures:
- Experiments persist if your SSH connection drops
- All stdout/stderr is saved to a file for review

### Tmux Basics

- `tmux new -s <name>` - Create new session
- `tmux attach -t <name>` - Reattach to session
- `Ctrl+b d` - Detach from session
- `tmux ls` - List sessions
