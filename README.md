# LASR_setup

Base template for LASR AI safety research projects.

## Directory Structure

- `src/` - Proven, reusable code
- `scripts/` - Experimental scripts (gitignored)
- `scratch/` - Experiment outputs (use `tee` to capture output)
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

### 4. Hugging Face Authentication

```bash
uv run huggingface-cli login
```

---

# Workshop: Steering Vector Extraction

## 1. Tmux Setup

TMUX is great. If you run experiments in the default terminal and ssh dies, your experiments die too! Tmux saves you here. **ALWAYS RUN EXPERIMENTS IN TMUX**

Here's a little tmux crash course:

```bash
tmux new -s training
# Ctrl+b d to detach

tmux new -s eval
# Ctrl+b d to detach
```

Practice switching:

```bash
tmux attach -t training
# Ctrl+b d

tmux attach -t eval
# Ctrl+b d

# Switch to the other session with
# Ctrl+b )

tmux ls  # list sessions
```

In the `training` session, split horizontally:

```bash
tmux attach -t training
# Ctrl+b %  (splits horizontally)
# Ctrl+b ↑/↓ to switch panes
# Ctrl+b [space] to change layouts
```

Kill the tmux pane:

```bash
# Ctrl+b x -> y
```

## 2. Code Organization

**`src/`** contains code you will be using going forward. This is committed to git and is clean and nice and has tests.

**`scripts/`** is your claude code workshop. Experimental scripts live here while you iterate. This is gitignored.

**Workflow:**
1. Write experimental code in `scripts/`
2. Use utilities from `src/`
3. Once a script is proven useful, migrate it to `src/` and add a `cli.py` file which allows you to verifiably use it

**Big Workflow tip:** Do frequent cleanup! Claude Code is a slop machine. If you don't clean up, you'll lose track of what's what within a week.

**Even Bigger Workflow tip:** THE WORST THING THAT CAN HAPPEN IS THAT YOU MISCHARACTERIZE YOUR RESULTS. Don't trust claude code to make changes to src/ without triple-checking it! Your repo will turn to absolute guck and you might lie when reporting results. Everything in src/ should be good, clean code which you have looked at yourself and tested yourself.

**Being-a-successful-researcher tip:** The most important skill as a technical AI safety researcher in the age of claude code is to know (a) what questions are most important to answer, and (b) how you would approach answering them. Whenever possible, you should be multiple experiments in parallel. `CUDA_VISIBLE_DEVICES=0` and all that.

## 3. Extract Steering Vectors

In `training`, run extraction at layer 12:

```bash
uv run python -m lasr_setup.steering.cli --layer 12 --num-vectors 2 2>&1 | tee scratch/extraction_layer12.txt
```

This saves vectors to `scratch/steering_vectors/`.

## 4. Test that it worked

What things do we need to test to ensure that this *did what we expect it to do*?

## 5. Write an Evaluation Script

Your task: write `scripts/evaluate_steering.py` that:

1. Loads steering vectors from a directory
2. Runs them against test datasets in `datasets/test/`
3. Computes dot product of activations with the **normalized** probe
4. Outputs results to `scratch/<descriptive_name>/`:
   - A CSV with activation scores
   - A matplotlib bar plot

Use descriptive names like `scratch/caps_probe_layer12_2024jan/` so you can find results later.

## 6. Run Evaluation

In `eval` session:

```bash
uv run python scripts/evaluate_steering.py --vectors-dir scratch/steering_vectors --layer 12
```

## Runpod Tips

- Use runpod network volumes to maintain data between runpod sessions
- Have a `runpod_bootup` to auto-run whenever you log into a new pod
- You can schedule runpod stops. So if you know your experiment will take ~8 hours but it'll be 2 am when it finishes, schedule the pod to shut down after it finishes.