# free-gpu

`free-gpu` is a CLI planner for open-model workflows.

It is designed to sit on top of [`llmfit`](https://www.llmfit.org/): `llmfit`
figures out what can run locally on your machine, while `free-gpu` uses the
repo's provider dataset to recommend the cheapest viable workflow across local
hardware and free or near-free cloud GPUs.

## What this MVP does

- Loads the existing provider CSV and scores providers by workload.
- Uses `llmfit` if it is installed to enrich local hardware detection.
- Falls back to manual `--ram-gb` and `--vram-gb` inputs if `llmfit` is not available.
- Produces staged workflow recommendations for:
  - `scratch-train`
  - `finetune-lora`
  - `inference`
  - `batch-eval`
  - `agent-loop`

## Quick start

```bash
python -m pip install -e .
free-gpu plan --workload inference --model qwen2.5-coder-7b --ram-gb 32 --vram-gb 8
free-gpu plan --workload finetune-lora --model llama-3.1-8b --budget free
free-gpu providers --workload agent-loop --limit 5
free-gpu local --ram-gb 32 --vram-gb 12
free-gpu ui
```

## llmfit integration

If `llmfit` is installed, `free-gpu` will try to:

- run `llmfit system`
- run `llmfit recommend --json --limit N`

The current adapter is intentionally defensive: if `llmfit` output changes or
is unavailable, the CLI continues in provider-only mode and tells you what it
could not infer.

## JSON output

Every command accepts `--json` for machine-readable output.

## Terminal UI

Run:

```bash
free-gpu ui
```

The TUI is inspired by `llmfit`, but focused on cloud and hybrid workflow
planning instead of local model ranking. It includes:

- a live local hardware banner populated from `llmfit`
- workload, budget, deadline, and model controls
- a provider ranking table
- a workflow pane that explains local vs remote steps
- a detail pane for the selected provider
