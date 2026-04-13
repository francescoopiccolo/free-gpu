# free-gpu

`free-gpu` is a terminal-first planner for free and near-free compute.

It is designed to sit on top of [`llmfit`](https://www.llmfit.org/):

- `llmfit` answers: what models fit my local hardware?
- `free-gpu` answers: given this workload and compute need, which providers should I use, and how should I split work across local plus remote stages?

The point is not to clone `llmfit`. The point is to use `llmfit` as the local-fit engine, then add provider filtering, role-aware ranking, and workflow planning around free, cheap, and grant-style compute.

## What users actually get

`free-gpu` helps answer questions like:

- "I need quick inference for a small coding task. Which free-tier provider is the least painful?"
- "I need a few hours of GPU time for LoRA fine-tuning. Should I look at credits, trials, or a cloud free tier?"
- "This task is too heavy for casual free tiers. Which grant or program lane should I think about instead?"
- "What should stay local, and what should move to remote compute?"

## What the repo includes

- The original provider dataset in [`free_gpu/gpu_compute_database.csv`](./free_gpu/gpu_compute_database.csv)
- A Python CLI for provider ranking and workflow planning
- A Textual TUI focused on provider selection rather than local model browsing
- A small MCP server so external agents can ask for provider plans programmatically
- A GitHub Pages-ready project page in [`docs/index.html`](./docs/index.html)

## Core product rules

- Role is a ranking lens, not a hard exclusion filter.
- Budget buckets are semantic UX buckets, not literal accounting truth.
- Grant-like providers behave like card-required options.
- The planner should surface the right provider lane for the task instead of treating every task as the same generic ranking problem.

## Provider lanes

`free-gpu` is not only about "free" in the narrow sense. It plans across several practical lanes:

- `free tier`: browser notebooks, starter quotas, short session access
- `under-25`: credits, trials, starter plans, light paid-but-cheap access
- `grant`: startup programs, research allocations, application-based access, and other heavier support paths

That matters because different tasks naturally fall into different lanes:

- quick demos and lightweight inference often fit the free-tier lane
- medium notebook work and moderate fine-tunes often fit the under-25 or credit lane
- heavier training and long-running work often belong in the grant lane

## Workflow logic

The planner estimates a compute lane from:

- workload
- model size
- target VRAM
- estimated task hours
- parallel jobs
- API needs

Then it schedules providers accordingly:

- `burst`: short runs, quick inference, fast-start options
- `session`: notebook or credit-backed work that lasts longer
- `heavy`: bigger VRAM or sustained remote compute
- `grant-scale`: tasks that look more like allocations, programs, or heavy research/startup support

Each workflow step carries its own compute summary, so a multi-stage plan can recommend different provider types for prep, fine-tune, eval, and serving.

## How Pages and MCP fit together

The project has two different surfaces:

- GitHub Pages hosts the public project site and docs
- the MCP server runs locally on the user's own machine

GitHub Pages cannot run the planner logic or host the Python MCP server. It is only the public website.

The actual MCP workflow is:

1. a user installs `free-gpu`
2. the user runs `free-gpu-mcp` locally
3. their coding agent connects to that local MCP server
4. the agent can call planner tools such as `plan_provider_workflow`

That means:

- no hosting cost for you
- no central backend to maintain
- users keep control because the tool runs locally
- any MCP-capable coding agent can use it if it supports local MCP servers

## Install

```bash
python -m pip install -e .
```

To use the MCP server too:

```bash
python -m pip install -e ".[mcp]"
```

## CLI

### Local profile

```bash
free-gpu local
free-gpu local --ram-gb 32 --vram-gb 12
```

### Provider ranking

```bash
free-gpu providers --workload inference --budget free
free-gpu providers --workload agent-loop --budget under-25 --task-hours 3 --parallel-jobs 4 --requires-api
```

### Workflow planning

```bash
free-gpu plan --workload inference --model qwen2.5-coder-7b --ram-gb 32 --vram-gb 8
free-gpu plan --workload finetune-lora --model llama-3.1-8b --budget under-25 --task-hours 6 --min-vram-gb 16
free-gpu plan --workload scratch-train --budget grant --task-hours 24 --min-vram-gb 40
```

Useful planning flags:

- `--task-hours`
- `--min-vram-gb`
- `--parallel-jobs`
- `--requires-api`
- `--budget any|free|under-25|grant`

Every command also accepts `--json`.

## Terminal UI

Run:

```bash
free-gpu ui
```

The TUI is inspired by `llmfit`'s visual grammar, but it stays focused on provider planning:

- a top system bar with local hardware context from `llmfit`
- broad provider browsing by default
- role, workload, budget, and payment filters
- a central provider table
- bottom panes for links, recommendation context, and workflow summary

Current budget options in the TUI:

- `Budget Any`
- `Free`
- `<25`
- `Grant`

## llmfit integration

If `llmfit` is installed, `free-gpu` will try to use:

- `llmfit system --json`
- `llmfit recommend -n N --json`

The adapter uses structured JSON output rather than scraping terminal text. If `llmfit` is missing or parsing fails, `free-gpu` continues in provider-first mode and reports what it could not infer.

## MCP server

Run:

```bash
free-gpu-mcp
```

The MCP server exposes tools for compute-aware planning, including:

- `plan_provider_workflow`
- `rank_providers_for_task`
- `assess_task_compute`

It also exposes a small dataset summary resource:

- `providers://snapshot`

### What the MCP is for

The MCP lets an agent ask questions such as:

- "Plan a cheap inference workflow for this task"
- "Rank providers for a 6-hour fine-tune that needs about 16 GB VRAM"
- "Does this task look like free-tier, credit-tier, or grant-scale work?"

### Generic local MCP setup

If your coding agent supports local MCP servers over stdio, the setup is conceptually:

```json
{
  "mcpServers": {
    "free-gpu": {
      "command": "free-gpu-mcp"
    }
  }
}
```

The exact config file depends on the agent, but the idea is the same: point the client at the local `free-gpu-mcp` command.

Example MCP-style request shape:

```json
{
  "tool": "plan_provider_workflow",
  "arguments": {
    "workload": "agent-loop",
    "budget": "under-25",
    "task_hours": 3,
    "parallel_jobs": 4,
    "requires_api": true
  }
}
```

Example agent flow:

1. You ask your coding agent: "I need to fine-tune an 8B model for about 6 hours and want to stay near free."
2. The agent calls `plan_provider_workflow`.
3. `free-gpu` estimates the compute lane.
4. The agent gets back a structured plan with:
   - local vs remote recommendation
   - stage-by-stage workflow
   - top providers for that compute need
   - whether the task fits free tier, cheap credits, or a grant-style path

## GitHub Pages

A project page is included in [`docs/index.html`](./docs/index.html).

On GitHub, enable Pages and point it at:

- Branch: `main`
- Folder: `/docs`

## Tests

Run:

```bash
python -m unittest tests.test_planner -v
```
