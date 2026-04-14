# free-gpu

`free-gpu` is a GPU compute planner on top of [`llmfit`](https://www.llmfit.org/).

It helps you answer a simple question: what should stay local, and when should you move to free-tier, cheap-credit, or grant-style providers?

PyPI: <https://pypi.org/project/free-gpu/>

## What it does

- Uses `llmfit` as the local-fit layer.
- Maps a workload to a practical provider lane.
- Ranks providers for free, under-25, or grant-style paths.
- Builds stage-aware plans for tasks such as inference, LoRA fine-tuning, and heavier remote jobs.
- Exposes the planner through a CLI, a TUI, a local MCP server, and a hosted HTTP MCP endpoint.

Canonical workloads:

- `scratch-train`
- `finetune-lora`
- `inference`
- `batch-eval`
- `agent-loop`

## Install

```bash
pip install free-gpu
```

## Use it

### TUI

```bash
free-gpu ui
```

### CLI

```bash
free-gpu providers --workload inference --budget free
free-gpu plan --workload finetune-lora --model llama-3.1-8b --budget under-25 --task-hours 6 --min-vram-gb 16
free-gpu plan --workload scratch-train --budget grant --task-hours 24 --min-vram-gb 40
```

### Hosted MCP

```text
https://free-gpu.vercel.app/mcp
```

### Local server

```bash
free-gpu-mcp
```

## Add free-gpu to your MCP client

### Codex

Hosted:

```bash
codex mcp add freeGpu --url https://free-gpu.vercel.app/mcp
```

Local:

```bash
codex mcp add free-gpu-local -- free-gpu-mcp
```

### Claude Code

Hosted:

```bash
claude mcp add --transport http free-gpu https://free-gpu.vercel.app/mcp
```

Local:

```bash
claude mcp add --transport stdio free-gpu -- free-gpu-mcp
```

### Cursor

Hosted:

```json
{
  "mcpServers": {
    "free-gpu": {
      "url": "https://free-gpu.vercel.app/mcp"
    }
  }
}
```

Local:

```json
{
  "mcpServers": {
    "free-gpu": {
      "command": "free-gpu-mcp"
    }
  }
}
```

### VS Code

Hosted:

```json
{
  "servers": {
    "freeGpu": {
      "type": "http",
      "url": "https://free-gpu.vercel.app/mcp"
    }
  }
}
```

Local:

```json
{
  "servers": {
    "freeGpu": {
      "type": "stdio",
      "command": "free-gpu-mcp"
    }
  }
}
```

## Dataset

The provider ledger lives in [`free_gpu/gpu_compute_database.csv`](./free_gpu/gpu_compute_database.csv).

## Project links

- Repository: <https://github.com/francescoopiccolo/free-gpu>
- PyPI: <https://pypi.org/project/free-gpu/>
- Hosted MCP: <https://free-gpu.vercel.app/mcp>
