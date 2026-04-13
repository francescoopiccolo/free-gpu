from __future__ import annotations

import json
import re
from typing import Any

from .data import load_providers
from .llmfit_adapter import load_local_profile
from .models import WorkloadRequest
from .planner import assess_compute_need, build_plan, rank_providers

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
    FastMCP = None
    _MCP_IMPORT_ERROR = exc
else:
    _MCP_IMPORT_ERROR = None


def _new_mcp(*, host: str = "127.0.0.1") -> FastMCP:
    if FastMCP is None:  # pragma: no cover - depends on optional extra
        missing = _MCP_IMPORT_ERROR.name if _MCP_IMPORT_ERROR else "mcp"
        raise ModuleNotFoundError(
            f"The MCP server needs the optional dependency '{missing}'. "
            "Install the published package with `pip install free-gpu` or install locally with `pip install .`."
        )

    return FastMCP("free-gpu", json_response=True, stateless_http=True, host=host)


def _normalize_budget(budget: str) -> str:
    value = budget.strip().lower()
    aliases = {
        "any": "any",
        "free": "free",
        "under-25": "under-25",
        "<25": "under-25",
        "grant": "grant",
    }
    return aliases.get(value, "under-25")


def _normalize_workload(workload: str) -> str:
    value = workload.strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    aliases = {
        "scratch-train": "scratch-train",
        "scratch-training": "scratch-train",
        "scratch-training-run": "scratch-train",
        "train": "scratch-train",
        "training": "scratch-train",
        "pretrain": "scratch-train",
        "pre-training": "scratch-train",
        "finetune-lora": "finetune-lora",
        "fine-tune-lora": "finetune-lora",
        "finetune": "finetune-lora",
        "fine-tune": "finetune-lora",
        "lora-finetune": "finetune-lora",
        "lora-fine-tune": "finetune-lora",
        "lora-training": "finetune-lora",
        "inference": "inference",
        "infer": "inference",
        "generation": "inference",
        "serving": "inference",
        "serve": "inference",
        "batch-eval": "batch-eval",
        "batch-evaluation": "batch-eval",
        "batch-evaluate": "batch-eval",
        "evaluation-batch": "batch-eval",
        "eval": "batch-eval",
        "evaluation": "batch-eval",
        "agent-loop": "agent-loop",
        "agent": "agent-loop",
        "agent-run": "agent-loop",
        "agent-workflow": "agent-loop",
    }
    if compact in aliases:
        return aliases[compact]

    tokens = set(compact.split("-"))
    if "lora" in tokens or "finetune" in tokens or ("fine" in tokens and "tune" in tokens):
        return "finetune-lora"
    if "agent" in tokens:
        return "agent-loop"
    if "batch" in tokens and ("eval" in tokens or "evaluation" in tokens):
        return "batch-eval"
    if "infer" in tokens or "inference" in tokens:
        return "inference"
    if "train" in tokens or "training" in tokens:
        return "scratch-train"
    return compact or "inference"


def _build_request(
    *,
    workload: str,
    model: str | None,
    params_b: float | None,
    budget: str,
    task_hours: float,
    min_vram_gb: float | None,
    parallel_jobs: int,
    requires_api: bool,
    prefer_local: bool,
    deadline: str,
    limit: int,
) -> WorkloadRequest:
    return WorkloadRequest(
        workload=_normalize_workload(workload),
        model=model,
        params_b=params_b,
        budget=_normalize_budget(budget),
        task_hours=task_hours,
        min_vram_gb=min_vram_gb,
        parallel_jobs=max(parallel_jobs, 1),
        requires_api=requires_api,
        prefer_local=prefer_local,
        deadline=deadline,
        limit=limit,
    )


def _provider_snapshot() -> dict[str, Any]:
    providers = load_providers()
    budgets = {"any": 0, "free": 0, "under-25": 0, "grant": 0}
    for provider in providers:
        text = provider.free_tier.lower()
        grant_like = any(token in f"{provider.free_tier} {provider.category} {provider.notes}".lower() for token in ("grant", "program", "application", "allocation"))
        budgets["any"] += 1
        if "free" in text and not grant_like:
            budgets["free"] += 1
        if any(token in text for token in ("trial", "credit", "starter", "lite", "promotional")) and not grant_like:
            budgets["under-25"] += 1
        if grant_like:
            budgets["grant"] += 1
    return {
        "provider_count": len(providers),
        "budget_buckets": budgets,
        "sample_services": [provider.service for provider in providers[:10]],
    }


def _register_handlers(mcp: FastMCP) -> FastMCP:
    @mcp.tool()
    def plan_provider_workflow(
        workload: str,
        model: str | None = None,
        params_b: float | None = None,
        budget: str = "under-25",
        task_hours: float = 1.0,
        min_vram_gb: float | None = None,
        parallel_jobs: int = 1,
        requires_api: bool = False,
        prefer_local: bool = True,
        deadline: str = "flexible",
        limit: int = 5,
        ram_gb: float | None = None,
        vram_gb: float | None = None,
        gpu_name: str | None = None,
        llmfit_limit: int = 5,
    ) -> dict[str, Any]:
        """Build a compute-aware provider workflow for a task.

        Canonical workloads are scratch-train, finetune-lora, inference,
        batch-eval, and agent-loop. Common natural-language aliases are
        normalized automatically.
        """
        request = _build_request(
            workload=workload,
            model=model,
            params_b=params_b,
            budget=budget,
            task_hours=task_hours,
            min_vram_gb=min_vram_gb,
            parallel_jobs=parallel_jobs,
            requires_api=requires_api,
            prefer_local=prefer_local,
            deadline=deadline,
            limit=limit,
        )
        profile = load_local_profile(
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            gpu_name=gpu_name,
            llmfit_limit=llmfit_limit,
        )
        plan = build_plan(request, profile, load_providers())
        return plan.to_dict()

    @mcp.tool()
    def rank_providers_for_task(
        workload: str,
        model: str | None = None,
        params_b: float | None = None,
        budget: str = "under-25",
        task_hours: float = 1.0,
        min_vram_gb: float | None = None,
        parallel_jobs: int = 1,
        requires_api: bool = False,
        prefer_local: bool = True,
        deadline: str = "flexible",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top providers for a compute profile without building the full workflow.

        Canonical workloads are scratch-train, finetune-lora, inference,
        batch-eval, and agent-loop. Common natural-language aliases are
        normalized automatically.
        """
        request = _build_request(
            workload=workload,
            model=model,
            params_b=params_b,
            budget=budget,
            task_hours=task_hours,
            min_vram_gb=min_vram_gb,
            parallel_jobs=parallel_jobs,
            requires_api=requires_api,
            prefer_local=prefer_local,
            deadline=deadline,
            limit=limit,
        )
        ranked = rank_providers(load_providers(), request, stage=None, limit=limit)
        return [provider.to_dict() for provider in ranked]

    @mcp.tool()
    def assess_task_compute(
        workload: str,
        model: str | None = None,
        params_b: float | None = None,
        task_hours: float = 1.0,
        min_vram_gb: float | None = None,
        parallel_jobs: int = 1,
        requires_api: bool = False,
    ) -> dict[str, Any]:
        """Estimate the compute lane free-gpu will use for a task.

        Canonical workloads are scratch-train, finetune-lora, inference,
        batch-eval, and agent-loop. Common natural-language aliases are
        normalized automatically.
        """
        request = _build_request(
            workload=workload,
            model=model,
            params_b=params_b,
            budget="under-25",
            task_hours=task_hours,
            min_vram_gb=min_vram_gb,
            parallel_jobs=parallel_jobs,
            requires_api=requires_api,
            prefer_local=True,
            deadline="flexible",
            limit=5,
        )
        return assess_compute_need(request).to_dict()

    @mcp.resource("providers://snapshot")
    def providers_snapshot() -> str:
        """Summarize the provider dataset and budget buckets."""
        return json.dumps(_provider_snapshot(), indent=2)

    return mcp


def create_mcp(*, host: str = "127.0.0.1") -> FastMCP:
    return _register_handlers(_new_mcp(host=host))


if FastMCP is not None:  # pragma: no branch - small import gate
    mcp = create_mcp()


def main() -> None:
    if FastMCP is None:  # pragma: no cover - depends on optional extra
        missing = _MCP_IMPORT_ERROR.name if _MCP_IMPORT_ERROR else "mcp"
        raise SystemExit(
            f"The MCP server needs the optional dependency '{missing}'. "
            "Install the published package with `pip install free-gpu`, then run `free-gpu-mcp`."
        )
    mcp.run()
