from __future__ import annotations

import json
import re
from typing import Annotated, Any

from pydantic import Field

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


CANONICAL_WORKLOADS = (
    "scratch-train",
    "finetune-lora",
    "inference",
    "batch-eval",
    "agent-loop",
)
CANONICAL_BUDGETS = ("any", "free", "under-25", "grant")
CANONICAL_DEADLINES = ("flexible", "urgent")

WORKLOAD_ALIASES = {
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

BUDGET_ALIASES = {
    "any": "any",
    "free": "free",
    "under-25": "under-25",
    "<25": "under-25",
    "cheap": "under-25",
    "low-cost": "under-25",
    "grant": "grant",
    "credits": "grant",
}

DEADLINE_ALIASES = {
    "flexible": "flexible",
    "normal": "flexible",
    "eventual": "flexible",
    "urgent": "urgent",
    "asap": "urgent",
    "soon": "urgent",
    "today": "urgent",
}

WorkloadArg = Annotated[
    str,
    Field(
        description=(
            "Task type. Prefer one of: scratch-train, finetune-lora, inference, "
            "batch-eval, agent-loop. Common aliases such as 'training', "
            "'fine-tune', 'batch evaluation', and 'agent run' are normalized automatically."
        )
    ),
]
BudgetArg = Annotated[
    str,
    Field(
        description=(
            "Budget bucket. Prefer one of: any, free, under-25, grant. "
            "Aliases such as '<25', 'cheap', and 'credits' are normalized automatically."
        )
    ),
]
DeadlineArg = Annotated[
    str,
    Field(
        description=(
            "Deadline urgency. Prefer 'flexible' or 'urgent'. "
            "Aliases such as 'asap', 'soon', and 'today' are normalized automatically."
        )
    ),
]
ModelArg = Annotated[str | None, Field(description="Optional model name, e.g. llama-3.1-8b or qwen2.5-coder-7b.")]
ParamsArg = Annotated[float | None, Field(description="Optional model size in billions of parameters if already known.")]
TaskHoursArg = Annotated[float, Field(description="Estimated total task duration in hours. Non-positive values fall back to 1.0.")]
VRAMArg = Annotated[float | None, Field(description="Minimum VRAM requirement in GB if already known.")]
ParallelJobsArg = Annotated[int, Field(description="Number of parallel jobs to support. Values below 1 are clamped to 1.")]
LimitArg = Annotated[int, Field(description="Maximum number of providers to return. Values are clamped to the 1-10 range.")]
RamArg = Annotated[float | None, Field(description="Local system RAM in GB.")]
GpuVramArg = Annotated[float | None, Field(description="Local GPU VRAM in GB.")]
LlmfitLimitArg = Annotated[int, Field(description="Maximum number of local llmfit candidates to inspect.")]


def _new_mcp(*, host: str = "127.0.0.1") -> FastMCP:
    if FastMCP is None:  # pragma: no cover - depends on optional extra
        missing = _MCP_IMPORT_ERROR.name if _MCP_IMPORT_ERROR else "mcp"
        raise ModuleNotFoundError(
            f"The MCP server needs the optional dependency '{missing}'. "
            "Install the published package with `pip install free-gpu` or install locally with `pip install .`."
        )

    return FastMCP("free-gpu", json_response=True, stateless_http=True, host=host)


def _normalize_budget(budget: str) -> str:
    return BUDGET_ALIASES.get(budget.strip().lower(), "under-25")


def _normalize_deadline(deadline: str) -> str:
    return DEADLINE_ALIASES.get(deadline.strip().lower(), "flexible")


def _normalize_workload(workload: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "-", workload.strip().lower()).strip("-")
    if compact in WORKLOAD_ALIASES:
        return WORKLOAD_ALIASES[compact]

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


def _clamp_limit(limit: int) -> int:
    return min(max(limit, 1), 10)


def _clamp_parallel_jobs(parallel_jobs: int) -> int:
    return min(max(parallel_jobs, 1), 32)


def _coerce_positive_float(value: float | None, *, fallback: float) -> float:
    if value is None or value <= 0:
        return fallback
    return value


def _build_request_resolution(
    *,
    raw_workload: str,
    raw_budget: str,
    raw_deadline: str,
    request: WorkloadRequest,
    model: str | None,
    params_b: float | None,
    min_vram_gb: float | None,
) -> dict[str, Any]:
    assumptions: list[str] = []
    if params_b is None and model:
        assumptions.append("params_b will be inferred from the model name when possible.")
    if min_vram_gb is None:
        assumptions.append("min_vram_gb was not supplied, so the planner estimated VRAM from workload and model size.")

    return {
        "canonical_request": {
            "workload": request.workload,
            "budget": request.budget,
            "deadline": request.deadline,
            "task_hours": request.task_hours,
            "min_vram_gb": request.min_vram_gb,
            "parallel_jobs": request.parallel_jobs,
            "requires_api": request.requires_api,
            "prefer_local": request.prefer_local,
            "limit": request.limit,
            "model": request.model,
            "params_b": request.params_b,
        },
        "normalization": {
            "workload_changed": _normalize_workload(raw_workload) != raw_workload.strip().lower(),
            "budget_changed": _normalize_budget(raw_budget) != raw_budget.strip().lower(),
            "deadline_changed": _normalize_deadline(raw_deadline) != raw_deadline.strip().lower(),
            "accepted_workloads": list(CANONICAL_WORKLOADS),
            "accepted_budgets": list(CANONICAL_BUDGETS),
            "accepted_deadlines": list(CANONICAL_DEADLINES),
        },
        "assumptions": assumptions,
    }


def _client_guidance(tool_name: str, *, follow_up: str | None = None) -> dict[str, Any]:
    when_to_use = {
        "assess_task_compute": "Use this first only when you need a quick compute-lane estimate.",
        "rank_providers_for_task": "Use this when you already know the task shape and only need a shortlist.",
        "plan_provider_workflow": "Use this for the final answer when you want a full local-versus-remote workflow.",
    }
    return {
        "tool": tool_name,
        "when_to_use": when_to_use[tool_name],
        "follow_up_tool": follow_up,
        "response_hint": "Treat canonical_request as authoritative and present the recommendation directly. Do not mention internal alias normalization unless the user explicitly asks.",
    }


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
        task_hours=_coerce_positive_float(task_hours, fallback=1.0),
        min_vram_gb=min_vram_gb,
        parallel_jobs=_clamp_parallel_jobs(parallel_jobs),
        requires_api=requires_api,
        prefer_local=prefer_local,
        deadline=_normalize_deadline(deadline),
        limit=_clamp_limit(limit),
    )


def _provider_snapshot() -> dict[str, Any]:
    providers = load_providers()
    budgets = {"any": 0, "free": 0, "under-25": 0, "grant": 0}
    for provider in providers:
        text = provider.free_tier.lower()
        grant_like = any(
            token in f"{provider.free_tier} {provider.category} {provider.notes}".lower()
            for token in ("grant", "program", "application", "allocation")
        )
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
        "canonical_workloads": list(CANONICAL_WORKLOADS),
        "canonical_budgets": list(CANONICAL_BUDGETS),
        "canonical_deadlines": list(CANONICAL_DEADLINES),
        "sample_services": [provider.service for provider in providers[:10]],
    }


def _tool_selection_guide() -> dict[str, Any]:
    return {
        "recommended_order": [
            "If the task is underspecified, start with assess_task_compute.",
            "If the user wants a shortlist only, use rank_providers_for_task.",
            "If the user wants a concrete execution path, use plan_provider_workflow.",
        ],
        "canonical_workloads": {
            "scratch-train": "Train a model from scratch or pretrain.",
            "finetune-lora": "Fine-tune or LoRA-adapt an existing model.",
            "inference": "Run generation, serving, or one-off inference.",
            "batch-eval": "Run evaluation, benchmarking, or repeated scoring jobs.",
            "agent-loop": "Run tool-using agents or repeated autonomous loops.",
        },
        "budget_buckets": {
            "any": "No budget constraint.",
            "free": "Strictly free-tier access if possible.",
            "under-25": "Near-free credits, starter plans, or short trials.",
            "grant": "Application-based grants, research allocations, or longer-lived credits.",
        },
        "client_behavior": [
            "Prefer canonical workload names in tool calls.",
            "Use canonical_request from tool output as the authoritative interpretation.",
            "Avoid telling the end user about fallback reasoning unless they asked about assumptions.",
        ],
    }


def _register_handlers(mcp: FastMCP) -> FastMCP:
    @mcp.tool(
        title="Plan Provider Workflow",
        description="Build a full compute plan with local-fit analysis, staged workflow steps, and top provider recommendations.",
        structured_output=True,
    )
    def plan_provider_workflow(
        workload: WorkloadArg,
        model: ModelArg = None,
        params_b: ParamsArg = None,
        budget: BudgetArg = "under-25",
        task_hours: TaskHoursArg = 1.0,
        min_vram_gb: VRAMArg = None,
        parallel_jobs: ParallelJobsArg = 1,
        requires_api: bool = False,
        prefer_local: bool = True,
        deadline: DeadlineArg = "flexible",
        limit: LimitArg = 5,
        ram_gb: RamArg = None,
        vram_gb: GpuVramArg = None,
        gpu_name: str | None = None,
        llmfit_limit: LlmfitLimitArg = 5,
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
        payload = build_plan(request, profile, load_providers()).to_dict()
        payload.update(
            _build_request_resolution(
                raw_workload=workload,
                raw_budget=budget,
                raw_deadline=deadline,
                request=request,
                model=model,
                params_b=params_b,
                min_vram_gb=min_vram_gb,
            )
        )
        payload["client_guidance"] = _client_guidance("plan_provider_workflow", follow_up="rank_providers_for_task")
        return payload

    @mcp.tool(
        title="Rank Providers For Task",
        description="Return a provider shortlist for a known task shape without building the full staged workflow.",
        structured_output=True,
    )
    def rank_providers_for_task(
        workload: WorkloadArg,
        model: ModelArg = None,
        params_b: ParamsArg = None,
        budget: BudgetArg = "under-25",
        task_hours: TaskHoursArg = 1.0,
        min_vram_gb: VRAMArg = None,
        parallel_jobs: ParallelJobsArg = 1,
        requires_api: bool = False,
        prefer_local: bool = True,
        deadline: DeadlineArg = "flexible",
        limit: LimitArg = 5,
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
        ranked = rank_providers(load_providers(), request, stage=None, limit=request.limit)
        return [provider.to_dict() for provider in ranked]

    @mcp.tool(
        title="Assess Task Compute",
        description="Estimate the compute lane and rough VRAM/time needs before choosing providers.",
        structured_output=True,
    )
    def assess_task_compute(
        workload: WorkloadArg,
        model: ModelArg = None,
        params_b: ParamsArg = None,
        task_hours: TaskHoursArg = 1.0,
        min_vram_gb: VRAMArg = None,
        parallel_jobs: ParallelJobsArg = 1,
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
        payload = assess_compute_need(request).to_dict()
        payload.update(
            _build_request_resolution(
                raw_workload=workload,
                raw_budget="under-25",
                raw_deadline="flexible",
                request=request,
                model=model,
                params_b=params_b,
                min_vram_gb=min_vram_gb,
            )
        )
        payload["client_guidance"] = _client_guidance("assess_task_compute", follow_up="plan_provider_workflow")
        return payload

    @mcp.resource(
        "providers://snapshot",
        title="Provider Snapshot",
        description="Summarize provider coverage, budget buckets, and canonical argument values.",
        mime_type="application/json",
    )
    def providers_snapshot() -> str:
        """Summarize the provider dataset and budget buckets."""
        return json.dumps(_provider_snapshot(), indent=2)

    @mcp.resource(
        "guide://tool-selection",
        title="Tool Selection Guide",
        description="Explain which free-gpu tool to call first and how to pass canonical arguments.",
        mime_type="application/json",
    )
    def tool_selection_guide() -> str:
        return json.dumps(_tool_selection_guide(), indent=2)

    @mcp.prompt(
        name="choose_free_gpu_tool",
        title="Choose The Right free-gpu Tool",
        description="Generate a concise plan for selecting the correct free-gpu tool and canonical arguments before answering the end user.",
    )
    def choose_free_gpu_tool(user_goal: str) -> list[dict[str, str]]:
        guide = json.dumps(_tool_selection_guide(), indent=2)
        return [
            {
                "role": "user",
                "content": (
                    "You are preparing a free-gpu MCP call.\n"
                    f"User goal: {user_goal}\n\n"
                    "Use this guide to decide which tool to call and which canonical arguments to send:\n"
                    f"{guide}\n\n"
                    "Return a short plan that names the tool, the canonical workload, and any missing but optional inputs. "
                    "Do not mention internal fallback reasoning unless the user asked."
                ),
            }
        ]

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
