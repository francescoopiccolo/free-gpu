from __future__ import annotations

import re

from .models import (
    ComputeNeed,
    LocalCapabilityProfile,
    ProviderRecord,
    RankedProvider,
    RecommendationPlan,
    WorkloadRequest,
    WorkflowStep,
)


WORKLOAD_STAGES = {
    "scratch-train": ["dataset-prep", "training", "evaluation"],
    "finetune-lora": ["dataset-prep", "finetune", "evaluation", "serving"],
    "inference": ["inference"],
    "batch-eval": ["evaluation"],
    "agent-loop": ["inference", "evaluation"],
}

DEFAULT_TASK_HOURS = {
    "scratch-train": 18.0,
    "finetune-lora": 8.0,
    "inference": 1.0,
    "batch-eval": 2.0,
    "agent-loop": 2.5,
}

DEFAULT_VRAM_BY_WORKLOAD = {
    "scratch-train": 24.0,
    "finetune-lora": 16.0,
    "inference": 8.0,
    "batch-eval": 8.0,
    "agent-loop": 10.0,
}

STAGE_HOUR_MULTIPLIER = {
    "dataset-prep": 0.25,
    "training": 1.0,
    "finetune": 1.0,
    "evaluation": 0.35,
    "serving": 0.4,
    "inference": 1.0,
}

STAGE_VRAM_MULTIPLIER = {
    "dataset-prep": 0.35,
    "training": 1.0,
    "finetune": 1.0,
    "evaluation": 0.6,
    "serving": 0.5,
    "inference": 1.0,
}


def build_plan(
    request: WorkloadRequest,
    local_profile: LocalCapabilityProfile,
    providers: list[ProviderRecord],
) -> RecommendationPlan:
    compute_need = assess_compute_need(request)
    local_verdict, blockers = evaluate_local_fit(request, local_profile)
    ranked = rank_providers(providers, request, stage=None, limit=request.limit)
    steps = build_workflow_steps(request, local_profile, providers, local_verdict)
    summary = summarize_plan(request, local_verdict, local_profile, compute_need)
    return RecommendationPlan(
        request=request,
        local_profile=local_profile,
        local_verdict=local_verdict,
        summary=summary,
        compute_need=compute_need,
        top_providers=ranked,
        workflow_steps=steps,
        blockers=blockers,
    )


def rank_providers(
    providers: list[ProviderRecord],
    request: WorkloadRequest,
    *,
    stage: str | None,
    limit: int,
) -> list[RankedProvider]:
    compute_need = assess_compute_need(request, stage=stage)
    ranked: list[RankedProvider] = []
    for provider in providers:
        score, reasons = score_provider(provider, request, stage=stage, compute_need=compute_need)
        if score <= 0:
            continue
        ranked.append(
            RankedProvider(
                service=provider.service,
                provider=provider.provider,
                score=score,
                reason="; ".join(reasons[:3]),
                source_url=provider.source_url,
                signup_link=provider.signup_link,
                category=provider.category,
                free_tier=provider.free_tier,
                compute=provider.compute,
                notes=provider.notes,
                credit_card_required=provider.credit_card_required,
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.service.lower()))
    return ranked[:limit]


def build_workflow_steps(
    request: WorkloadRequest,
    local_profile: LocalCapabilityProfile,
    providers: list[ProviderRecord],
    local_verdict: str,
) -> list[WorkflowStep]:
    steps: list[WorkflowStep] = []
    for stage in WORKLOAD_STAGES[request.workload]:
        compute_need = assess_compute_need(request, stage=stage)
        ranked = rank_providers(providers, request, stage=stage, limit=min(request.limit, 3))
        environment, reason, blockers = choose_stage_environment(
            request=request,
            local_profile=local_profile,
            local_verdict=local_verdict,
            stage=stage,
            compute_need=compute_need,
        )
        steps.append(
            WorkflowStep(
                stage=stage,
                recommended_environment=environment,
                reason=reason,
                compute_need=compute_need,
                suggested_providers=ranked if environment != "local" else ranked[:2],
                blockers=blockers,
            )
        )
    return steps


def assess_compute_need(request: WorkloadRequest, stage: str | None = None) -> ComputeNeed:
    params_b = request.params_b if request.params_b is not None else infer_model_size(request.model)
    base_vram = request.min_vram_gb if request.min_vram_gb is not None else infer_required_vram(request.workload, params_b)
    total_hours = request.task_hours if request.task_hours > 0 else DEFAULT_TASK_HOURS.get(request.workload, 1.0)
    parallel_jobs = max(request.parallel_jobs, 1)

    if stage:
        hours = max(0.5, round(total_hours * STAGE_HOUR_MULTIPLIER.get(stage, 1.0), 1))
        required_vram = round(base_vram * STAGE_VRAM_MULTIPLIER.get(stage, 1.0), 1)
        if stage == "dataset-prep":
            required_vram = max(4.0, required_vram)
        elif stage in {"evaluation", "serving"}:
            required_vram = max(6.0, required_vram)
        if stage == "dataset-prep":
            parallel_jobs = 1
    else:
        hours = total_hours
        required_vram = base_vram

    lane = classify_compute_lane(
        workload=request.workload,
        required_vram_gb=required_vram,
        estimated_hours=hours,
        parallel_jobs=parallel_jobs,
        requires_api=request.requires_api,
        stage=stage,
    )
    summary = describe_compute_need(lane, required_vram, hours, parallel_jobs, stage or request.workload)
    return ComputeNeed(
        lane=lane,
        summary=summary,
        estimated_hours=hours,
        required_vram_gb=required_vram,
        parallel_jobs=parallel_jobs,
    )


def evaluate_local_fit(
    request: WorkloadRequest,
    local_profile: LocalCapabilityProfile,
) -> tuple[str, list[str]]:
    compute_need = assess_compute_need(request)
    params_b = request.params_b if request.params_b is not None else infer_model_size(request.model)
    vram = local_profile.vram_gb or 0
    blockers: list[str] = []

    if not local_profile.has_hardware_data():
        return "unknown", blockers

    if compute_need.required_vram_gb and vram >= compute_need.required_vram_gb and compute_need.estimated_hours <= 4:
        if request.workload in {"inference", "agent-loop", "batch-eval"}:
            return "good-local", blockers

    if request.workload == "scratch-train":
        if vram >= 16:
            blockers.append("Scratch training is educationally viable locally, but free-tier overflow is likely still needed.")
            return "partial-local", blockers
        blockers.append("From-scratch training is usually not realistic on free or consumer hardware beyond toy-scale experiments.")
        return "cloud-assisted", blockers

    if request.workload == "finetune-lora":
        if params_b and params_b <= 7 and vram >= 16:
            return "good-local", blockers
        if params_b and params_b <= 8 and vram >= 12:
            blockers.append("Local evaluation and light prep are viable, but remote fine-tuning is likely safer.")
            return "partial-local", blockers
        blockers.append("Local VRAM is likely too small for comfortable LoRA fine-tuning.")
        return "cloud-assisted", blockers

    if request.workload in {"inference", "agent-loop"}:
        if params_b and params_b <= 14 and vram >= 12:
            return "good-local", blockers
        if params_b and params_b <= 7 and vram >= 8:
            return "good-local", blockers
        if vram >= 6:
            blockers.append("A smaller or more aggressively quantized model is likely required for reliable local use.")
            return "partial-local", blockers
        blockers.append("Local hardware looks better suited for cloud-assisted inference.")
        return "cloud-assisted", blockers

    if request.workload == "batch-eval":
        if vram >= 8 or (local_profile.ram_gb or 0) >= 32:
            blockers.append("Local eval is possible for smaller runs, but remote APIs may be more convenient.")
            return "partial-local", blockers
        blockers.append("Batch evaluation will likely be easier with API-capable providers.")
        return "cloud-assisted", blockers

    return "unknown", blockers


def summarize_plan(
    request: WorkloadRequest,
    local_verdict: str,
    local_profile: LocalCapabilityProfile,
    compute_need: ComputeNeed,
) -> str:
    if not local_profile.has_hardware_data():
        return f"Local hardware was not described, so the planner is using provider data only to build a {compute_need.lane} workflow."
    if request.workload == "inference" and local_verdict == "good-local":
        return f"Local inference looks viable. Keep {compute_need.lane} providers ready as overflow when the task outgrows local capacity."
    if local_verdict == "cloud-assisted":
        return f"A hybrid or cloud-first path is more realistic than a fully local workflow. This request looks like a {compute_need.lane} compute lane."
    if local_profile.llmfit_available:
        return f"llmfit can handle local-fit detection here, and free-gpu is filling in the {compute_need.lane} cloud workflow around it."
    return f"The planner is using the provider dataset plus manual hardware hints to build a {compute_need.lane} workflow."


def choose_stage_environment(
    *,
    request: WorkloadRequest,
    local_profile: LocalCapabilityProfile,
    local_verdict: str,
    stage: str,
    compute_need: ComputeNeed,
) -> tuple[str, str, list[str]]:
    if stage == "dataset-prep":
        return "local", "Dataset prep and preprocessing are usually cheapest and simplest locally.", []

    if (
        request.workload == "inference"
        and local_verdict == "good-local"
        and (compute_need.required_vram_gb or 0) <= (local_profile.vram_gb or 0)
    ):
        return "local", "Your local profile looks capable enough for the requested inference workload.", []

    if request.workload == "agent-loop" and local_verdict == "good-local" and (local_profile.ram_gb or 0) >= 16:
        return "local", "A local-first agent loop is viable, with remote providers reserved for heavier runs.", []

    if stage in {"finetune", "training"}:
        if (
            local_verdict == "good-local"
            and (local_profile.vram_gb or 0) >= (compute_need.required_vram_gb or 0)
            and compute_need.estimated_hours <= 6
        ):
            return "local", "Local hardware is strong enough to attempt this stage directly.", []
        if compute_need.lane == "grant-scale":
            return "remote", "This stage needs more sustained compute than consumer free tiers usually offer, so grant-style remote access is the better fit.", []
        return "remote", "A remote GPU is more realistic for this stage because VRAM and persistence matter more here.", []

    if stage == "serving" and local_verdict in {"good-local", "partial-local"}:
        return "local", "Serve locally when possible and keep remote providers as fallback.", []

    if stage == "evaluation" and request.workload in {"batch-eval", "agent-loop"} and request.requires_api:
        return "remote", "API-friendly providers are the better fit for repeated evaluation loops.", []

    if local_verdict == "cloud-assisted":
        return "remote", f"Remote providers are the safer path for this {compute_need.lane} stage.", []

    return "hybrid", f"A hybrid path keeps local costs low while using remote providers in the {compute_need.lane} lane only where they help.", []


def score_provider(
    provider: ProviderRecord,
    request: WorkloadRequest,
    *,
    stage: str | None,
    compute_need: ComputeNeed | None = None,
) -> tuple[int, list[str]]:
    compute_need = compute_need or assess_compute_need(request, stage=stage)
    score = provider.compute_score
    reasons = [f"compute score {provider.compute_score}"]
    text = provider.text_blob
    target = stage or request.workload

    if request.budget == "free":
        if "free tier" in text or "free" in provider.free_tier.lower():
            score += 18
            reasons.append("clear free-tier path")
        if "no" in provider.credit_card_required.lower():
            score += 8
            reasons.append("no card required")
    elif request.budget == "under-25":
        if any(token in text for token in ("trial", "credits", "free", "starter", "lite", "promotional")):
            score += 12
            reasons.append("fits near-free budget")
    elif request.budget == "grant":
        if _is_grant_like(provider):
            score += 22
            reasons.append("grant-style access path")
        else:
            score -= 12

    if request.requires_api or target in {"agent-loop", "evaluation", "inference"}:
        if "yes" in provider.api_available.lower() or "api" in text:
            score += 14
            reasons.append("api-friendly")
        elif target in {"agent-loop", "evaluation"}:
            score -= 12

    if target in {"training", "finetune", "scratch-train", "finetune-lora"}:
        if any(token in text for token in ("docker", "notebook", "ml platform", "cloud gpu", "grant", "credits")):
            score += 16
            reasons.append("usable for training-style workflows")
        if "inference api" in text:
            score -= 18

    if target == "agent-loop":
        if any(token in text for token in ("docker", "api", "serverless", "global")):
            score += 12
            reasons.append("suited to tool-driven agent loops")

    if target in {"evaluation", "batch-eval"}:
        if any(token in text for token in ("api", "docker", "batch")):
            score += 12
            reasons.append("friendly for repeated eval runs")

    if request.deadline == "urgent":
        if any(token in text for token in ("grant", "application", "proposal", "academic")):
            score -= 12
            reasons.append("slower access path")
        if "global" in text or "free tier" in text:
            score += 4

    if request.prefer_local and target in {"inference", "serving"}:
        score -= 4

    score += _score_for_compute_need(provider, compute_need, target, reasons)

    return max(score, 0), reasons


def infer_required_vram(workload: str, params_b: float | None) -> float:
    base = DEFAULT_VRAM_BY_WORKLOAD.get(workload, 8.0)
    if params_b is None:
        return base

    multiplier = {
        "scratch-train": 4.0,
        "finetune-lora": 2.0,
        "inference": 0.9,
        "batch-eval": 0.7,
        "agent-loop": 1.0,
    }.get(workload, 1.0)
    return max(base, round(params_b * multiplier, 1))


def classify_compute_lane(
    *,
    workload: str,
    required_vram_gb: float,
    estimated_hours: float,
    parallel_jobs: int,
    requires_api: bool,
    stage: str | None,
) -> str:
    if workload == "scratch-train" or required_vram_gb >= 32 or estimated_hours >= 18:
        return "grant-scale"
    if required_vram_gb >= 20 or estimated_hours >= 8 or parallel_jobs >= 4:
        return "heavy"
    if required_vram_gb >= 12 or estimated_hours >= 3 or parallel_jobs >= 2:
        return "session"
    if requires_api and stage in {"evaluation", "inference"}:
        return "session"
    return "burst"


def describe_compute_need(
    lane: str,
    required_vram_gb: float | None,
    estimated_hours: float,
    parallel_jobs: int,
    stage_label: str,
) -> str:
    vram_text = f"{required_vram_gb:.1f} GB VRAM" if required_vram_gb is not None else "unknown VRAM"
    return (
        f"{lane} lane for {stage_label}: about {estimated_hours:.1f}h, "
        f"{vram_text}, {parallel_jobs} parallel job{'s' if parallel_jobs != 1 else ''}."
    )


def _score_for_compute_need(
    provider: ProviderRecord,
    compute_need: ComputeNeed,
    target: str,
    reasons: list[str],
) -> int:
    text = provider.text_blob
    score = 0
    lane = compute_need.lane
    required_vram = compute_need.required_vram_gb or 0

    if lane == "burst":
        if any(token in text for token in ("free tier", "trial", "no card", "serverless", "notebook")):
            score += 12
            reasons.append("good for short burst tasks")
        if any(token in text for token in ("h100", "a100", "allocation", "proposal")):
            score -= 4
    elif lane == "session":
        if any(token in text for token in ("credits", "trial", "docker", "notebook", "cloud gpu", "l4", "t4", "v100", "a10g")):
            score += 14
            reasons.append("good for medium sessions")
        if _is_grant_like(provider):
            score -= 4
    elif lane == "heavy":
        if any(token in text for token in ("a100", "h100", "h200", "mi300x", "l40", "v100", "cloud gpu", "hpc")):
            score += 18
            reasons.append("strong fit for heavier compute")
        if any(token in text for token in ("credits", "gpu hours", "hours", "global")):
            score += 8
        if "cpu only" in text:
            score -= 18
    elif lane == "grant-scale":
        if _is_grant_like(provider):
            score += 22
            reasons.append("grant-scale capacity")
        if any(token in text for token in ("hpc", "allocation", "supercomputer", "a100", "h100", "mi300x")):
            score += 12
        if any(token in text for token in ("free tier", "lite", "starter")):
            score -= 10

    if required_vram >= 24:
        if any(token in text for token in ("h100", "a100", "h200", "mi300x", "80gb", "48gb")):
            score += 12
            reasons.append("high-memory signal")
        elif any(token in text for token in ("t4", "l4", "p100")):
            score -= 6

    if compute_need.estimated_hours >= 6:
        if any(token in text for token in ("gpu hours", "credits", "program", "grant", "allocation")):
            score += 8
        if target in {"training", "finetune"} and "free tier" in text and "credits" not in text:
            score -= 10

    if compute_need.parallel_jobs >= 2:
        if "api" in text or "serverless" in text:
            score += 8
            reasons.append("handles parallel task fan-out")
        elif target in {"evaluation", "inference"}:
            score -= 6

    return score


def _is_grant_like(provider: ProviderRecord) -> bool:
    text = f"{provider.free_tier} {provider.category} {provider.notes}".lower()
    return any(token in text for token in ("grant", "program", "application", "allocation"))


def infer_model_size(model_name: str | None) -> float | None:
    if not model_name:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*b", model_name.lower())
    if match:
        return float(match.group(1))
    return None
