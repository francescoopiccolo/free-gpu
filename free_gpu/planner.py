from __future__ import annotations

import re

from .models import (
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


def build_plan(
    request: WorkloadRequest,
    local_profile: LocalCapabilityProfile,
    providers: list[ProviderRecord],
) -> RecommendationPlan:
    local_verdict, blockers = evaluate_local_fit(request, local_profile)
    ranked = rank_providers(providers, request, stage=None, limit=request.limit)
    steps = build_workflow_steps(request, local_profile, providers, local_verdict)
    summary = summarize_plan(request, local_verdict, local_profile)
    return RecommendationPlan(
        request=request,
        local_profile=local_profile,
        local_verdict=local_verdict,
        summary=summary,
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
    ranked: list[RankedProvider] = []
    for provider in providers:
        score, reasons = score_provider(provider, request, stage=stage)
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
        ranked = rank_providers(providers, request, stage=stage, limit=min(request.limit, 3))
        environment, reason, blockers = choose_stage_environment(
            request=request,
            local_profile=local_profile,
            local_verdict=local_verdict,
            stage=stage,
        )
        steps.append(
            WorkflowStep(
                stage=stage,
                recommended_environment=environment,
                reason=reason,
                suggested_providers=ranked if environment != "local" else ranked[:2],
                blockers=blockers,
            )
        )
    return steps


def evaluate_local_fit(
    request: WorkloadRequest,
    local_profile: LocalCapabilityProfile,
) -> tuple[str, list[str]]:
    params_b = request.params_b if request.params_b is not None else infer_model_size(request.model)
    vram = local_profile.vram_gb or 0
    blockers: list[str] = []

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
) -> str:
    if request.workload == "inference" and local_verdict == "good-local":
        return "Local inference looks viable. Use cloud providers as overflow or for larger-context jobs."
    if local_verdict == "cloud-assisted":
        return "A hybrid or cloud-first path is more realistic than a fully local workflow for this request."
    if local_profile.llmfit_available:
        return "llmfit can handle local-fit detection here, and free-gpu is filling in the cloud/free-tier workflow around it."
    return "The planner is using the provider dataset plus manual hardware hints to build the cheapest viable workflow."


def choose_stage_environment(
    *,
    request: WorkloadRequest,
    local_profile: LocalCapabilityProfile,
    local_verdict: str,
    stage: str,
) -> tuple[str, str, list[str]]:
    if stage == "dataset-prep":
        return "local", "Dataset prep and preprocessing are usually cheapest and simplest locally.", []

    if request.workload == "inference" and local_verdict == "good-local":
        return "local", "Your local profile looks capable enough for the requested inference workload.", []

    if request.workload == "agent-loop" and local_verdict == "good-local" and (local_profile.ram_gb or 0) >= 16:
        return "local", "A local-first agent loop is viable, with remote providers reserved for heavier runs.", []

    if stage in {"finetune", "training"}:
        if local_verdict == "good-local" and (local_profile.vram_gb or 0) >= 16:
            return "local", "Local hardware is strong enough to attempt this stage directly.", []
        return "remote", "A remote GPU is more realistic for this stage because VRAM and persistence matter more here.", []

    if stage == "serving" and local_verdict in {"good-local", "partial-local"}:
        return "local", "Serve locally when possible and keep remote providers as fallback.", []

    if stage == "evaluation" and request.workload in {"batch-eval", "agent-loop"} and request.requires_api:
        return "remote", "API-friendly providers are the better fit for repeated evaluation loops.", []

    if local_verdict == "cloud-assisted":
        return "remote", "Remote providers are the safer path for this stage.", []

    return "hybrid", "A hybrid path keeps local costs low while using remote GPUs only where they help.", []


def score_provider(
    provider: ProviderRecord,
    request: WorkloadRequest,
    *,
    stage: str | None,
) -> tuple[int, list[str]]:
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
    elif request.budget == "under-10":
        if any(token in text for token in ("trial", "credits", "free")):
            score += 10
            reasons.append("fits near-free budget")

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

    return max(score, 0), reasons


def infer_model_size(model_name: str | None) -> float | None:
    if not model_name:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*b", model_name.lower())
    if match:
        return float(match.group(1))
    return None
