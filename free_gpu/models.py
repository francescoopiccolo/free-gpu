from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ProviderRecord:
    service: str
    provider: str
    category: str
    compute: str
    vram: str
    free_tier: str
    max_hours: str
    authentication: str
    credit_card_required: str
    docker_support: str
    api_available: str
    region: str
    signup_link: str
    source_url: str
    notes: str
    compute_score: int
    text_blob: str


@dataclass(slots=True)
class LocalModelMatch:
    name: str
    fit: str = "unknown"
    score: float | None = None
    provider: str | None = None


@dataclass(slots=True)
class LocalCapabilityProfile:
    source: str
    llmfit_available: bool
    ram_gb: float | None = None
    vram_gb: float | None = None
    gpu_name: str | None = None
    system_summary: str | None = None
    top_local_models: list[LocalModelMatch] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "llmfit_available": self.llmfit_available,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "gpu_name": self.gpu_name,
            "system_summary": self.system_summary,
            "top_local_models": [asdict(model) for model in self.top_local_models],
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class WorkloadRequest:
    workload: str
    model: str | None = None
    params_b: float | None = None
    budget: str = "free"
    limit: int = 5
    prefer_local: bool = True
    requires_api: bool = False
    deadline: str = "flexible"


@dataclass(slots=True)
class RankedProvider:
    service: str
    provider: str
    score: int
    reason: str
    source_url: str
    signup_link: str
    category: str
    free_tier: str
    compute: str
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class WorkflowStep:
    stage: str
    recommended_environment: str
    reason: str
    suggested_providers: list[RankedProvider] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "recommended_environment": self.recommended_environment,
            "reason": self.reason,
            "suggested_providers": [provider.to_dict() for provider in self.suggested_providers],
            "blockers": self.blockers,
        }


@dataclass(slots=True)
class RecommendationPlan:
    request: WorkloadRequest
    local_profile: LocalCapabilityProfile
    local_verdict: str
    summary: str
    top_providers: list[RankedProvider]
    workflow_steps: list[WorkflowStep]
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request": asdict(self.request),
            "local_profile": self.local_profile.to_dict(),
            "local_verdict": self.local_verdict,
            "summary": self.summary,
            "top_providers": [provider.to_dict() for provider in self.top_providers],
            "workflow_steps": [step.to_dict() for step in self.workflow_steps],
            "blockers": self.blockers,
        }
