from __future__ import annotations

from dataclasses import replace

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Input, Static

from .data import load_providers
from .llmfit_adapter import load_local_profile
from .models import LocalCapabilityProfile, RankedProvider, WorkloadRequest
from .planner import build_plan, score_provider


WORKLOAD_OPTIONS = [
    ("All", "all"),
    ("Inference", "inference"),
    ("Fine-tune", "finetune-lora"),
    ("Train", "scratch-train"),
    ("Eval", "batch-eval"),
    ("Agent", "agent-loop"),
]
BUDGET_OPTIONS = [("Budget Any", "any"), ("Free", "free"), ("<$10", "under-10"), ("<$25", "under-25"), ("Flexible", "flexible")]
ROLE_OPTIONS = [("Student", "student"), ("Researcher", "researcher"), ("Founder", "founder"), ("Other", "other")]
ROLE_ANY_OPTIONS = [("Role Any", "any")] + ROLE_OPTIONS
ACCESS_OPTIONS = [("Access Any", "any"), ("Fast start", "urgent"), ("Can wait", "flexible")]


class FreeGpuApp(App):
    CSS_PATH = "tui.tcss"
    TITLE = "free-gpu"
    SUB_TITLE = "hybrid compute planner"
    BINDINGS = [("r", "reload_local", "Reload"), ("q", "quit", "Quit")]

    def __init__(self, *, initial_request: WorkloadRequest, manual_hardware: dict) -> None:
        super().__init__()
        self.initial_request = initial_request
        self.manual_hardware = manual_hardware
        self.provider_records = load_providers()
        self.local_profile: LocalCapabilityProfile | None = None
        self.current_ranked: list[RankedProvider] = []
        self.current_plan = None
        self.workload_idx = 0
        self.role_idx = 0
        self.budget_idx = 0
        self.deadline_idx = 0
        self.no_card_only = False

    def compose(self) -> ComposeResult:
        yield Static("", id="system-bar")
        yield Horizontal(
            Input(placeholder="Search providers or notes", id="search"),
            Button("", id="role"),
            Button("", id="workload"),
            Button("", id="budget"),
            Button("", id="access"),
            Button("", id="no-card"),
            id="filters-bar",
        )
        yield DataTable(id="providers-table")
        yield Horizontal(
            Static("", id="selection-bar"),
            Static("", id="flow-bar"),
            id="status-bars",
        )
        yield Static("", id="help-bar")

    def on_mount(self) -> None:
        table = self.query_one("#providers-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns(" ", "Service", "Provider", "Use Case", "Rec", "Fit", "Compute", "Card", "API", "Tier")
        self._refresh_filter_labels()
        self.refresh_local_profile()
        self.refresh_plan()

    def action_reload_local(self) -> None:
        self.refresh_local_profile()
        self.refresh_plan()

    def refresh_local_profile(self) -> None:
        self.local_profile = load_local_profile(
            ram_gb=self.manual_hardware.get("ram_gb"),
            vram_gb=self.manual_hardware.get("vram_gb"),
            gpu_name=self.manual_hardware.get("gpu_name"),
            llmfit_limit=self.manual_hardware.get("llmfit_limit", 5),
        )
        self.query_one("#system-bar", Static).update(self._system_bar_text(self.local_profile))

    def refresh_plan(self) -> None:
        if self.local_profile is None:
            self.refresh_local_profile()
        request = self._current_request()
        request.limit = len(self.provider_records)
        plan = build_plan(request, self.local_profile, self.provider_records)
        ranked = self._build_ranked_list_for_ui(request)
        ranked = self._apply_search_filters(ranked)
        self.current_plan = plan
        self.current_ranked = ranked
        self._render_table(ranked, plan)
        self._update_selection_bar(0)
        self._update_flow_bar(plan)
        self.query_one("#help-bar", Static).update(
            "Tab moves focus. Enter or click cycles a filter. Access means how quickly you need access. Arrows move in the table. r reloads llmfit. q quits."
        )

    def _apply_search_filters(self, ranked: list[RankedProvider]) -> list[RankedProvider]:
        search = self.query_one("#search", Input).value.strip().lower()

        filtered = ranked
        if search:
            filtered = [
                provider
                for provider in filtered
                if search in provider.service.lower()
                or search in provider.provider.lower()
                or search in provider.reason.lower()
                or search in provider.notes.lower()
                or search in provider.compute.lower()
            ]
        return filtered

    def _build_ranked_list_for_ui(self, request: WorkloadRequest) -> list[RankedProvider]:
        role = ROLE_ANY_OPTIONS[self.role_idx][1]
        ui_workload = WORKLOAD_OPTIONS[self.workload_idx][1]
        ranked: list[RankedProvider] = []

        for provider in self.provider_records:
            if self.no_card_only and not self._provider_no_card(provider):
                continue
            if not self._provider_matches_budget(provider, request.budget):
                continue
            if not self._provider_matches_access(provider, ACCESS_OPTIONS[self.deadline_idx][1]):
                continue
            if ui_workload != "all" and not self._provider_matches_workload(provider, ui_workload):
                continue

            if ui_workload == "all":
                score = provider.compute_score + self._score_for_role(provider, role)
                reasons = [f"role fit for {role}", f"compute score {provider.compute_score}"]
            else:
                score, reasons = score_provider(provider, request, stage=None)
                score += self._score_for_role(provider, role)
            ranked.append(
                RankedProvider(
                    service=provider.service,
                    provider=provider.provider,
                    score=max(score, 0),
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
        return ranked

    @on(Input.Changed, "#search")
    def handle_search(self) -> None:
        self.refresh_plan()

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "role":
            self.role_idx = (self.role_idx + 1) % len(ROLE_ANY_OPTIONS)
        elif button_id == "workload":
            self.workload_idx = (self.workload_idx + 1) % len(WORKLOAD_OPTIONS)
        elif button_id == "budget":
            self.budget_idx = (self.budget_idx + 1) % len(BUDGET_OPTIONS)
        elif button_id == "access":
            self.deadline_idx = (self.deadline_idx + 1) % len(ACCESS_OPTIONS)
        elif button_id == "no-card":
            self.no_card_only = not self.no_card_only
        else:
            return
        self._refresh_filter_labels()
        self.refresh_plan()

    @on(DataTable.RowHighlighted)
    def handle_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._update_selection_bar(event.cursor_row)

    def _current_request(self) -> WorkloadRequest:
        request = replace(self.initial_request)
        selected_workload = WORKLOAD_OPTIONS[self.workload_idx][1]
        request.workload = "inference" if selected_workload == "all" else selected_workload
        request.budget = BUDGET_OPTIONS[self.budget_idx][1]
        selected_deadline = ACCESS_OPTIONS[self.deadline_idx][1]
        request.deadline = "flexible" if selected_deadline == "any" else selected_deadline
        request.requires_api = False
        request.prefer_local = True
        return request

    def _refresh_filter_labels(self) -> None:
        self.query_one("#role", Button).label = ROLE_ANY_OPTIONS[self.role_idx][0]
        self.query_one("#workload", Button).label = WORKLOAD_OPTIONS[self.workload_idx][0]
        self.query_one("#budget", Button).label = BUDGET_OPTIONS[self.budget_idx][0]
        self.query_one("#access", Button).label = ACCESS_OPTIONS[self.deadline_idx][0]
        self.query_one("#no-card", Button).label = "No card" if self.no_card_only else "Card any"

    def _render_table(self, ranked: list[RankedProvider], plan) -> None:
        table = self.query_one("#providers-table", DataTable)
        table.clear(columns=False)
        for index, provider in enumerate(ranked, start=1):
            ui_workload = WORKLOAD_OPTIONS[self.workload_idx][1]
            use_case = self._use_case_for(ui_workload)
            fit = self._fit_label(plan.local_verdict, plan, provider)
            table.add_row(
                self._indicator_for(fit),
                provider.service[:28],
                provider.provider[:10],
                use_case,
                self._recommendation_label(provider.score),
                fit,
                self._truncate(provider.compute, 16),
                self._card_label(provider),
                self._api_label(provider),
                self._tier_label(provider),
            )
        if ranked:
            table.move_cursor(row=0, animate=False)

    def _update_selection_bar(self, row_index: int) -> None:
        if not self.current_ranked or row_index < 0 or row_index >= len(self.current_ranked):
            self.query_one("#selection-bar", Static).update("No provider selected.")
            return
        provider = self.current_ranked[row_index]
        text = (
            f"[b]{provider.service}[/b]  "
            f"Prov {provider.provider}  "
            f"Recommendation {self._recommendation_label(provider.score)}  "
            f"Type {provider.category}  "
            f"Card {self._card_label(provider)}  "
            f"API {self._api_label(provider)}  "
            f"Tier {self._tier_label(provider)}\n"
            f"{provider.reason}\n"
            f"Signup: {self._link(provider.signup_link)}\n"
            f"Source: {self._link(provider.source_url)}"
        )
        self.query_one("#selection-bar", Static).update(text)

    def _update_flow_bar(self, plan) -> None:
        ui_workload = WORKLOAD_OPTIONS[self.workload_idx][1]
        if ui_workload == "all":
            lines = [
                "[b]Full provider list[/b]",
                f"Role ranking: {ROLE_ANY_OPTIONS[self.role_idx][0]}",
                "Showing all providers by default.",
                "Card, Budget, Work, and Access now hide non-matching providers.",
                "Access = how quickly you need access: Fast start hides grants and application-based programs.",
            ]
            self.query_one("#flow-bar", Static).update("\n".join(lines))
            return
        lines = [
            f"[b]Local[/b] {plan.local_verdict}",
            plan.summary,
        ]
        for step in plan.workflow_steps[:3]:
            lines.append(f"{step.stage}: {step.recommended_environment}")
        self.query_one("#flow-bar", Static).update("\n".join(lines))

    def _system_bar_text(self, profile: LocalCapabilityProfile) -> str:
        models = ", ".join(model.name.split("/")[-1] for model in profile.top_local_models[:3]) or "none"
        return (
            f"[b]free-gpu[/b]\n"
            f"CPU/RAM source: {profile.source}  |  "
            f"RAM: {self._fmt_num(profile.ram_gb)}  |  "
            f"GPU: {profile.gpu_name or 'unknown'}  |  "
            f"VRAM: {self._fmt_num(profile.vram_gb)}  |  "
            f"llmfit: {'yes' if profile.llmfit_available else 'no'}\n"
            f"Top local fits: {models}"
        )

    @staticmethod
    def _fmt_num(value: float | None) -> str:
        return f"{value:.1f} GB" if value is not None else "unknown"

    @staticmethod
    def _truncate(value: str, width: int) -> str:
        return value if len(value) <= width else value[: width - 3] + "..."

    @staticmethod
    def _use_case_for(workload: str) -> str:
        mapping = {
            "all": "All",
            "scratch-train": "Build",
            "finetune-lora": "Tune",
            "inference": "Infer",
            "batch-eval": "Eval",
            "agent-loop": "Agent",
        }
        return mapping.get(workload, "All")

    @staticmethod
    def _indicator_for(fit: str) -> str:
        if fit == "Perfect":
            return "●"
        if fit == "Good":
            return "●"
        if fit == "Hybrid":
            return "◐"
        return "○"

    @staticmethod
    def _tier_label(provider: RankedProvider) -> str:
        text = provider.free_tier.lower()
        if "free" in text:
            return "Free"
        if "trial" in text or "credit" in text:
            return "Credit"
        if "grant" in text or "program" in text:
            return "Grant"
        return "Mixed"

    @staticmethod
    def _api_label(provider: RankedProvider) -> str:
        haystack = f"{provider.reason} {provider.compute} {provider.notes}".lower()
        return "Yes" if "api" in haystack else "No"

    @staticmethod
    def _card_label(provider: RankedProvider) -> str:
        text = f"{provider.reason} {provider.notes} {provider.free_tier}".lower()
        if "no card" in text:
            return "No"
        return "Maybe"

    @staticmethod
    def _fit_label(local_verdict: str, plan, provider: RankedProvider) -> str:
        if plan.request.workload == "inference" and local_verdict == "good-local":
            return "Perfect"
        if any(step.recommended_environment == "hybrid" for step in plan.workflow_steps):
            return "Hybrid"
        if any(step.recommended_environment == "remote" for step in plan.workflow_steps):
            return "Remote"
        if provider.score >= 90:
            return "Good"
        return "Marginal"

    @staticmethod
    def _provider_has_api(provider) -> bool:
        text = f"{getattr(provider, 'api_available', '')} {provider.notes} {provider.compute}".lower()
        return "yes" in getattr(provider, "api_available", "").lower() or "api" in text or "cluster apis" in text

    @staticmethod
    def _provider_no_card(provider) -> bool:
        return "no" in getattr(provider, "credit_card_required", "").lower()

    @staticmethod
    def _provider_matches_budget(provider, budget: str) -> bool:
        text = provider.free_tier.lower()
        if budget == "any":
            return True
        if budget == "free":
            return "free" in text
        if budget == "under-10":
            return "free" in text or "trial" in text or "credit" in text
        if budget == "under-25":
            return "free" in text or "trial" in text or "credit" in text or "grant" in text or "program" in text
        return True

    @staticmethod
    def _provider_matches_access(provider, access: str) -> bool:
        bag = " ".join([provider.category, provider.free_tier, provider.notes]).lower()
        if access == "urgent":
            return not any(token in bag for token in ("grant", "application", "proposal", "allocation", "academic"))
        if access == "flexible":
            return any(token in bag for token in ("grant", "application", "proposal", "allocation", "academic", "research"))
        return True

    @staticmethod
    def _provider_matches_workload(provider, workload: str) -> bool:
        bag = " ".join(
            [
                provider.category,
                provider.compute,
                provider.free_tier,
                provider.notes,
                getattr(provider, "docker_support", ""),
                getattr(provider, "api_available", ""),
            ]
        ).lower()
        if workload == "finetune-lora":
            return any(token in bag for token in ("docker", "notebook", "cloud gpu", "ml platform", "gpu", "credits", "free tier"))
        if workload == "scratch-train":
            return any(token in bag for token in ("cloud gpu", "hpc", "grant", "allocation", "notebook", "gpu"))
        if workload == "batch-eval":
            return any(token in bag for token in ("api", "docker", "serverless", "cloud", "gpu", "notebook"))
        if workload == "agent-loop":
            return any(token in bag for token in ("api", "serverless", "docker", "global", "cloud"))
        if workload == "inference":
            return not any(token in bag for token in ("allocation only", "proposal required"))
        return True

    @staticmethod
    def _link(url: str) -> str:
        if not url:
            return "N/A"
        return f"[link={url}]{url}[/link]"

    @staticmethod
    def _recommendation_label(score: int) -> str:
        if score >= 90:
            return "Highly rec"
        if score >= 70:
            return "Recommended"
        if score >= 50:
            return "Worth a look"
        return "Specialized"

    @staticmethod
    def _score_for_role(provider, role: str) -> int:
        bag = " | ".join(
            [
                provider.service,
                provider.provider,
                provider.category,
                provider.free_tier,
                getattr(provider, "max_hours", ""),
                getattr(provider, "authentication", ""),
                getattr(provider, "credit_card_required", ""),
                provider.notes,
                provider.compute,
            ]
        ).lower()

        score = 0
        if "free" in bag:
            score += 10
        if "credit" in bag:
            score += 5
        if "gpu" in bag:
            score += 6

        if role == "student":
            if any(token in bag for token in ("student", "educate", "notebook", "kaggle", "colab", "studiolab", "github student")):
                score += 45
            if "no card" in bag:
                score += 18
            if "grant" in bag or "proposal" in bag:
                score -= 8

        if role == "researcher":
            if any(token in bag for token in ("research", "hpc", "grant", "allocation", "proposal", "academic", "university", "pilot")):
                score += 50
            if any(token in bag for token in ("10,000 gpu hours", "allocation", "supercomputer", "director's discretionary", "incite")):
                score += 25

        if role == "founder":
            if any(token in bag for token in ("startup", "founder", "activate", "program credits", "cloud trial", "microsoft for startups", "aws activate", "google for startups", "runpod startup", "ovhcloud startup")):
                score += 52
            if any(token in bag for token in ("api", "serverless", "automation", "modal")):
                score += 10

        if role == "other":
            if any(token in bag for token in ("free tier", "trial", "credits", "pay-as-you-go", "marketplace", "global")):
                score += 25

        return score

    @staticmethod
    def _index_for(options: list[tuple[str, str]], value: str) -> int:
        for index, (_, option_value) in enumerate(options):
            if option_value == value:
                return index
        return 0
