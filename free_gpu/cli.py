from __future__ import annotations

import argparse
import json
import sys

from .data import load_providers
from .llmfit_adapter import load_local_profile
from .models import WorkloadRequest
from .planner import build_plan, rank_providers


WORKLOAD_CHOICES = ["scratch-train", "finetune-lora", "inference", "batch-eval", "agent-loop"]
BUDGET_CHOICES = ["free", "under-10", "under-25", "flexible"]
DEADLINE_CHOICES = ["flexible", "urgent"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="free-gpu", description="Plan local and free-tier GPU workflows around llmfit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ui_parser = subparsers.add_parser("ui", help="Launch the terminal UI.")
    _add_shared_request_args(ui_parser, workload_required=False)
    _add_local_args(ui_parser)

    plan_parser = subparsers.add_parser("plan", help="Build a staged local/cloud workflow plan.")
    _add_shared_request_args(plan_parser, workload_required=True)
    _add_local_args(plan_parser)
    plan_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    provider_parser = subparsers.add_parser("providers", help="Rank providers for a workload.")
    _add_shared_request_args(provider_parser, workload_required=True)
    provider_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    local_parser = subparsers.add_parser("local", help="Inspect the detected local profile.")
    _add_local_args(local_parser)
    local_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    return parser


def _add_shared_request_args(parser: argparse.ArgumentParser, *, workload_required: bool) -> None:
    parser.add_argument("--workload", required=workload_required, default="inference", choices=WORKLOAD_CHOICES)
    parser.add_argument("--model", help="Optional model name such as llama-3.1-8b.")
    parser.add_argument("--params-b", type=float, help="Optional parameter size in billions.")
    parser.add_argument("--budget", default="free", choices=BUDGET_CHOICES)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--deadline", default="flexible", choices=DEADLINE_CHOICES)
    parser.add_argument("--requires-api", action="store_true")
    parser.add_argument("--prefer-cloud", action="store_true", help="Prefer remote recommendations over local-first output.")


def _add_local_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ram-gb", type=float, help="Manual override for system RAM.")
    parser.add_argument("--vram-gb", type=float, help="Manual override for GPU VRAM.")
    parser.add_argument("--gpu-name", help="Manual override for GPU name.")
    parser.add_argument("--llmfit-limit", type=int, default=5, help="How many llmfit recommendations to import.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "ui":
        try:
            from .tui import FreeGpuApp
        except ModuleNotFoundError as exc:
            missing = exc.name or "textual"
            print(
                f"The terminal UI needs the optional dependency '{missing}'. "
                "Install the package with a Python environment that has pip, then run `free-gpu ui` again."
            )
            return 1

        app = FreeGpuApp(
            initial_request=WorkloadRequest(
                workload=args.workload,
                model=args.model,
                params_b=args.params_b,
                budget=args.budget,
                limit=args.limit,
                prefer_local=not args.prefer_cloud,
                requires_api=args.requires_api,
                deadline=args.deadline,
            ),
            manual_hardware={
                "ram_gb": args.ram_gb,
                "vram_gb": args.vram_gb,
                "gpu_name": args.gpu_name,
                "llmfit_limit": args.llmfit_limit,
            },
        )
        app.run()
        return 0

    if args.command == "local":
        profile = load_local_profile(
            ram_gb=args.ram_gb,
            vram_gb=args.vram_gb,
            gpu_name=args.gpu_name,
            llmfit_limit=args.llmfit_limit,
        )
        return _print_local(profile, as_json=args.json)

    providers = load_providers()
    request = WorkloadRequest(
        workload=args.workload,
        model=args.model,
        params_b=args.params_b,
        budget=args.budget,
        limit=args.limit,
        prefer_local=not args.prefer_cloud,
        requires_api=args.requires_api,
        deadline=args.deadline,
    )

    if args.command == "providers":
        ranked = rank_providers(providers, request, stage=None, limit=args.limit)
        if args.json:
            print(json.dumps([provider.to_dict() for provider in ranked], indent=2))
        else:
            _print_ranked(ranked, heading=f"Top providers for {request.workload}")
        return 0

    profile = load_local_profile(
        ram_gb=args.ram_gb,
        vram_gb=args.vram_gb,
        gpu_name=args.gpu_name,
        llmfit_limit=args.llmfit_limit,
    )
    plan = build_plan(request, profile, providers)
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        _print_plan(plan)
    return 0


def _print_local(profile, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(profile.to_dict(), indent=2))
        return 0

    print("Local profile")
    print(f"  source: {profile.source}")
    print(f"  llmfit available: {'yes' if profile.llmfit_available else 'no'}")
    print(f"  RAM: {_fmt_num(profile.ram_gb)}")
    print(f"  VRAM: {_fmt_num(profile.vram_gb)}")
    print(f"  GPU: {profile.gpu_name or 'unknown'}")
    if profile.top_local_models:
        print("  top local models:")
        for model in profile.top_local_models:
            print(f"    - {model.name} ({model.fit})")
    if profile.warnings:
        print("  warnings:")
        for warning in profile.warnings:
            print(f"    - {warning}")
    return 0


def _print_plan(plan) -> None:
    print("Plan summary")
    print(f"  workload: {plan.request.workload}")
    print(f"  model: {plan.request.model or 'unspecified'}")
    print(f"  local verdict: {plan.local_verdict}")
    print(f"  summary: {plan.summary}")
    print()

    print("Local profile")
    print(f"  source: {plan.local_profile.source}")
    print(f"  RAM: {_fmt_num(plan.local_profile.ram_gb)}")
    print(f"  VRAM: {_fmt_num(plan.local_profile.vram_gb)}")
    print(f"  GPU: {plan.local_profile.gpu_name or 'unknown'}")
    if plan.local_profile.warnings:
        for warning in plan.local_profile.warnings:
            print(f"  warning: {warning}")
    print()

    _print_ranked(plan.top_providers, heading="Top providers")
    print()
    print("Workflow")
    for step in plan.workflow_steps:
        print(f"  - {step.stage}: {step.recommended_environment}")
        print(f"    {step.reason}")
        for blocker in step.blockers:
            print(f"    blocker: {blocker}")
        for provider in step.suggested_providers[:2]:
            print(f"    provider: {provider.service} ({provider.score})")


def _print_ranked(ranked, *, heading: str) -> None:
    print(heading)
    for provider in ranked:
        print(f"  - {provider.service} [{provider.score}]")
        print(f"    {provider.reason}")
        if provider.signup_link:
            print(f"    signup: {provider.signup_link}")


def _fmt_num(value: float | None) -> str:
    return f"{value:.1f} GB" if value is not None else "unknown"


if __name__ == "__main__":
    sys.exit(main())
