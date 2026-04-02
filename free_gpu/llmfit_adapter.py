from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .models import LocalCapabilityProfile, LocalModelMatch


def llmfit_available() -> bool:
    return resolve_llmfit_executable() is not None


def resolve_llmfit_executable() -> str | None:
    env_override = os.environ.get("FREE_GPU_LLMFIT_BIN")
    candidates: list[Path] = []

    if env_override:
        candidates.append(Path(env_override))

    which_path = shutil.which("llmfit")
    if which_path:
        candidates.append(Path(which_path))

    repo_root = Path(__file__).resolve().parent.parent
    tools_dir = repo_root.parent / "tools"
    if tools_dir.exists():
        candidates.extend(sorted(tools_dir.glob("llmfit-*/*/llmfit.exe"), reverse=True))
        candidates.extend(sorted(tools_dir.glob("llmfit-*/*/llmfit"), reverse=True))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def load_local_profile(
    *,
    ram_gb: float | None = None,
    vram_gb: float | None = None,
    gpu_name: str | None = None,
    llmfit_limit: int = 5,
) -> LocalCapabilityProfile:
    warnings: list[str] = []
    top_models: list[LocalModelMatch] = []
    source = "manual"
    system_summary = None
    detected_ram = ram_gb
    detected_vram = vram_gb
    detected_gpu = gpu_name
    executable = resolve_llmfit_executable()
    available = executable is not None

    if available:
        source = "llmfit"
        system_payload, parse_warning = _run_llmfit_system(executable)
        if parse_warning:
            warnings.append(parse_warning)
        system_summary = json.dumps(system_payload, indent=2) if system_payload else None
        llmfit_ram, llmfit_vram, llmfit_gpu = _parse_system_payload(system_payload)
        detected_ram = detected_ram if detected_ram is not None else llmfit_ram
        detected_vram = detected_vram if detected_vram is not None else llmfit_vram
        detected_gpu = detected_gpu or llmfit_gpu
        top_models, rec_warning = _run_llmfit_recommendations(executable, limit=llmfit_limit)
        if rec_warning:
            warnings.append(rec_warning)
    else:
        warnings.append("llmfit is not installed; using manual hardware values only.")

    if detected_ram is None and detected_vram is None:
        warnings.append("No local hardware data was detected. Results will focus on provider planning.")

    return LocalCapabilityProfile(
        source=source,
        llmfit_available=available,
        ram_gb=detected_ram,
        vram_gb=detected_vram,
        gpu_name=detected_gpu,
        system_summary=system_summary,
        top_local_models=top_models,
        warnings=warnings,
    )


def _run_llmfit_system(executable: str) -> tuple[dict | None, str | None]:
    try:
        proc = subprocess.run(
            [executable, "system", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
        return payload, None
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return None, f"Could not read llmfit system output: {exc}"


def _run_llmfit_recommendations(executable: str, limit: int) -> tuple[list[LocalModelMatch], str | None]:
    try:
        proc = subprocess.run(
            [executable, "recommend", "-n", str(limit), "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(proc.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return [], f"Could not parse llmfit recommendations: {exc}"

    models: list[LocalModelMatch] = []
    for item in payload.get("models", [])[:limit]:
        models.append(
            LocalModelMatch(
                name=str(item.get("name", "unknown")),
                fit=str(item.get("fit_level") or item.get("fit") or "unknown"),
                score=_coerce_float(item.get("score")),
                provider=item.get("runtime_label") or item.get("provider"),
            )
        )
    return models, None


def _parse_system_payload(payload: dict | None) -> tuple[float | None, float | None, str | None]:
    system = payload.get("system", {}) if isinstance(payload, dict) else {}
    ram_gb = _coerce_float(system.get("total_ram_gb"))
    vram_gb = _coerce_float(system.get("gpu_vram_gb"))
    gpu_name = system.get("gpu_name")
    return ram_gb, vram_gb, str(gpu_name) if gpu_name else None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
