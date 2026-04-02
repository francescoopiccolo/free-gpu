from __future__ import annotations

import csv
from pathlib import Path

from .models import ProviderRecord


CSV_PATH = Path(__file__).resolve().parent.parent / "gpu_compute_database - Database.csv"


def load_providers(path: Path | None = None) -> list[ProviderRecord]:
    csv_path = path or CSV_PATH
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    providers: list[ProviderRecord] = []
    for row in rows:
        text_blob = " | ".join(
            [
                row.get("Service", ""),
                row.get("Provider", ""),
                row.get("Category", ""),
                row.get("GPU / Compute", ""),
                row.get("VRAM", ""),
                row.get("Free Tier Type", ""),
                row.get("Max GPU Hours / Credits", ""),
                row.get("Credit Card Required", ""),
                row.get("Docker Support", ""),
                row.get("API Available", ""),
                row.get("Notes", ""),
            ]
        ).lower()
        providers.append(
            ProviderRecord(
                service=row.get("Service", "Unknown"),
                provider=row.get("Provider", "Unknown"),
                category=row.get("Category", "Unknown"),
                compute=row.get("GPU / Compute", "N/A"),
                vram=row.get("VRAM", "N/A"),
                free_tier=row.get("Free Tier Type", "N/A"),
                max_hours=row.get("Max GPU Hours / Credits", "N/A"),
                authentication=row.get("Authentication", "N/A"),
                credit_card_required=row.get("Credit Card Required", "N/A"),
                docker_support=row.get("Docker Support", "N/A"),
                api_available=row.get("API Available", "N/A"),
                region=row.get("Region", "N/A"),
                signup_link=row.get("Signup Link", ""),
                source_url=row.get("Source URL", ""),
                notes=row.get("Notes", ""),
                compute_score=_estimate_compute_score(text_blob),
                text_blob=text_blob,
            )
        )
    return providers


def _estimate_compute_score(text: str) -> int:
    score = 15
    if any(token in text for token in ("h100", "a100", "h200", "mi300x")):
        score += 30
    if any(token in text for token in ("t4", "l4", "a10g", "v100", "p100")):
        score += 16
    if any(token in text for token in ("gpu hours", "30 h/week", "10,000", "hours", "credits")):
        score += 20
    if any(token in text for token in ("cpu only", "n/a", "varies")):
        score -= 8
    return max(1, min(100, score))
