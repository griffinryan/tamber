"""
CLI entry point to run a one-off generation through the composition orchestrator.

Example:
    uv run --project worker python -m timbre_worker.generate --prompt "quiet ambient drones"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .app.models import GenerationRequest
from .app.settings import Settings
from .services.musicgen import MusicGenService
from .services.orchestrator import ComposerOrchestrator
from .services.planner import CompositionPlanner
from .services.riffusion import RiffusionService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio via the Timbre worker backend.")
    parser.add_argument("--prompt", required=True, help="Prompt to feed into the generator.")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Optional clip duration override in seconds (1-30).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional model id override (defaults to worker settings).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Override artifact directory (defaults to worker settings).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Override config directory (defaults to worker settings).",
    )
    return parser.parse_args()


async def _run(
    prompt: str,
    *,
    duration: Optional[int],
    model_id: Optional[str],
    artifact_dir: Optional[Path],
    config_dir: Optional[Path],
) -> None:
    settings_kwargs: dict[str, object] = {}
    if artifact_dir is not None:
        settings_kwargs["artifact_root"] = artifact_dir
    if config_dir is not None:
        settings_kwargs["config_dir"] = config_dir

    settings = Settings(**settings_kwargs)
    settings.ensure_directories()

    planner = CompositionPlanner()
    riffusion = RiffusionService(settings)
    musicgen = MusicGenService(settings=settings)
    orchestrator = ComposerOrchestrator(settings, planner, riffusion, musicgen)

    request = GenerationRequest(
        prompt=prompt,
        duration_seconds=duration or settings.default_duration_seconds,
        model_id=model_id or settings.default_model_id,
    )
    request.plan = planner.build_plan(request)
    await orchestrator.warmup(plan=request.plan, model_hint=request.model_id)

    job_id = f"cli-{uuid4()}"
    artifact = await orchestrator.generate(job_id=job_id, request=request)

    extras = artifact.metadata.extras
    placeholder = extras.get("placeholder", False)

    print(f"job_id        : {artifact.job_id}")
    print(f"artifact_path : {artifact.artifact_path}")
    print(f"backend       : {extras.get('backend', 'unknown')}")
    print(f"sample_rate   : {extras.get('sample_rate', 'unknown')}")
    print(f"placeholder   : {placeholder}")
    if placeholder:
        print(f"reason        : {extras.get('placeholder_reason', 'unknown')}")


def main() -> None:
    args = _parse_args()
    asyncio.run(
        _run(
            args.prompt,
            duration=args.duration,
            model_id=args.model_id,
            artifact_dir=args.artifact_dir,
            config_dir=args.config_dir,
        )
    )


if __name__ == "__main__":
    main()
