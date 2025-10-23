#!/usr/bin/env python3
"""
Quick smoke test for the Riffusion backend.

Creates a temporary generation request, waits for completion, and
prints the artifact location along with placeholder metadata so
contributors can verify whether real inference ran or the fallback
placeholder path was used.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "worker" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:
    from timbre_worker.app.models import GenerationRequest
    from timbre_worker.app.settings import Settings
    from timbre_worker.services.riffusion import RiffusionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone Riffusion smoke test.")
    parser.add_argument(
        "--prompt",
        default="dreamy lo-fi piano over gentle rain",
        help="Prompt to feed into the generation pipeline.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Clip duration in seconds (1-30).",
    )
    parser.add_argument(
        "--model-id",
        default="riffusion-v1",
        help="Model identifier registered with the worker.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("~/Music/Timbre").expanduser(),
        help="Directory where generated artifacts should be written.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("~/.config/timbre").expanduser(),
        help="Configuration directory for the worker Settings object.",
    )
    return parser.parse_args()


def ensure_inference_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("torch", "diffusers"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(f"{module_name} (module not found)")

    if missing:
        message = "\n".join(
            [
                "Missing inference dependencies:",
                *[f"  - {item}" for item in missing],
                "Install them with `uv sync --project worker --extra inference` before running the smoke test.",
            ]
        )
        print(message, file=sys.stderr)
        sys.exit(2)


async def run_smoke(args: argparse.Namespace) -> None:
    from timbre_worker.app.models import GenerationRequest
    from timbre_worker.app.settings import Settings
    from timbre_worker.services.riffusion import RiffusionService

    ensure_inference_dependencies()

    settings = Settings(artifact_root=args.artifact_dir, config_dir=args.config_dir)
    settings.ensure_directories()

    service = RiffusionService(settings)

    start = time.perf_counter()
    await service.warmup()
    warmup_elapsed = time.perf_counter() - start

    request = GenerationRequest(
        prompt=args.prompt,
        duration_seconds=args.duration,
        model_id=args.model_id,
    )
    job_id = f"smoke-{uuid4()}"

    start = time.perf_counter()
    artifact = await service.generate(job_id=job_id, request=request)
    generation_elapsed = time.perf_counter() - start

    payload = {
        "job_id": job_id,
        "artifact_path": artifact.artifact_path,
        "metadata": artifact.metadata.model_dump(),
        "warmup_seconds": round(warmup_elapsed, 3),
        "generation_seconds": round(generation_elapsed, 3),
    }
    print(json.dumps(payload, indent=2))

    if artifact.metadata.extras.get("placeholder"):
        reason = artifact.metadata.extras.get("placeholder_reason", "unknown")
        print(f"Placeholder audio emitted (reason={reason})", file=sys.stderr)
        print(
            "The smoke test requires real inference. Ensure torch/diffusers are installed and the model weights are available.",
            file=sys.stderr,
        )
        sys.exit(3)
    else:
        print("Riffusion pipeline generated real audio.", file=sys.stderr)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_smoke(args))
    except KeyboardInterrupt:  # pragma: no cover - operator friendly exit
        print("Cancelled smoke test.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
