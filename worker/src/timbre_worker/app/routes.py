from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request

from ..services.planner import PLAN_VERSION
from .jobs import JobManager
from .models import GenerationArtifact, GenerationRequest, GenerationStatus, JobState
from .settings import Settings

router = APIRouter()


def get_job_manager(request: Request) -> JobManager:
    return cast(JobManager, request.app.state.job_manager)


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings = cast(Settings, request.app.state.settings)
    composer = getattr(request.app.state, "composer", None)
    status_map = getattr(request.app.state, "backend_status", {})
    backend_status: dict[str, object] = {}
    if isinstance(status_map, dict):
        for name, status in status_map.items():
            if hasattr(status, "as_dict"):
                backend_status[name] = status.as_dict()  # type: ignore[attr-defined]
            else:
                backend_status[name] = status
    if composer is not None:
        composer_status = composer.backend_status()
        for name, status in composer_status.items():
            backend_status[name] = status.as_dict()

    available_backends = sorted(backend_status.keys())
    warmup_complete = bool(backend_status) and all(
        isinstance(value, dict) and value.get("ready") for value in backend_status.values()
    )
    return {
        "status": "ok",
        "default_model_id": settings.default_model_id,
        "riffusion_default_model_id": settings.riffusion_default_model_id,
        "musicgen_default_model_id": settings.musicgen_default_model_id,
        "artifact_root": str(settings.artifact_root),
        "planner_version": PLAN_VERSION,
        "available_backends": available_backends,
        "backend_status": backend_status,
        "warmup_complete": warmup_complete,
    }


@router.post("/generate", response_model=GenerationStatus)
async def generate(payload: GenerationRequest, request: Request) -> GenerationStatus:
    manager = get_job_manager(request)
    status = await manager.enqueue(payload)
    return status


@router.get("/status/{job_id}", response_model=GenerationStatus)
async def status(job_id: str, request: Request) -> GenerationStatus:
    manager = get_job_manager(request)
    status = await manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@router.get("/artifact/{job_id}", response_model=GenerationArtifact)
async def artifact(job_id: str, request: Request) -> GenerationArtifact:
    manager = get_job_manager(request)
    status = await manager.get_status(job_id)
    if status is None or status.state != JobState.SUCCEEDED:
        raise HTTPException(status_code=404, detail="artifact not available")
    artifact = await manager.get_artifact(job_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not available")
    return artifact
