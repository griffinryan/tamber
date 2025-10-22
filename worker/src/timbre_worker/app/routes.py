from fastapi import APIRouter, HTTPException, Request

from typing import cast

from .models import GenerationArtifact, GenerationRequest, GenerationStatus, JobState
from .jobs import JobManager
from .settings import Settings

router = APIRouter()


def get_job_manager(request: Request) -> JobManager:
    return cast(JobManager, request.app.state.job_manager)


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    settings = cast(Settings, request.app.state.settings)
    return {
        "status": "ok",
        "default_model_id": settings.default_model_id,
        "artifact_root": str(settings.artifact_root),
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
